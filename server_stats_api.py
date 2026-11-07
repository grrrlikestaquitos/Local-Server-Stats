import os
import logging
import secrets
import psutil
import time
import sqlite3
import asyncio
import copy
import re
import requests
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

SERVER_START_TIME = time.time()
app = FastAPI(title="Advanced Server Stats API")

log = logging.getLogger("server-stats")

TOKEN_PATH = os.getenv("STATS_TOKEN_FILE") or os.path.join(
    os.path.expanduser("~"), ".config", "server-stats", "token"
)


def _load_or_create_token() -> str:
    """Read the API token, generating a 0600 one on first run."""
    try:
        with open(TOKEN_PATH) as f:
            tok = f.read().strip()
        if tok:
            return tok
    except FileNotFoundError:
        pass
    tok = secrets.token_urlsafe(24)
    os.makedirs(os.path.dirname(TOKEN_PATH), mode=0o700, exist_ok=True)
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    log.warning("generated new API token at %s", TOKEN_PATH)
    return tok


REQUIRE_AUTH = os.getenv("STATS_REQUIRE_AUTH", "1").lower() not in ("0", "false", "no")
API_KEY = _load_or_create_token()
API_KEY_NAME = "X-Server-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

DB_PATH = os.path.join(os.path.dirname(__file__), "power_tracker.db")
BACKGROUND_INTERVAL_SECONDS = 5  
# Opt-in only. The vLLM gateway launches backends on dynamic ports, so there is
# no fixed URL to scrape; prefer the gateway's own telemetry. Set explicitly to
# re-enable against a known backend.
# The vLLM gateway aggregates whatever backends are live; scraping a
# backend port directly is unreliable since they only exist while loaded.
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL") or "http://127.0.0.1:8080/metrics"


# Global persistent database connection reference
db_connection = None

# Serves as our working state manager
_internal_metrics = {
    "status": "online",
    # Seconds since the epoch at which the loop finished collecting the figures below.
    "sampled_at": None,
    "server_uptime_seconds": 0,
    "cpu": {
        "model_name": "Unknown Processor",
        "usage_percent": 0.0,
        "cores_physical": 0,
        "cores_logical": 0,
        "temp_celsius": "N/A"
    },
    "ram": {
        "usage_percent": 0.0,
        "used_gb": 0.0,
        "total_gb": 0.0
    },
    "gpu": {
        "available": False,
        "devices": []
    },
    "tokens": {
        "prompt": 0,
        "generation": 0
    }
}

# The true read-only public mirror cache (Provides 100% thread safety)
_public_stats_cache = copy.deepcopy(_internal_metrics)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_tokens INTEGER DEFAULT 0,
                generation_tokens INTEGER DEFAULT 0,
                last_updated REAL
            )
        ''')

        
        # Per-backend high-water marks. The gateway's aggregate is a sum over
        # *currently loaded* backends, so it drops when one unloads and restarts
        # at zero when one reloads. Tracking each backend separately lets us
        # accumulate real deltas instead of trusting a non-monotonic total.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_backend_state (
                backend TEXT PRIMARY KEY,
                last_prompt REAL DEFAULT 0.0,
                last_generation REAL DEFAULT 0.0,
                last_seen REAL
            )
        ''')

        conn.commit()

@app.on_event("startup")
async def startup_event():
    global db_connection
    init_db()
    
    # Establish single persistent connection object
    db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    _internal_metrics["cpu"]["model_name"] = get_cpu_model_name()
    _internal_metrics["cpu"]["cores_physical"] = psutil.cpu_count(logical=False)
    _internal_metrics["cpu"]["cores_logical"] = psutil.cpu_count(logical=True)
    _internal_metrics["gpu"]["available"] = HAS_NVIDIA
    
    update_token_cache_from_db()
    psutil.cpu_percent(interval=None)
    
    asyncio.create_task(metrics_collection_worker())

@app.on_event("shutdown")
def shutdown_event():
    global db_connection
    if db_connection:
        db_connection.close()
    if HAS_NVIDIA:
        try:
            from pynvml import nvmlShutdown
            nvmlShutdown() # Clean driver teardown preventing orphaned memory states
        except Exception:
            pass

def update_token_cache_from_db():
    cursor = db_connection.cursor()
    cursor.execute('SELECT prompt_tokens, generation_tokens FROM token_metrics WHERE id = 1')
    row = cursor.fetchone()
    if row:
        _internal_metrics["tokens"]["prompt"] = row[0]
        _internal_metrics["tokens"]["generation"] = row[1]

_BACKEND_SPLIT_RE = re.compile(r'^# backend (\S+).*$', re.M)
_PROMPT_TOTAL_RE = re.compile(r'^vllm:prompt_tokens_total(?:\{[^}]*\})?\s+([\d.eE+-]+)', re.M)
_GEN_TOTAL_RE = re.compile(r'^vllm:generation_tokens_total(?:\{[^}]*\})?\s+([\d.eE+-]+)', re.M)
_PROMPT_SUM_RE = re.compile(r'^vllm:prompt_tokens_sum\s+([\d.eE+-]+)', re.M)
_GEN_SUM_RE = re.compile(r'^vllm:generation_tokens_sum\s+([\d.eE+-]+)', re.M)


def _sum_floats(pattern, text):
    total = 0.0
    for m in pattern.findall(text):
        try:
            total += float(m)
        except ValueError:
            pass
    return total


def parse_backend_tokens(text):
    """Map backend id -> (prompt, generation) from the gateway's /metrics.

    The gateway emits one `# backend <model_id>` chunk per live backend, then an
    aggregate. We prefer the per-backend chunks because each one is genuinely
    monotonic for as long as that backend lives. The aggregate is only a
    fallback for when nothing is loaded (or an older gateway), and is keyed
    separately so it can never be mistaken for a real backend.
    """
    parts = _BACKEND_SPLIT_RE.split(text)
    out = {}
    if len(parts) > 1:
        # parts == [preamble, id1, body1, id2, body2, ...]
        for i in range(1, len(parts) - 1, 2):
            body = parts[i + 1]
            out[parts[i]] = (_sum_floats(_PROMPT_TOTAL_RE, body),
                             _sum_floats(_GEN_TOTAL_RE, body))
    if out:
        return out
    pm, gm = _PROMPT_SUM_RE.search(text), _GEN_SUM_RE.search(text)
    if pm and gm:
        return {"__aggregate__": (float(pm.group(1)), float(gm.group(1)))}
    return {}


def update_vllm_token_metrics():
    global _internal_metrics
    if not VLLM_METRICS_URL:
        return
    try:
        response = requests.get(VLLM_METRICS_URL, timeout=2)
        if response.status_code != 200:
            return
        current = parse_backend_tokens(response.text)
        if not current:
            return

        cursor = db_connection.cursor()
        cursor.execute('SELECT backend, last_prompt, last_generation FROM token_backend_state')
        previous = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        delta_prompt = delta_gen = 0.0
        now = time.time()
        for backend, (prompt, gen) in current.items():
            last_prompt, last_gen = previous.get(backend, (0.0, 0.0))
            # A reload restarts the backend's counter at zero, so a decrease
            # means "restarted", not "negative usage" - count the new value
            # whole rather than differencing against a stale high-water mark.
            delta_prompt += (prompt - last_prompt) if prompt >= last_prompt else prompt
            delta_gen += (gen - last_gen) if gen >= last_gen else gen
            cursor.execute('''
                INSERT INTO token_backend_state (backend, last_prompt, last_generation, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(backend) DO UPDATE SET
                    last_prompt=excluded.last_prompt,
                    last_generation=excluded.last_generation,
                    last_seen=excluded.last_seen
            ''', (backend, prompt, gen, now))

        # Rows for vanished backends are deliberately kept: the lifetime total
        # already includes their tokens, and the high-water mark is what lets a
        # later reload be recognised as a restart.
        if delta_prompt or delta_gen:
            _internal_metrics["tokens"]["prompt"] += int(delta_prompt)
            _internal_metrics["tokens"]["generation"] += int(delta_gen)
            cursor.execute('''
                INSERT INTO token_metrics (id, prompt_tokens, generation_tokens, last_updated)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    prompt_tokens=excluded.prompt_tokens,
                    generation_tokens=excluded.generation_tokens,
                    last_updated=excluded.last_updated
            ''', (_internal_metrics["tokens"]["prompt"],
                   _internal_metrics["tokens"]["generation"], now))
        db_connection.commit()
    except Exception as e:
        # Never crash the telemetry loop, but do not fail invisibly either.
        log.debug("vLLM metrics scrape failed (%s): %s", VLLM_METRICS_URL, e)

def collect_all_system_metrics():
    global _public_stats_cache
    
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    
    cpu_temp = "N/A"
    try:
        temps = psutil.sensors_temperatures()
        for label in ["k10temp", "zenpower", "coretemp", "cpu_thermal"]:
            if label in temps and temps[label]:
                cpu_temp = round(temps[label][0].current)
                break
    except Exception:
        pass

    _internal_metrics["server_uptime_seconds"] = int(time.time() - SERVER_START_TIME)
    _internal_metrics["cpu"]["usage_percent"] = cpu_percent
    _internal_metrics["cpu"]["temp_celsius"] = cpu_temp
    _internal_metrics["ram"]["usage_percent"] = ram.percent
    _internal_metrics["ram"]["used_gb"] = round(ram.used / (1024**3), 2)
    _internal_metrics["ram"]["total_gb"] = round(ram.total / (1024**3), 2)
    
    if HAS_NVIDIA:
        try:
            device_count = nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                power = nvmlDeviceGetPowerUsage(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                mem = nvmlDeviceGetMemoryInfo(handle) 

                raw_name = nvmlDeviceGetName(handle)
                model_name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else raw_name
                
                gpus.append({
                    "index": i,
                    "model_name": model_name,
                    "usage_percent": util.gpu,
                    "power_watts": round(power / 1000.0, 2),
                    "temp_celsius": temp,
                    "vram_used_gb": round(mem.used / (1024**3), 2),
                    "vram_total_gb": round(mem.total / (1024**3), 2),
                    "vram_percent": round((mem.used / mem.total) * 100, 1)
                })
            
            _internal_metrics["gpu"]["available"] = True
            _internal_metrics["gpu"]["devices"] = gpus
        except Exception as e:
            _internal_metrics["gpu"]["error"] = str(e)
            
    _internal_metrics["sampled_at"] = time.time()

    # Atomic swap assignment: Creates an isolated point-in-time state for API consumers
    _public_stats_cache = copy.deepcopy(_internal_metrics)

async def metrics_collection_worker():
    while True:
        try:
            # Update tokens first, then system stats
            await asyncio.to_thread(update_vllm_token_metrics)
            await asyncio.to_thread(collect_all_system_metrics)
        except Exception as e:
            print(f"Error executing background telemetry loop: {e}")
        await asyncio.sleep(BACKGROUND_INTERVAL_SECONDS)

def get_api_key(header_value: str = Security(api_key_header)):
    if REQUIRE_AUTH and not secrets.compare_digest(header_value or "", API_KEY):
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return header_value

def get_cpu_model_name():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown Processor"

try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetPowerUsage, nvmlDeviceGetName, nvmlDeviceGetTemperature,
        nvmlDeviceGetMemoryInfo, NVML_TEMPERATURE_GPU
    )
    nvmlInit()
    HAS_NVIDIA = True
except Exception:
    HAS_NVIDIA = False

@app.get("/stats", dependencies=[Security(get_api_key)])
def get_stats():
    # Serves the safe cloned version instantly
    return _public_stats_cache

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
