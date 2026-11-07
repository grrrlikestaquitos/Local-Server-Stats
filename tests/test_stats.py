"""Tests of the stats daemon at its edges: no NVIDIA card, no RAPL, no vLLM gateway, no key file in the
checkout, and the database in a temporary folder. Set STATS_API_FILE to run them against another copy of
server_stats_api.py (that is how they were shown to fail against the old code)."""
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("STATS_API_FILE") or os.path.join(os.path.dirname(HERE), "server_stats_api.py")
KEY = "test-key-not-a-real-one"


def load(tmp):
    """Import the daemon afresh with its key file and database inside tmp."""
    keyfile = os.path.join(tmp, "token")
    with open(keyfile, "w") as f:
        f.write(KEY + "\n")
    with mock.patch.dict(os.environ, {"STATS_TOKEN_FILE": keyfile}):
        spec = importlib.util.spec_from_file_location("server_stats_api_under_test", SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    mod.DB_PATH = os.path.join(tmp, "power_tracker.db")
    return mod


class Card:
    gpu = 37


class Mem:
    used = 4 * 1024 ** 3
    total = 16 * 1024 ** 3


class StatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.m = load(self.tmp)
        # The file an existing machine has: an energy table already holding a row.
        with sqlite3.connect(self.m.DB_PATH) as c:
            c.execute("CREATE TABLE energy_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, cumulative_kwh REAL DEFAULT 0.0,"
                      " cumulative_cost REAL DEFAULT 0.0, month_cumulative_kwh REAL DEFAULT 0.0,"
                      " month_cumulative_cost REAL DEFAULT 0.0, current_month_integer INTEGER, last_updated REAL)")
            c.execute("INSERT INTO energy_metrics VALUES (1, 12.5, 1.63, 3.0, 0.39, ?, ?)",
                      (time.localtime().tm_mon, time.time() - 3600))
        self.energy_before = self.energy_rows()
        self.m.init_db()
        self.m.db_connection = sqlite3.connect(self.m.DB_PATH, check_same_thread=False)
        self.addCleanup(self.m.db_connection.close)
        # One card, as nvml would describe it.
        nv = {
            "HAS_NVIDIA": True,
            "nvmlDeviceGetCount": lambda: 1,
            "nvmlDeviceGetHandleByIndex": lambda i: i,
            "nvmlDeviceGetUtilizationRates": lambda h: Card,
            "nvmlDeviceGetPowerUsage": lambda h: 251500,
            "nvmlDeviceGetTemperature": lambda h, k: 61,
            "nvmlDeviceGetMemoryInfo": lambda h: Mem,
            "nvmlDeviceGetName": lambda h: b"RTX 5070 Ti",
            "NVML_TEMPERATURE_GPU": 0,
        }
        for name, value in nv.items():
            p = mock.patch.object(self.m, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)

    def energy_rows(self):
        with sqlite3.connect(self.m.DB_PATH) as c:
            return c.execute("SELECT * FROM energy_metrics").fetchall()

    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(self.m.app)  # no `with`: the startup event, and so the loop, is not run

    def get(self, key=KEY):
        headers = {"X-Server-Token": key} if key is not None else {}
        return self.client().get("/stats", headers=headers)

    def test_sampled_at_is_a_number_from_the_collection_not_the_request(self):
        with mock.patch.object(self.m.time, "time", return_value=1_700_000_000.5):
            self.m.collect_all_system_metrics()
        first = self.get().json()
        time.sleep(0.05)
        second = self.get().json()
        self.assertIn("sampled_at", first)
        self.assertIsInstance(first["sampled_at"], float)
        self.assertEqual(first["sampled_at"], 1_700_000_000.5)
        self.assertEqual(second["sampled_at"], first["sampled_at"])  # a later request does not move it
        before = time.time()
        self.m.collect_all_system_metrics()
        after = time.time()
        self.assertTrue(before <= self.get().json()["sampled_at"] <= after)  # a later collection does

    def test_answer_has_no_energy(self):
        self.m.collect_all_system_metrics()
        body = self.get().json()
        self.assertNotIn("energy", body)
        text = self.get().text
        for word in ("current_total_watts", "lifetime_cumulative", "current_month", "kwh", "cost"):
            self.assertNotIn(word, text)

    def test_loop_leaves_energy_metrics_alone(self):
        self.assertEqual(len(self.energy_before), 1)
        self.m.collect_all_system_metrics()
        self.m.collect_all_system_metrics()
        self.assertEqual(self.energy_rows(), self.energy_before)

    def test_fresh_database_is_given_no_energy_table(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        m = load(tmp)
        m.init_db()
        with sqlite3.connect(m.DB_PATH) as c:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("energy_metrics", tables)
        self.assertLessEqual({"token_metrics", "token_backend_state"}, tables)

    def test_each_cards_own_figures_remain(self):
        self.m.collect_all_system_metrics()
        body = self.get().json()
        self.assertTrue(body["gpu"]["available"])
        card = body["gpu"]["devices"][0]
        self.assertEqual(card["power_watts"], 251.5)
        self.assertEqual(card["usage_percent"], 37)
        self.assertEqual(card["temp_celsius"], 61)
        self.assertEqual(card["vram_total_gb"], 16.0)
        for key in ("cpu", "ram", "server_uptime_seconds", "status", "tokens"):
            self.assertIn(key, body)

    def test_token_counts_are_kept_in_the_database_and_served(self):
        text = ("# backend m1\nvllm:prompt_tokens_total 100\nvllm:generation_tokens_total 40\n"
                "# backend m2\nvllm:prompt_tokens_total 7\nvllm:generation_tokens_total 3\n")
        reply = mock.Mock(status_code=200, text=text)
        with mock.patch.object(self.m.requests, "get", return_value=reply):
            self.m.update_vllm_token_metrics()
        self.m.collect_all_system_metrics()
        self.assertEqual(self.get().json()["tokens"], {"prompt": 107, "generation": 43})
        with sqlite3.connect(self.m.DB_PATH) as c:
            self.assertEqual(c.execute("SELECT prompt_tokens, generation_tokens FROM token_metrics").fetchall(), [(107, 43)])
        self.assertEqual(self.energy_rows(), self.energy_before)

    def test_key_is_required(self):
        self.m.collect_all_system_metrics()
        self.assertEqual(self.get(key=None).status_code, 403)
        self.assertEqual(self.get(key="wrong").status_code, 403)
        self.assertEqual(self.get().status_code, 200)

    def test_no_key_file_is_written_into_the_checkout(self):
        self.assertTrue(self.m.TOKEN_PATH.startswith(self.tmp))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(HERE), "power_tracker.db")))


if __name__ == "__main__":
    unittest.main()
