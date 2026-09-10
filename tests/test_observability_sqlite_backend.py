"""Tests for observability sqlite run-state and telemetry helpers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from observability import run_state as run_state_module
from observability.run_state import RunStateWriter, _resolve_sqlite_path, load_run_state
from observability.server import (
    ServerConfig,
    _allowed_configs,
    _heartbeat_age_seconds,
    _is_run_state_stale,
    _read_gpu_stats,
    _read_linux_memory_stats,
    _run_state_path_warnings,
)


class TestObservabilitySqliteRunState(unittest.TestCase):
    def test_resolve_sqlite_path_rejects_non_sqlite_uri(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_sqlite_path("tmp/observability/run_state.json")

    def test_get_run_state_writer_invalid_env_path_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = str(Path(tmp_dir) / "run_state.json")
            with mock.patch.dict(os.environ, {"RUN_STATE_PATH": invalid_path}, clear=False):
                with mock.patch("observability.run_state._WRITER", None):
                    writer = run_state_module.get_run_state_writer()

        self.assertIsNone(writer)

    def test_writer_persists_and_loads_state_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            writer = RunStateWriter(db_uri)
            writer.start(run_id="run-xyz")
            writer.set_stage("trial")
            state = load_run_state(db_uri)

        self.assertIsInstance(state, dict)
        assert state is not None
        self.assertEqual(state.get("run_id"), "run-xyz")
        self.assertEqual(state.get("stage"), "trial")
        self.assertEqual(state.get("status"), "running")

    def test_writer_records_process_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            run_log_path = Path(tmp_dir) / "run.log"
            with mock.patch.dict(os.environ, {"RUN_LOG_PATH": str(run_log_path)}, clear=False):
                writer = RunStateWriter(db_uri)
                writer.start(run_id="run-pid")
            state = load_run_state(db_uri)

        self.assertIsInstance(state, dict)
        assert state is not None
        self.assertGreater(int(state.get("run_process_pid") or 0), 0)
        self.assertEqual(state.get("run_state_path"), db_uri)
        self.assertEqual(state.get("run_log_path"), str(run_log_path))

    def test_writer_start_prefers_explicit_run_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            explicit_run_log_path = Path(tmp_dir) / "explicit.log"
            env_run_log_path = Path(tmp_dir) / "env.log"
            with mock.patch.dict(os.environ, {"RUN_LOG_PATH": str(env_run_log_path)}, clear=False):
                writer = RunStateWriter(db_uri)
                writer.start(run_id="run-explicit", run_log_path=str(explicit_run_log_path))
            state = load_run_state(db_uri)

        self.assertIsInstance(state, dict)
        assert state is not None
        self.assertEqual(state.get("run_log_path"), str(explicit_run_log_path))

    def test_writer_updates_hpo_wave_memory_and_watchdog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            writer = RunStateWriter(db_uri)
            writer.start(run_id="run-hpo")
            writer.update_hpo_wave_memory({111: 1024, 222: 4096, 333: 2048})
            writer.mark_hpo_rss_watchdog_trigger(pid=222, rss_bytes=4096, limit_bytes=3072)
            state = load_run_state(db_uri)

        self.assertIsInstance(state, dict)
        assert state is not None
        self.assertEqual(int(state.get("hpo_wave_worker_count") or 0), 3)
        self.assertEqual(int(state.get("hpo_wave_worker_rss_current_bytes") or 0), 4096)
        self.assertEqual(int(state.get("hpo_wave_worker_rss_max_bytes") or 0), 4096)
        top = state.get("hpo_wave_worker_rss_top")
        self.assertIsInstance(top, list)
        assert isinstance(top, list)
        self.assertGreaterEqual(len(top), 1)
        self.assertEqual(int(top[0].get("pid") or 0), 222)
        self.assertEqual(int(top[0].get("rss_bytes") or 0), 4096)
        self.assertEqual(int(state.get("hpo_rss_watchdog_trigger_count") or 0), 1)
        self.assertEqual(int(state.get("hpo_rss_watchdog_last_trigger_pid") or 0), 222)

    def test_load_run_state_ttl_cache_reduces_sqlite_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            fake_time = [1000.0]

            with mock.patch("observability.run_state.time.time", side_effect=lambda: fake_time[0]):
                writer = RunStateWriter(db_uri)
                writer.start(run_id="run-cache-test")

                run_state_module._RUN_STATE_CACHE.clear()
                run_state_module._RUN_STATE_CACHE_TIME.clear()

                state1 = load_run_state(db_uri)
                self.assertIsInstance(state1, dict)
                assert state1 is not None
                self.assertEqual(state1.get("run_id"), "run-cache-test")
                initial_stage = state1.get("stage")

                writer.set_stage("training")

                state2 = load_run_state(db_uri)
                self.assertIsInstance(state2, dict)
                assert state2 is not None
                self.assertEqual(state2.get("stage"), initial_stage)

                fake_time[0] += 2.1

                state3 = load_run_state(db_uri)
                self.assertIsInstance(state3, dict)
                assert state3 is not None
                self.assertEqual(state3.get("stage"), "training")

    def test_load_run_state_ttl_configurable_via_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_uri = f"sqlite:///{tmp_dir}/run_state.db"
            fake_time = [2000.0]

            with mock.patch("observability.run_state.time.time", side_effect=lambda: fake_time[0]):
                writer = RunStateWriter(db_uri)
                writer.start(run_id="run-config-test")

                with mock.patch.dict(os.environ, {"RUN_STATE_CACHE_TTL_SECONDS": "0.5"}, clear=False):
                    run_state_module._RUN_STATE_CACHE.clear()
                    run_state_module._RUN_STATE_CACHE_TIME.clear()

                    state1 = load_run_state(db_uri)
                    self.assertIsInstance(state1, dict)
                    assert state1 is not None
                    initial_stage = state1.get("stage")

                    writer.set_stage("evaluation")

                    state2 = load_run_state(db_uri)
                    self.assertIsInstance(state2, dict)
                    assert state2 is not None
                    self.assertEqual(state2.get("stage"), initial_stage)

                    fake_time[0] += 0.6

                    state3 = load_run_state(db_uri)
                    self.assertIsInstance(state3, dict)
                    assert state3 is not None
                    self.assertEqual(state3.get("stage"), "evaluation")


class TestObservabilityServerConfigValidation(unittest.TestCase):
    def test_from_sources_rejects_non_sqlite_run_state_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "observability.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        'host: "127.0.0.1"',
                        "port: 8008",
                        'run_state_path: "tmp/observability/run_state.json"',
                        'run_log_path: "tmp/dashboard/run.log"',
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "OBSERVABILITY_USER": "user",
                    "OBSERVABILITY_PASSWORD": "pass",
                    "RUN_STATE_PATH": "",
                },
                clear=False,
            ):
                with self.assertRaises(RuntimeError):
                    ServerConfig.from_sources(config_path=str(cfg_path))


class TestGpuStatsParsing(unittest.TestCase):
    def test_read_gpu_stats_parses_nvidia_smi_output(self) -> None:
        completed = types.SimpleNamespace(
            stdout="0, RTX3090, 67, 12000, 24576, 240, 350\n",
        )
        with mock.patch("observability.server.subprocess.run", return_value=completed):
            stats = _read_gpu_stats()

        self.assertEqual(len(stats), 1)
        gpu = stats[0]
        self.assertEqual(gpu.get("name"), "RTX3090")
        utilization_ratio = gpu.get("utilization_ratio")
        memory_used_ratio = gpu.get("memory_used_ratio")
        if not isinstance(utilization_ratio, (int, float)):
            self.fail("utilization_ratio should be numeric")
        if not isinstance(memory_used_ratio, (int, float)):
            self.fail("memory_used_ratio should be numeric")
        self.assertAlmostEqual(float(utilization_ratio), 0.67, places=6)
        self.assertAlmostEqual(float(memory_used_ratio), 12000.0 / 24576.0, places=6)

    def test_read_gpu_stats_skips_malformed_output(self) -> None:
        completed = types.SimpleNamespace(
            stdout="bad,line\n1,TooShort\n",
        )
        with mock.patch("observability.server.subprocess.run", return_value=completed):
            stats = _read_gpu_stats()

        self.assertEqual(stats, [])


class TestMemoryAndStaleHelpers(unittest.TestCase):
    def test_read_linux_memory_stats_handles_malformed_meminfo(self) -> None:
        with mock.patch("observability.server.Path.exists", return_value=True), mock.patch(
            "observability.server.Path.read_text",
            return_value="MemTotal: not_a_number kB\nMemAvailable: ??? kB\n",
        ):
            stats = _read_linux_memory_stats()

        self.assertIsNone(stats.get("mem_total_bytes"))
        self.assertIsNone(stats.get("mem_available_bytes"))
        self.assertIsNone(stats.get("mem_used_bytes"))
        self.assertIsNone(stats.get("mem_used_ratio"))

    def test_stale_helpers_behave_for_recent_and_stale_heartbeat(self) -> None:
        now = 1_000.0
        with mock.patch("observability.server.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.timestamp.return_value = now
            stale_state = {"heartbeat_time": now - 31.0}
            fresh_state = {"heartbeat_time": now - 5.0}

            stale_age = _heartbeat_age_seconds(stale_state.get("heartbeat_time"))
            fresh_age = _heartbeat_age_seconds(fresh_state.get("heartbeat_time"))

            self.assertIsNotNone(stale_age)
            self.assertIsNotNone(fresh_age)
            self.assertTrue(_is_run_state_stale(stale_state))
            self.assertFalse(_is_run_state_stale(fresh_state))


class TestRunStatePathDiagnostics(unittest.TestCase):
    def test_run_state_path_warnings_reports_divergent_paths(self) -> None:
        config = ServerConfig(
            user="user",
            password="pass",
            run_state_path="sqlite:///tmp/observability/server.db",
            run_log_path="tmp/observability/server.log",
            host="127.0.0.1",
            port=8008,
            allow_run_control=False,
            allowed_configs_glob="config/e2e_trial_*.yaml",
            static_dir="static",
            tail_max_lines=200,
            config_path="",
            file_config={},
        )
        state = {
            "run_state_path": "sqlite:///tmp/observability/writer.db",
            "run_log_path": "tmp/observability/writer.log",
        }

        warnings = _run_state_path_warnings(state, config)

        self.assertEqual(len(warnings), 2)
        self.assertIn("Run-state path mismatch", warnings[0])
        self.assertIn("Run-log path mismatch", warnings[1])

    def test_allowed_configs_accepts_absolute_glob_under_config_root(self) -> None:
        absolute_glob = str((Path("config").resolve() / "e2e_trial_*.yaml"))

        allowed = _allowed_configs(absolute_glob)

        self.assertIn("config/e2e_trial_13_snapshot_hpo_trial.yaml", allowed)


if __name__ == "__main__":
    unittest.main()
