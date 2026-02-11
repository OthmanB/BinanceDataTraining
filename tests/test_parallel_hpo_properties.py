"""Property-based tests for parallel HPO and sequential helpers."""

from __future__ import annotations

from datetime import date, timedelta
import os
import re
import tempfile
import unittest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

from models.hyperparameter_tuning import (
    _allocate_trials_to_workers,
    _apply_worker_resource,
    _compute_next_batch_size,
    _select_wave_resources,
    _should_trigger_rss_watchdog,
)
from training.pipeline import (
    _aggregate_hpo_window_metrics,
    _cleanup_completed_window_dirs,
    _resolve_sequential_resume_paths,
    _resolve_sequential_windows,
)


_MAX_EXAMPLES = 10 if os.environ.get("CI") else 50


def _date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestParallelAllocationProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        n_trials=st.integers(min_value=1, max_value=200),
        n_workers=st.integers(min_value=1, max_value=32),
    )
    def test_allocate_trials_invariants(self, n_trials: int, n_workers: int) -> None:
        allocations = _allocate_trials_to_workers(n_trials, n_workers)
        self.assertEqual(len(allocations), n_workers)
        self.assertEqual(sum(allocations), n_trials)
        self.assertTrue(all(value >= 0 for value in allocations))
        self.assertLessEqual(max(allocations) - min(allocations), 1)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestWorkerResourceProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(resource=st.text(min_size=0, max_size=30))
    def test_resource_parser_accepts_only_expected_forms(self, resource: str) -> None:
        config = {"training": {"runtime": {"device": "gpu", "gpu_visible_devices": "0"}}}
        normalized = resource.strip().lower()

        cpu_valid = normalized == "cpu"
        gpu_match = re.match(r"^gpu:(.*)$", normalized)
        gpu_valid = bool(gpu_match and gpu_match.group(1).strip())
        valid = cpu_valid or gpu_valid

        if not valid:
            with self.assertRaises(ValueError):
                _apply_worker_resource(config, resource)
            return

        updated = _apply_worker_resource(config, resource)
        runtime = updated["training"]["runtime"]
        if cpu_valid:
            self.assertEqual(runtime["device"], "cpu")
            self.assertIsNone(runtime["gpu_visible_devices"])
        else:
            self.assertEqual(runtime["device"], "gpu")
            self.assertTrue(str(runtime["gpu_visible_devices"]).strip())


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestRegimeBackoffProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        current_batch=st.integers(min_value=2, max_value=1024),
        min_batch=st.integers(min_value=1, max_value=128),
        factor=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    def test_next_batch_size_monotonic_and_bounded(
        self,
        current_batch: int,
        min_batch: int,
        factor: float,
    ) -> None:
        if min_batch > current_batch:
            min_batch = current_batch
        next_batch = _compute_next_batch_size(current_batch, factor, min_batch)
        self.assertGreaterEqual(next_batch, min_batch)
        self.assertLessEqual(next_batch, current_batch)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestRssWatchdogProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        rss=st.integers(min_value=1, max_value=10_000_000),
        limit=st.integers(min_value=1, max_value=1_000_000),
        elapsed=st.floats(min_value=0.0, max_value=59.0, allow_nan=False, allow_infinity=False),
        grace=st.floats(min_value=60.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    def test_watchdog_never_triggers_before_grace(
        self,
        rss: int,
        limit: int,
        elapsed: float,
        grace: float,
    ) -> None:
        should_trigger, _, _ = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={111: max(rss, limit + 1)},
            rss_watchdog_limit_bytes=limit,
            wave_started_monotonic=100.0,
            now_monotonic=100.0 + elapsed,
            startup_grace_seconds=grace,
            require_trial_start=False,
            trial_started_in_wave=False,
            startup_timeout_seconds=1800.0,
        )
        self.assertFalse(should_trigger)

    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        grace=st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False),
        timeout=st.floats(min_value=31.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    def test_watchdog_respects_trial_start_gate_before_timeout(
        self,
        grace: float,
        timeout: float,
    ) -> None:
        elapsed = (grace + timeout) / 2.0
        should_trigger, _, _ = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={222: 9_999_999},
            rss_watchdog_limit_bytes=1,
            wave_started_monotonic=200.0,
            now_monotonic=200.0 + elapsed,
            startup_grace_seconds=grace,
            require_trial_start=True,
            trial_started_in_wave=False,
            startup_timeout_seconds=timeout,
        )
        self.assertFalse(should_trigger)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestWaveResourceSelectionProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        resources=st.lists(st.sampled_from(["gpu:0", "gpu:1", "cpu"]), min_size=1, max_size=5),
        remaining_trials=st.integers(min_value=0, max_value=20),
        force_single=st.booleans(),
    )
    def test_wave_resource_selection_invariants(
        self,
        resources: list[str],
        remaining_trials: int,
        force_single: bool,
    ) -> None:
        selected = _select_wave_resources(resources, remaining_trials, force_single)
        self.assertLessEqual(len(selected), len(resources))
        self.assertLessEqual(len(selected), max(0, remaining_trials))
        if remaining_trials <= 0:
            self.assertEqual(selected, [])
        if force_single and remaining_trials > 0:
            self.assertLessEqual(len(selected), 1)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestSequentialResumePathProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        root_name=st.text(min_size=1, max_size=20),
        snapshot_name=st.text(min_size=1, max_size=20),
        asset=st.text(min_size=1, max_size=12),
        start_day=st.integers(min_value=1, max_value=20),
        span_days=st.integers(min_value=1, max_value=10),
    )
    def test_resume_paths_stable_and_change_with_windows(
        self,
        root_name: str,
        snapshot_name: str,
        asset: str,
        start_day: int,
        span_days: int,
    ) -> None:
        start_dt = date(2024, 1, start_day)
        end_dt = start_dt + timedelta(days=span_days)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "snapshot": {
                    "enabled": True,
                    "directory": tmpdir,
                    "root_name": root_name,
                    "name": snapshot_name,
                },
                "data": {
                    "asset_pairs": {"target_asset": asset},
                    "time_range": {
                        "start_date": _date_str(start_dt),
                        "end_date": _date_str(end_dt),
                    },
                },
            }

            windows_a = [(_date_str(start_dt), _date_str(start_dt + timedelta(days=1)))]
            windows_b = windows_a + [(_date_str(end_dt), _date_str(end_dt))]

            path_a_state, path_a_model = _resolve_sequential_resume_paths(config, windows_a)
            path_b_state, path_b_model = _resolve_sequential_resume_paths(config, windows_a)
            self.assertEqual(path_a_state, path_b_state)
            self.assertEqual(path_a_model, path_b_model)

            changed_state, changed_model = _resolve_sequential_resume_paths(config, windows_b)
            self.assertNotEqual(path_a_state, changed_state)
            self.assertNotEqual(path_a_model, changed_model)

    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        namespace_a=st.from_regex(r"[A-Za-z0-9_-]{1,30}", fullmatch=True),
        namespace_b=st.from_regex(r"[A-Za-z0-9_-]{1,30}", fullmatch=True),
    )
    def test_resume_paths_change_with_namespace(self, namespace_a: str, namespace_b: str) -> None:
        if namespace_a == namespace_b:
            namespace_b = namespace_b + "-alt"

        windows = [("2024-01-01", "2024-01-02")]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "snapshot": {
                    "enabled": True,
                    "directory": tmpdir,
                    "root_name": "root",
                    "name": "snap",
                },
                "data": {
                    "asset_pairs": {"target_asset": "BTCUSDT"},
                    "time_range": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-02",
                    },
                },
                "training": {
                    "sequential_training": {
                        "enabled": True,
                        "resume_enabled": True,
                        "resume_namespace": namespace_a,
                    }
                },
            }

            state_a, model_a = _resolve_sequential_resume_paths(config, windows)
            config["training"]["sequential_training"]["resume_namespace"] = namespace_b
            state_b, model_b = _resolve_sequential_resume_paths(config, windows)

            self.assertNotEqual(state_a, state_b)
            self.assertNotEqual(model_a, model_b)

    @settings(max_examples=_MAX_EXAMPLES)
    @given(namespace=st.from_regex(r"[A-Za-z0-9_-]{1,30}", fullmatch=True))
    def test_resume_paths_deterministic_for_same_namespace(self, namespace: str) -> None:
        windows = [("2024-01-01", "2024-01-03")]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "snapshot": {
                    "enabled": True,
                    "directory": tmpdir,
                    "root_name": "root",
                    "name": "snap",
                },
                "data": {
                    "asset_pairs": {"target_asset": "BTCUSDT"},
                    "time_range": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-03",
                    },
                },
                "training": {
                    "sequential_training": {
                        "enabled": True,
                        "resume_enabled": True,
                        "resume_namespace": namespace,
                    }
                },
            }

            state_a, model_a = _resolve_sequential_resume_paths(config, windows)
            state_b, model_b = _resolve_sequential_resume_paths(config, windows)

            self.assertEqual(state_a, state_b)
            self.assertEqual(model_a, model_b)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestSequentialWindowGenerationProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        start_day=st.integers(min_value=1, max_value=20),
        duration_days=st.integers(min_value=0, max_value=10),
        window_days=st.integers(min_value=1, max_value=5),
    )
    def test_resolved_windows_are_contiguous_and_cover_range(
        self,
        start_day: int,
        duration_days: int,
        window_days: int,
    ) -> None:
        start_dt = date(2024, 1, start_day)
        end_dt = start_dt + timedelta(days=duration_days)

        config = {
            "data": {
                "time_range": {
                    "start_date": _date_str(start_dt),
                    "end_date": _date_str(end_dt),
                }
            },
            "training": {
                "sequential_training": {
                    "enabled": True,
                    "window_days": window_days,
                }
            },
        }

        windows = _resolve_sequential_windows(config)
        assert windows is not None
        self.assertGreaterEqual(len(windows), 1)
        self.assertEqual(windows[0][0], _date_str(start_dt))
        self.assertEqual(windows[-1][1], _date_str(end_dt))

        for idx in range(len(windows) - 1):
            current_end = date.fromisoformat(windows[idx][1])
            next_start = date.fromisoformat(windows[idx + 1][0])
            self.assertEqual(next_start, current_end + timedelta(days=1))


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestHPOAggregationProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        pairs=st.lists(
            st.tuples(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=25,
        )
    )
    def test_aggregated_metric_stays_within_input_bounds(self, pairs: list[tuple[float, float]]) -> None:
        result = _aggregate_hpo_window_metrics(pairs)
        assert result is not None
        values = [value for value, _ in pairs]
        eps = 1e-9
        self.assertGreaterEqual(result + eps, min(values))
        self.assertLessEqual(result, max(values) + eps)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestCleanupRetentionProperties(unittest.TestCase):
    @settings(max_examples=_MAX_EXAMPLES)
    @given(
        n_dirs=st.integers(min_value=0, max_value=12),
        keep_last=st.integers(min_value=0, max_value=12),
    )
    def test_cleanup_retains_exact_tail_set(self, n_dirs: int, keep_last: int) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = []
            for idx in range(n_dirs):
                path = os.path.join(tmpdir, f"window_{idx}")
                os.makedirs(path, exist_ok=True)
                dirs.append(path)

            retained = _cleanup_completed_window_dirs(dirs, keep_last_windows=keep_last)

            expected = dirs[-min(len(dirs), keep_last) :] if keep_last > 0 else []
            self.assertEqual(retained, expected)

            for path in expected:
                self.assertTrue(os.path.isdir(path))
            for path in dirs[: len(dirs) - len(expected)]:
                self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
