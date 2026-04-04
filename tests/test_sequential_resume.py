"""Tests for sequential training resume helpers and flow."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from training.pipeline import (
    _cleanup_completed_window_dirs,
    _resolve_sequential_resume_paths,
    _run_snapshot_training_pipeline_sequential,
    _run_snapshot_training_pipeline_sequential_result,
    _save_sequential_resume_state,
)
from utils.config_loader import ConfigError


class TestSequentialResume(unittest.TestCase):
    def _base_config(self, snapshot_dir: str) -> dict:
        return {
            "training": {
                "sequential_training": {
                    "enabled": True,
                    "window_days": 1,
                    "cleanup_completed_windows": False,
                    "resume_enabled": True,
                }
            },
            "snapshot": {
                "enabled": True,
                "directory": snapshot_dir,
                "root_name": "e2e_trials_v2",
                "name": "resume_test",
            },
            "data": {
                "asset_pairs": {"target_asset": "BTCUSDT"},
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                },
            },
            "mlflow": {
                "artifact_logging": {
                    "trained_model": False,
                }
            },
        }

    def test_resume_paths_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            windows = [
                ("2024-01-01", "2024-01-01"),
                ("2024-01-02", "2024-01-02"),
            ]

            state_a, model_a = _resolve_sequential_resume_paths(config, windows)
            state_b, model_b = _resolve_sequential_resume_paths(config, windows)

            self.assertEqual(state_a, state_b)
            self.assertEqual(model_a, model_b)
            self.assertTrue(state_a.endswith(".json"))
            self.assertTrue(model_a.endswith(".keras"))

    def test_resume_paths_change_when_namespace_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            windows = [
                ("2024-01-01", "2024-01-01"),
                ("2024-01-02", "2024-01-02"),
            ]

            state_default, model_default = _resolve_sequential_resume_paths(config, windows)

            config["training"]["sequential_training"]["resume_namespace"] = "hpo__trial=17__resource=gpu:1"
            state_namespaced, model_namespaced = _resolve_sequential_resume_paths(config, windows)

            self.assertNotEqual(state_default, state_namespaced)
            self.assertNotEqual(model_default, model_namespaced)

    def test_cleanup_retains_only_last_n_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = os.path.join(tmpdir, "w1")
            d2 = os.path.join(tmpdir, "w2")
            d3 = os.path.join(tmpdir, "w3")
            os.makedirs(d1, exist_ok=True)
            os.makedirs(d2, exist_ok=True)
            os.makedirs(d3, exist_ok=True)

            retained = _cleanup_completed_window_dirs([d1, d2, d3], keep_last_windows=1)

            self.assertEqual(retained, [d3])
            self.assertFalse(os.path.exists(d1))
            self.assertFalse(os.path.exists(d2))
            self.assertTrue(os.path.exists(d3))

    def test_cleanup_raises_for_negative_retention(self) -> None:
        with self.assertRaises(ConfigError):
            _cleanup_completed_window_dirs([], keep_last_windows=-1)

    @mock.patch("training.pipeline._save_sequential_resume_model")
    @mock.patch("training.pipeline._fit_snapshot_model_once")
    @mock.patch("training.pipeline.prepare_snapshot_dataset")
    @mock.patch("training.pipeline._load_sequential_resume_model")
    def test_run_sequential_resumes_from_saved_window(
        self,
        mock_load_model: mock.MagicMock,
        mock_prepare_snapshot: mock.MagicMock,
        mock_fit_once: mock.MagicMock,
        mock_save_model: mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._base_config(tmpdir)
            windows = [
                ("2024-01-01", "2024-01-01"),
                ("2024-01-02", "2024-01-02"),
                ("2024-01-03", "2024-01-03"),
            ]
            state_path, model_path = _resolve_sequential_resume_paths(config, windows)

            _save_sequential_resume_state(
                state_path,
                {
                    "version": 1,
                    "windows": windows,
                    "next_window_index": 2,
                    "epoch_offset": 4,
                    "hpo_window_metrics": [[0.20, 100.0], [0.30, 120.0]],
                },
            )
            with open(model_path, "w", encoding="utf-8") as handle:
                handle.write("checkpoint")

            mock_load_model.return_value = "RESUMED_MODEL"
            mock_prepare_snapshot.return_value = object()
            mock_fit_once.return_value = ("UPDATED_MODEL", 1, 0.40, 80.0)

            result = _run_snapshot_training_pipeline_sequential_result(config, windows)
            model = result.model

            self.assertEqual(model, "UPDATED_MODEL")
            self.assertEqual(mock_fit_once.call_count, 1)
            fit_args = mock_fit_once.call_args.args
            self.assertEqual(fit_args[2], "RESUMED_MODEL")
            self.assertAlmostEqual(float(result.hpo_metric_value or 0.0), 0.29333333333333333)
            self.assertFalse(os.path.exists(state_path))
            self.assertFalse(os.path.exists(model_path))
            self.assertGreaterEqual(mock_save_model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
