"""Characterization tests for config-based HPO metric state flow."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from training.pipeline import (
    _resolve_sequential_resume_paths,
    _run_snapshot_training_pipeline_sequential_result,
    _save_sequential_resume_state,
)


class TestConfigStateContract(unittest.TestCase):
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

    @mock.patch("training.pipeline._save_sequential_resume_model")
    @mock.patch("training.pipeline._fit_snapshot_model_once")
    @mock.patch("training.pipeline.prepare_snapshot_dataset")
    @mock.patch("training.pipeline._load_sequential_resume_model")
    def test_sequential_resume_reports_aggregated_hpo_metric(
        self,
        mock_load_model: mock.MagicMock,
        mock_prepare_snapshot: mock.MagicMock,
        mock_fit_once: mock.MagicMock,
        _mock_save_model: mock.MagicMock,
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

            self.assertAlmostEqual(float(result.hpo_metric_value or 0.0), 0.29333333333333333)
            self.assertFalse(os.path.exists(state_path))
            self.assertFalse(os.path.exists(model_path))


if __name__ == "__main__":
    unittest.main()
