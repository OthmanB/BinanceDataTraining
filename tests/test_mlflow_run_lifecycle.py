"""Tests for MLflow run lifecycle safety.

These tests ensure that the experiment_tracker helpers:
- Do not start a new run when one is already active.
- Do not end runs that were not started by the tracker.
- Can end a leaked nested run when it is a direct child of the started run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _minimal_config(local_tmp_dir: str) -> dict:
    return {
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test-experiment",
            "local_tmp_dir": local_tmp_dir,
            "run_naming": {"pattern": "test"},
            "artifact_logging": {
                "trained_model": False,
                "model_architecture_plot": False,
                "training_plots": False,
                "confusion_matrix": False,
                "class_distribution": False,
                "feature_importance": False,
            },
            "model_registry": {"register_model": False, "model_name_pattern": "test"},
        },
        "data": {"asset_pairs": {"target_asset": "BTCUSDT", "correlated_assets": []}},
        "model": {
            "architecture": "cnn_lstm",
            "compilation": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "mse",
                "metrics": ["accuracy"],
            },
        },
        "training": {"epochs": 1, "batch_size": 16, "debug_max_samples": 100},
    }


class TestMlflowRunLifecycle(unittest.TestCase):
    def test_start_run_reuses_active_run_and_does_not_end_it(self) -> None:
        import mlflow_integration.experiment_tracker as tracker

        tracker.clear_started_run_id()
        mock_mlflow = MagicMock()
        existing = SimpleNamespace(info=SimpleNamespace(run_id="existing"), data=SimpleNamespace(tags={}))
        mock_mlflow.active_run.return_value = existing

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_config(tmpdir)
            with patch("mlflow_integration.experiment_tracker._import_mlflow", return_value=mock_mlflow):
                run = tracker.start_run(config, run_name="ignored")

        self.assertIs(run, existing)
        self.assertIsNone(tracker.get_started_run_id())
        mock_mlflow.start_run.assert_not_called()

        with patch("mlflow_integration.experiment_tracker._import_mlflow", return_value=mock_mlflow):
            tracker.end_run(expected_run_id="existing")
        mock_mlflow.end_run.assert_not_called()

    def test_end_run_closes_direct_child_then_parent(self) -> None:
        import mlflow_integration.experiment_tracker as tracker

        tracker.clear_started_run_id()
        mock_mlflow = MagicMock()
        parent = SimpleNamespace(info=SimpleNamespace(run_id="parent"), data=SimpleNamespace(tags={}))
        child = SimpleNamespace(
            info=SimpleNamespace(run_id="child"),
            data=SimpleNamespace(tags={"mlflow.parentRunId": "parent"}),
        )

        mock_mlflow.active_run.return_value = None
        mock_mlflow.start_run.return_value = parent

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_config(tmpdir)
            with patch("mlflow_integration.experiment_tracker._import_mlflow", return_value=mock_mlflow):
                run = tracker.start_run(config, run_name="parent")
        self.assertEqual(str(run.info.run_id), "parent")
        self.assertEqual(tracker.get_started_run_id(), "parent")

        # end_run() should end the leaked child first, then the parent.
        mock_mlflow.active_run.side_effect = [child, parent]
        with patch("mlflow_integration.experiment_tracker._import_mlflow", return_value=mock_mlflow):
            tracker.end_run(expected_run_id="parent")

        self.assertIsNone(tracker.get_started_run_id())
        self.assertEqual(mock_mlflow.end_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
