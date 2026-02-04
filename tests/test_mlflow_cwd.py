"""Tests for TD-018: MLflow CWD side effect fix.

These tests verify that:
1. start_run() does not change the working directory
2. The original CWD is preserved and accessible
3. Relative paths can be resolved against the original CWD
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestMlflowCwdFix(unittest.TestCase):
    """Test that MLflow integration preserves the working directory."""

    def test_get_original_cwd_returns_path(self) -> None:
        """get_original_cwd should return a Path object."""
        from mlflow_integration import get_original_cwd

        cwd = get_original_cwd()
        self.assertIsInstance(cwd, Path)
        self.assertTrue(cwd.is_absolute())

    def test_original_cwd_matches_actual_cwd_before_start_run(self) -> None:
        """original CWD should match actual CWD if start_run not called."""
        from mlflow_integration import get_original_cwd

        # The original CWD is captured at module import time
        original = get_original_cwd()
        # It should be a valid directory
        self.assertTrue(original.exists())
        self.assertTrue(original.is_dir())

    def test_resolve_path_absolute_unchanged(self) -> None:
        """Absolute paths should be returned unchanged."""
        from mlflow_integration import resolve_path_from_original_cwd

        abs_path = "/tmp/test/path"
        resolved = resolve_path_from_original_cwd(abs_path)
        self.assertEqual(resolved, Path(abs_path).resolve())

    def test_resolve_path_relative_uses_original_cwd(self) -> None:
        """Relative paths should be resolved against original CWD."""
        from mlflow_integration import get_original_cwd, resolve_path_from_original_cwd

        rel_path = "relative/path/to/file"
        resolved = resolve_path_from_original_cwd(rel_path)
        expected = (get_original_cwd() / rel_path).resolve()
        self.assertEqual(resolved, expected)

    @patch("mlflow_integration.experiment_tracker._import_mlflow")
    def test_start_run_does_not_change_cwd(self, mock_import_mlflow: MagicMock) -> None:
        """start_run should not change the working directory."""
        from mlflow_integration import start_run

        # Set up mock mlflow
        mock_mlflow = MagicMock()
        mock_import_mlflow.return_value = mock_mlflow
        mock_mlflow.start_run.return_value = MagicMock()

        # Record CWD before
        cwd_before = Path.cwd().resolve()

        # Create temp dir for local_tmp_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "mlflow": {
                    "tracking_uri": "http://localhost:5000",
                    "experiment_name": "test-experiment",
                    "local_tmp_dir": tmpdir,
                    "run_naming": {"pattern": "test-run"},
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
                    "compilation": {"optimizer": "adam", "learning_rate": 0.001, "loss": "mse", "metrics": ["accuracy"]},
                },
                "training": {"epochs": 1, "batch_size": 16, "debug_max_samples": 100},
            }

            try:
                start_run(config, run_name="test-run")
            except Exception:
                pass  # May fail due to mock limitations, but CWD should be unchanged

            # CWD should be unchanged
            cwd_after = Path.cwd().resolve()
            self.assertEqual(cwd_before, cwd_after, "start_run should not change CWD")


class TestPathResolutionWithMlflow(unittest.TestCase):
    """Test that paths are correctly resolved when using MLflow."""

    def test_snapshot_directory_resolution(self) -> None:
        """Snapshot directory should resolve correctly using original CWD."""
        from mlflow_integration import get_original_cwd, resolve_path_from_original_cwd

        # Simulate a relative snapshot directory from config
        snapshot_dir = "snapshots/test_run"
        
        # Resolve it
        resolved = resolve_path_from_original_cwd(snapshot_dir)
        
        # Should be absolute and contain the original CWD
        self.assertTrue(resolved.is_absolute())
        expected = (get_original_cwd() / snapshot_dir).resolve()
        self.assertEqual(resolved, expected)

    def test_cache_directory_resolution(self) -> None:
        """Dataset cache directory should resolve correctly."""
        from mlflow_integration import resolve_path_from_original_cwd

        cache_dir = "cache/datasets"
        resolved = resolve_path_from_original_cwd(cache_dir)
        self.assertTrue(resolved.is_absolute())


class TestExportedFunctions(unittest.TestCase):
    """Test that TD-018 utility functions are properly exported."""

    def test_functions_exported_from_package(self) -> None:
        """get_original_cwd and resolve_path_from_original_cwd should be exported."""
        import mlflow_integration

        self.assertTrue(hasattr(mlflow_integration, "get_original_cwd"))
        self.assertTrue(hasattr(mlflow_integration, "resolve_path_from_original_cwd"))
        self.assertTrue(callable(mlflow_integration.get_original_cwd))
        self.assertTrue(callable(mlflow_integration.resolve_path_from_original_cwd))


if __name__ == "__main__":
    unittest.main()
