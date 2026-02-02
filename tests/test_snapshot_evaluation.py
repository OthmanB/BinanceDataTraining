import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

from evaluation.evaluator import evaluate_snapshot_model
from training.snapshot_store import load_or_create_manifest, resolve_snapshot_context, save_manifest
from utils.config_loader import load_config


class _DummyModel:
    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes

    def predict(self, x, batch_size=None, verbose=0):  # type: ignore[override]
        n_samples = x.shape[0]
        probs = np.full((n_samples, self._num_classes), 1.0 / self._num_classes, dtype="float64")
        return [probs.copy(), probs.copy()]


class TestSnapshotEvaluation(unittest.TestCase):
    def test_evaluate_snapshot_model_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "DATABASE_URI": "http://example-db",
                    "DATABASE_URI_HIST": "http://example-db-hist",
                    "DATABASE_URI_LIVE": "http://example-db-live",
                    "MLFLOW_TRACKING_URI": "http://mlflow",
                },
                clear=False,
            ):
                config = load_config(
                    config_path="config/training_config.yaml",
                    schema_path="config/validation_schema.yaml",
                )

            config["snapshot"]["enabled"] = True
            config["snapshot"]["directory"] = tmp_dir
            config["snapshot"]["root_name"] = "snapshot_eval_test"
            config["snapshot"]["name"] = "auto"
            config["snapshot"]["on_config_mismatch"] = "create_new"
            config["snapshot"]["max_snapshots"] = 0

            config["data"]["asset_pairs"]["correlated_assets"] = []
            config["training"]["batch_size"] = 4
            config["training"]["debug_max_samples"] = 4

            config["evaluation"]["calibration_analysis"]["enabled"] = False
            config["mlflow"]["artifact_logging"]["confusion_matrix"] = False

            context = resolve_snapshot_context(config)
            os.makedirs(os.path.join(context.snapshot_dir, "chunks"), exist_ok=True)

            manifest = load_or_create_manifest(context, config)

            x_core = np.random.randn(10, 2, 2, 2, 1).astype("float32")
            mask = np.ones_like(x_core, dtype="float32")
            x = np.concatenate([x_core, mask], axis=-1)
            y_up = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype="int64")
            y_down = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="int64")
            anchor_ts = np.arange(10, dtype="int64")

            chunk_path = os.path.join(context.snapshot_dir, "chunks", "chunk.npz")
            np.savez_compressed(chunk_path, x=x, y_up=y_up, y_down=y_down, anchor_ts=anchor_ts)

            manifest["chunks"] = [
                {
                    "start": "2024-01-01 00:00:00",
                    "end": "2024-01-01 01:00:00",
                    "file": "chunks/chunk.npz",
                    "num_samples": int(x.shape[0]),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
            manifest["complete"] = True
            save_manifest(context, manifest)

            num_classes = int(config["model"]["output"]["num_classes"])
            model = _DummyModel(num_classes)

            evaluate_snapshot_model(config, model)

            stats_path = os.path.join(context.snapshot_dir, "normalization_stats_train.npz")
            self.assertTrue(os.path.exists(stats_path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
