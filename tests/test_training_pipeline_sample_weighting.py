import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from training.pipeline import run_training_pipeline
from training.snapshot_dataset import SnapshotChunk, SnapshotDataset, build_training_generator
from utils.config_loader import ConfigError, load_config


class TestTrainingPipelineSampleWeighting(unittest.TestCase):
    def _build_tiny_snapshot_dataset(self, tmpdir: str) -> SnapshotDataset:
        chunk_path = os.path.join(tmpdir, "chunk_000.npz")
        x = np.zeros((4, 2, 1, 1, 1), dtype="float32")
        y_up = np.array([0, 1, 0, 1], dtype="int64")
        y_down = np.array([1, 0, 1, 0], dtype="int64")
        anchor_ts = np.array([86_400, 172_800, 259_200, 345_600], dtype="int64")
        duty_cycle = np.array([1.0, 0.5, 0.25, 1.0], dtype="float32")
        np.savez(chunk_path, x=x, y_up=y_up, y_down=y_down, anchor_ts=anchor_ts, duty_cycle=duty_cycle)

        chunk = SnapshotChunk(
            start="2024-01-01",
            end="2024-01-04",
            file_path=chunk_path,
            num_samples=4,
            start_index=0,
        )
        return SnapshotDataset(
            snapshot_dir=tmpdir,
            manifest={},
            chunks=[chunk],
            total_samples=4,
            config_hash="unit-test",
        )

    def test_legacy_pipeline_disabled(self) -> None:
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

        config["snapshot"]["enabled"] = False

        with self.assertRaises(ConfigError):
            run_training_pipeline(config, None)

    def test_build_training_generator_rejects_invalid_apply_to(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._build_tiny_snapshot_dataset(tmpdir)
            with self.assertRaises(ValueError):
                build_training_generator(
                    dataset=dataset,
                    start_index=0,
                    end_index=4,
                    batch_size=2,
                    num_classes=2,
                    normalization=None,
                    sample_weight_cfg={
                        "enabled": True,
                        "method": "exponential_decay",
                        "half_life_days": 30,
                        "apply_to": "metrics",
                    },
                )

    def test_build_training_generator_applies_duty_cycle_and_class_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = self._build_tiny_snapshot_dataset(tmpdir)
            generator, _ = build_training_generator(
                dataset=dataset,
                start_index=0,
                end_index=4,
                batch_size=2,
                num_classes=2,
                normalization=None,
                sample_weight_cfg=None,
                class_weights_up={0: 2.0, 1: 3.0},
                class_weights_down={0: 5.0, 1: 7.0},
            )
            _x, _y, sample_weight = next(generator)
            weights_up, weights_down = sample_weight

            np.testing.assert_allclose(weights_up, np.array([2.0, 1.5], dtype="float32"))
            np.testing.assert_allclose(weights_down, np.array([7.0, 2.5], dtype="float32"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
