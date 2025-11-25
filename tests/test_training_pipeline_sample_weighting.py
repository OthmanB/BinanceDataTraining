import os
import unittest
from unittest import mock

import numpy as np

from training.pipeline import run_training_pipeline
from utils.config_loader import load_config


class _DummyHistory:
    def __init__(self) -> None:
        self.history = {"loss": [1.0]}


class _DummyModel:
    def __init__(self) -> None:
        self.fit_kwargs = None

    def fit(self, **kwargs):  # type: ignore[override]
        self.fit_kwargs = kwargs
        return _DummyHistory()


class TestTrainingPipelineSampleWeighting(unittest.TestCase):
    @mock.patch("models.cnn_lstm_multiclass.build_cnn_lstm_model")
    def test_exponential_decay_sample_weighting_applied_to_both_heads(self, mock_builder) -> None:
        dummy_model = _DummyModel()
        mock_builder.return_value = dummy_model

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

        training_cfg = config["training"]
        training_cfg["epochs"] = 1
        training_cfg["batch_size"] = 2
        training_cfg["debug_max_samples"] = 3
        training_cfg["missing_snapshot_strategy"] = "synthetic"
        training_cfg["sample_weighting"]["enabled"] = True
        training_cfg["sample_weighting"]["half_life_days"] = 1
        training_cfg["sample_weighting"]["apply_to"] = "loss_function"

        config["mlflow"]["artifact_logging"]["trained_model"] = False
        config["mlflow"]["model_registry"]["register_model"] = False

        config["model"]["input_representation"]["temporal_features"]["integration_mode"] = "none"

        target_asset = config["data"]["asset_pairs"]["target_asset"]

        metadata = {
            "num_samples": 3,
            "anchor_indices": [0, 1, 2],
        }

        snapshot_timestamps = [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ]

        order_books = {
            target_asset: {
                "snapshot_timestamps": snapshot_timestamps,
            },
        }

        labels = [0, 1, 1]
        targets = {
            "labels_up_intensity": labels,
            "labels_down_intensity": labels,
        }

        data_object = {
            "metadata": metadata,
            "order_books": order_books,
            "temporal_features": {},
            "targets": targets,
            "external_data": {},
        }

        run_training_pipeline(config, data_object)

        fit_kwargs = dummy_model.fit_kwargs
        self.assertIsNotNone(fit_kwargs)
        self.assertIn("sample_weight", fit_kwargs)

        sample_weight = fit_kwargs["sample_weight"]
        self.assertIsInstance(sample_weight, list)
        self.assertEqual(len(sample_weight), 2)

        w_up, w_down = sample_weight
        np.testing.assert_allclose(w_up, w_down)

        expected_weights = np.array([0.25, 0.5], dtype="float32")
        np.testing.assert_allclose(w_up, expected_weights, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
