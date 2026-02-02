import os
import unittest
from unittest import mock

from training.pipeline import run_training_pipeline
from utils.config_loader import ConfigError, load_config


class TestTrainingPipelineSampleWeighting(unittest.TestCase):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
