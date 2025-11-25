import os
import unittest
from unittest import mock

from utils.config_loader import ConfigError, _resolve_env_placeholders, load_config


class TestConfigLoader(unittest.TestCase):
    def test_load_config_resolves_env_placeholders(self) -> None:
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

        self.assertIn("data", config)
        self.assertIn("training", config)
        self.assertIn("mlflow", config)

        db_uri = config["data"]["connection"]["database_uri"]
        self.assertIsInstance(db_uri, str)
        self.assertNotIn("${", db_uri)

    def test_resolve_env_placeholders_missing_variable_raises(self) -> None:
        with self.assertRaises(ConfigError):
            _resolve_env_placeholders("${MISSING_ENV_VAR}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
