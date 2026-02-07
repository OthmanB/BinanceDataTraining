import os
import tempfile
import unittest
from pathlib import Path
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

    def test_base_config_override_merges(self) -> None:
        base_config_path = os.path.abspath("config/training_config.yaml")
        override_yaml = f"""
base_config: "{base_config_path}"
training:
  epochs: 2
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "override.yaml"
            override_path.write_text(override_yaml, encoding="utf-8")

            env = {
                "DATABASE_URI": "http://example-db",
                "DATABASE_URI_HIST": "http://example-db-hist",
                "DATABASE_URI_LIVE": "http://example-db-live",
                "MLFLOW_TRACKING_URI": "http://mlflow",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                config = load_config(
                    config_path=str(override_path),
                    schema_path="config/validation_schema.yaml",
                )

            self.assertEqual(int(config["training"]["epochs"]), 2)

    def test_optional_schema_key_type_is_validated_when_present(self) -> None:
        base_config_path = os.path.abspath("config/training_config.yaml")
        override_yaml = f"""
base_config: "{base_config_path}"
training:
  sequential_training:
    enabled: true
    window_days: "seven"
    cleanup_completed_windows: false
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = Path(tmpdir) / "override_bad_optional.yaml"
            override_path.write_text(override_yaml, encoding="utf-8")

            env = {
                "DATABASE_URI": "http://example-db",
                "DATABASE_URI_HIST": "http://example-db-hist",
                "DATABASE_URI_LIVE": "http://example-db-live",
                "MLFLOW_TRACKING_URI": "http://mlflow",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(ConfigError):
                    load_config(
                        config_path=str(override_path),
                        schema_path="config/validation_schema.yaml",
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
