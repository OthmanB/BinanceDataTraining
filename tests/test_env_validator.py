import os
import unittest
from unittest.mock import patch

from utils.config_loader import ConfigError
from utils.env_validator import _get_required_env_vars, validate_environment


def _build_base_config() -> dict:
    return {
        "security": {
            "environment_variables": ["API_KEY", "SECRET_KEY"],
            "validation": {
                "check_env_vars_at_startup": True,
                "fail_if_missing": True,
            },
        }
    }


class TestEnvValidator(unittest.TestCase):
    def test_get_required_env_vars_rejects_non_list(self) -> None:
        config = _build_base_config()
        config["security"]["environment_variables"] = "API_KEY"

        with self.assertRaises(ConfigError):
            _get_required_env_vars(config)

    def test_validate_environment_passes_when_required_vars_exist(self) -> None:
        config = _build_base_config()
        with patch.dict(os.environ, {"API_KEY": "x", "SECRET_KEY": "y"}, clear=True):
            validate_environment(config)

    def test_validate_environment_skips_when_check_disabled(self) -> None:
        config = _build_base_config()
        config["security"]["validation"]["check_env_vars_at_startup"] = False

        with patch.dict(os.environ, {}, clear=True):
            validate_environment(config)

    def test_validate_environment_raises_when_missing_and_required(self) -> None:
        config = _build_base_config()
        with patch.dict(os.environ, {"API_KEY": "x"}, clear=True):
            with self.assertRaises(ConfigError) as exc_info:
                validate_environment(config)

        self.assertIn("SECRET_KEY", str(exc_info.exception))

    def test_validate_environment_allows_missing_when_not_fatal(self) -> None:
        config = _build_base_config()
        config["security"]["validation"]["fail_if_missing"] = False

        with patch.dict(os.environ, {}, clear=True):
            validate_environment(config)


if __name__ == "__main__":
    unittest.main()
