"""Tests for TD-012: Fail-fast configuration validation.

These tests verify that the code fails fast when required configuration
values are missing, rather than falling back to implicit defaults.
"""

import pytest
from utils.colored_logging import setup_colored_logging


class TestColoredLoggingFailFast:
    """Test that colored_logging fails fast when config is missing required keys."""

    def test_missing_logging_section_raises_key_error(self) -> None:
        """Should raise KeyError when logging section is missing."""
        config: dict = {}
        with pytest.raises(KeyError):
            setup_colored_logging(config)

    def test_missing_level_raises_key_error(self) -> None:
        """Should raise KeyError when logging.level is missing."""
        config = {"logging": {"colors": {}}}
        with pytest.raises(KeyError):
            setup_colored_logging(config)

    def test_missing_colors_raises_key_error(self) -> None:
        """Should raise KeyError when logging.colors is missing."""
        config = {"logging": {"level": "INFO"}}
        with pytest.raises(KeyError):
            setup_colored_logging(config)

    def test_invalid_level_raises_value_error(self) -> None:
        """Should raise ValueError for invalid logging level."""
        config = {"logging": {"level": "INVALID_LEVEL", "colors": {}}}
        with pytest.raises(ValueError, match="Invalid logging level"):
            setup_colored_logging(config)

    def test_valid_config_creates_logger(self) -> None:
        """Should succeed with valid configuration."""
        config = {
            "logging": {
                "level": "INFO",
                "colors": {
                    "info": "green",
                    "warning": "yellow",
                    "error": "red",
                    "debug": "blue",
                    "function_names": "cyan",
                },
            }
        }
        logger = setup_colored_logging(config)
        assert logger is not None


class TestEnforceProductionSampleCapFailFast:
    """Test that _enforce_production_sample_cap fails fast with missing config."""

    def test_missing_run_mode_raises_key_error(self) -> None:
        """Should raise KeyError when run_mode is missing."""
        from main import _enforce_production_sample_cap

        config: dict = {"training": {"debug_max_samples": 100}}
        with pytest.raises(KeyError):
            _enforce_production_sample_cap(config, 50)

    def test_missing_training_raises_key_error_in_production(self) -> None:
        """Should raise KeyError when training section is missing in production mode."""
        from main import _enforce_production_sample_cap

        config: dict = {"run_mode": {"mode": "production"}}
        with pytest.raises(KeyError):
            _enforce_production_sample_cap(config, 50)

    def test_missing_debug_max_samples_raises_key_error(self) -> None:
        """Should raise KeyError when debug_max_samples is missing in production."""
        from main import _enforce_production_sample_cap

        config = {"run_mode": {"mode": "production"}, "training": {}}
        with pytest.raises(KeyError):
            _enforce_production_sample_cap(config, 50)

    def test_trial_mode_skips_validation(self) -> None:
        """Trial mode should skip validation and not raise."""
        from main import _enforce_production_sample_cap

        # Trial mode doesn't require training section since it returns early
        config = {"run_mode": {"mode": "trial"}, "training": {"debug_max_samples": 10}}
        # Should not raise
        _enforce_production_sample_cap(config, 50)

    def test_production_mode_valid_config(self) -> None:
        """Production mode with valid config should not raise."""
        from main import _enforce_production_sample_cap

        config = {"run_mode": {"mode": "production"}, "training": {"debug_max_samples": 100}}
        # Should not raise when debug_max_samples >= n_samples
        _enforce_production_sample_cap(config, 50)


class TestSchemaRequiredKeys:
    """Test that validation schema includes all required keys."""

    def test_run_naming_pattern_in_schema(self) -> None:
        """Schema should require mlflow.run_naming.pattern."""
        import yaml
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "config" / "validation_schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        mlflow_section = schema["sections"]["mlflow"]
        required_keys = mlflow_section.get("required_keys", {})
        assert "run_naming.pattern" in required_keys, "mlflow.run_naming.pattern should be required"

    def test_logging_level_in_schema(self) -> None:
        """Schema should require logging.level."""
        import yaml
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "config" / "validation_schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        logging_section = schema["sections"]["logging"]
        required_keys = logging_section.get("required_keys", {})
        assert "level" in required_keys, "logging.level should be required"

    def test_logging_colors_in_schema(self) -> None:
        """Schema should require logging.colors entries."""
        import yaml
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "config" / "validation_schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        logging_section = schema["sections"]["logging"]
        required_keys = logging_section.get("required_keys", {})
        # Check for at least one color entry
        assert "colors.info" in required_keys, "logging.colors.info should be required"
