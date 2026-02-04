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

    def test_alignment_keys_in_schema(self) -> None:
        """Schema should require alignment missing_policy and max_gap_seconds."""
        import yaml
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "config" / "validation_schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        data_section = schema["sections"]["data"]
        required_keys = data_section.get("required_keys", {})
        assert "asset_pairs.alignment.missing_policy" in required_keys
        assert "asset_pairs.alignment.max_gap_seconds" in required_keys

    def test_mlflow_artifact_logging_keys_in_schema(self) -> None:
        """Schema should require MLflow artifact logging flags."""
        import yaml
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "config" / "validation_schema.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        mlflow_section = schema["sections"]["mlflow"]
        required_keys = mlflow_section.get("required_keys", {})
        assert "artifact_logging.confusion_matrix" in required_keys
        assert "artifact_logging.trained_model" in required_keys
        assert "model_registry.register_model" in required_keys


class TestFailFastAlignmentConfig:
    """Fail-fast tests for alignment config usage."""

    def test_missing_alignment_key_raises_key_error(self) -> None:
        import numpy as np
        from training.snapshot_dataset import SnapshotRecord, _align_multi_asset_records

        t0 = np.datetime64("2024-01-01T00:00:00")
        record = SnapshotRecord(
            timestamp=t0,
            snapshot_features=[1.0, 1.0, 1.0, 1.0],
            depth=None,
            mid_price=1.0,
            hybrid_snapshot=None,
            volume_proxy=0.0,
            confidence=1.0,
            gap_reset=False,
        )

        asset_records = {"BTCUSDT": [record], "ETHUSDT": [record]}
        alignment_cfg = {
            "method": "interpolate",
            "max_gap_seconds": 60,
            "bucket_tolerance_seconds": 1.0,
        }

        with pytest.raises(KeyError):
            _align_multi_asset_records(
                asset_records=asset_records,
                assets=["BTCUSDT", "ETHUSDT"],
                target_asset="BTCUSDT",
                alignment_cfg=alignment_cfg,
                representation="top_of_book",
                cadence_seconds=10,
                hybrid_levels=None,
                fail_on_invalid=True,
            )


class TestFailFastLongTermConfig:
    """Fail-fast tests for long-term config usage."""

    def test_missing_long_term_key_raises(self) -> None:
        from preprocessing.long_term_features import LongTermConfig, LongTermFeatureError

        config = {
            "model": {
                "long_term": {
                    "enabled": True,
                    "windows_days": [7],
                    "resolution_days": 1,
                }
            }
        }

        with pytest.raises(LongTermFeatureError):
            LongTermConfig.from_config(config)
