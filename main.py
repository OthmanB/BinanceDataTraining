"""Entry point for Binance ML Training Platform.

Current responsibilities (Phase 2):
- Load configuration from YAML
- Resolve environment variables and validate schema
- Configure colored logging
- Validate required environment variables
- Start an MLFlow run using the configured tracking URI and experiment
- Execute a skeleton data pipeline:
  - Build a (placeholder) DataObject from configuration
  - Run temporal feature attachment (no-op)
  - Run preprocessing validation
  - Compute chronological train/validation/test splits

No model training is performed yet.
"""

import sys
from typing import Any, Dict
from datetime import datetime

from utils.config_loader import ConfigError, load_config
from utils.env_validator import validate_environment
from utils.colored_logging import setup_colored_logging
from data.data_loader import load_order_book_data
from preprocessing.temporal_features import attach_temporal_features
from preprocessing.transformer import run_preprocessing_pipeline
from preprocessing.train_test_split import chronological_split_indices
from diagnostics import run_data_diagnostics
from training import run_training_pipeline
from evaluation import evaluate_model, evaluate_snapshot_model
from mlflow_integration import start_run, end_run
from models.hyperparameter_tuning import run_hyperparameter_search


def _enforce_production_sample_cap(config: Dict[str, Any], n_samples: int) -> None:
    run_mode_cfg = config["run_mode"]  # Required by schema
    mode = str(run_mode_cfg["mode"])  # Required by schema
    if mode != "production":
        return

    training_cfg = config["training"]  # Required by schema
    debug_max_samples = int(training_cfg["debug_max_samples"])  # Required by schema
    if debug_max_samples < n_samples:
        raise ConfigError(
            "training.debug_max_samples must be >= metadata.num_samples when run_mode.mode='production'. "
            f"debug_max_samples={debug_max_samples}, num_samples={n_samples}."
        )


def main() -> int:
    # Initial minimal logging to stderr in case config loading fails
    try:
        config = load_config()
    except ConfigError as exc:
        # Use a very simple stderr output here; colored logging is not yet available
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1

    # Configure logging according to loaded config
    logger = setup_colored_logging(config)

    try:
        validate_environment(config)
    except ConfigError as exc:
        logger.error(f"Environment validation failed: {exc}")
        return 1

    # Determine run mode (production vs trial) from configuration.
    run_mode_cfg = config["run_mode"]  # Required by schema
    mode = str(run_mode_cfg["mode"])  # Required by schema
    if mode not in ("production", "trial"):
        logger.error("Invalid run_mode.mode in configuration: %r (expected 'production' or 'trial')", mode)
        return 1

    # Start MLFlow run
    mlflow_cfg = config["mlflow"]
    run_naming_cfg = mlflow_cfg["run_naming"]
    run_pattern = run_naming_cfg["pattern"]
    if not run_pattern:
        logger.error("mlflow.run_naming.pattern is required in configuration")
        return 1
    target_asset = config["data"]["asset_pairs"]["target_asset"]
    model_name = config["model"]["architecture"]
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    run_name = run_pattern.format(
        asset=target_asset,
        model=model_name,
        timestamp=timestamp,
    )

    try:
        start_run(config, run_name=run_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start MLFlow run: %s", exc)
        return 1

    snapshot_cfg = config["snapshot"]  # Required by schema
    snapshot_enabled = bool(snapshot_cfg["enabled"])  # Required by schema

    if not snapshot_enabled:
        logger.error(
            "Legacy in-memory pipeline is disabled. Set snapshot.enabled=true to use the snapshot pipeline.",
        )
        end_run()
        return 1

    data_object = None
    logger.info(
        "Snapshot mode enabled; skipping Phase 2 in-memory data pipeline and diagnostics.",
    )

    config_for_training = config

    hpo_cfg = config["hyperparameter_optimization"]  # Required by schema
    if bool(hpo_cfg["enabled"]):
        logger.error("Hyperparameter optimization is not supported when snapshot.enabled is true.")
        end_run()
        return 1

    if mode == "trial":
        logger.info(
            "run_mode.mode='trial'; skipping final production training and evaluation after hyperparameter search.",
        )
        end_run()
        return 0

    # Phase 3: minimal training pipeline (runs inside the same MLFlow run)
    try:
        model = run_training_pipeline(config_for_training, data_object)
    except Exception as exc:  # noqa: BLE001
        logger.error("Training pipeline (Phase 3 minimal) failed: %s", exc)
        end_run()
        return 1

    if model is None:
        logger.info("Training pipeline returned no model; skipping evaluation.")
        end_run()
        return 0

    logger.info("Snapshot mode enabled; invoking snapshot evaluation pipeline.")

    try:
        evaluate_snapshot_model(config_for_training, model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Snapshot evaluation pipeline failed: %s", exc)
        end_run()
        return 1

    # Close MLFlow run
    end_run()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
