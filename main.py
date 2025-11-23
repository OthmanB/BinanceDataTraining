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
from evaluation import evaluate_model
from mlflow_integration import start_run, end_run


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

    # Start MLFlow run
    mlflow_cfg = config.get("mlflow", {})
    run_pattern = (
        mlflow_cfg.get("run_naming", {}).get("pattern")
        or "{asset}_{model}_{timestamp}"
    )
    target_asset = config.get("data", {}).get("asset_pairs", {}).get("target_asset")
    model_name = config.get("model", {}).get("architecture")
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

    # Phase 2: execute data pipeline skeleton inside the MLFlow run
    try:
        data_object = load_order_book_data(config)
        data_object = run_preprocessing_pipeline(config, data_object)
        data_object = attach_temporal_features(config, data_object)
    except Exception as exc:  # noqa: BLE001
        logger.error("Data pipeline (Phase 2 skeleton) failed: %s", exc)
        end_run()
        return 1

    metadata = data_object["metadata"]
    n_samples = int(metadata["num_samples"])

    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    train_idx, val_idx, test_idx = chronological_split_indices(
        n_samples,
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    logger.info(
        "Configuration loaded and environment validated successfully. "
        f"MLFlow tracking_uri={mlflow_cfg.get('tracking_uri')}, "
        f"experiment_name={mlflow_cfg.get('experiment_name')}"
    )
    logger.info(
        "Phase 2 data pipeline summary: n_samples=%d, train=%d, val=%d, test=%d",
        n_samples,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    logger.info("Phase 2 setup complete. Invoking data diagnostics stage before training.")

    try:
        run_data_diagnostics(config, data_object, train_idx, val_idx, test_idx)
    except Exception as exc:  # noqa: BLE001
        logger.error("Data diagnostics stage failed: %s", exc)

    logger.info("Data diagnostics stage complete. Invoking training pipeline (Phase 3 minimal).")

    # Phase 3: minimal training pipeline (runs inside the same MLFlow run)
    try:
        model = run_training_pipeline(config, data_object)
    except Exception as exc:  # noqa: BLE001
        logger.error("Training pipeline (Phase 3 minimal) failed: %s", exc)
        end_run()
        return 1

    if model is None:
        logger.info("Training pipeline returned no model; skipping evaluation.")
        end_run()
        return 0

    logger.info("Phase 3 training complete. Invoking evaluation pipeline (Phase 4 minimal).")

    # Phase 4: evaluation pipeline (runs inside the same MLFlow run)
    try:
        evaluate_model(config, model, data_object)
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation pipeline (Phase 4 minimal) failed: %s", exc)
        end_run()
        return 1

    # Close MLFlow run
    end_run()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
