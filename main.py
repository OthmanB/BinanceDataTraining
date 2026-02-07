"""Entry point for Binance ML Training Platform.

Current responsibilities:
- Load configuration from YAML
- Resolve environment variables and validate schema
- Configure colored logging (+ optional file logging)
- Validate required environment variables
- Start an MLFlow run using the configured tracking URI and experiment
- Execute the snapshot training pipeline and snapshot evaluation
"""

import argparse
import copy
import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import time
import traceback

from utils.config_loader import ConfigError, load_config
from utils.env_validator import validate_environment
from utils.colored_logging import setup_colored_logging
from data.data_loader import load_order_book_data
from preprocessing.temporal_features import attach_temporal_features
from preprocessing.transformer import run_preprocessing_pipeline
from preprocessing.train_test_split import chronological_split_indices
from diagnostics import run_data_diagnostics
from training import run_training_pipeline
from training.pipeline import _resolve_sequential_windows
from training.snapshot_store import resolve_snapshot_context
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


def _format_visible_devices(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _resolve_runtime_tuning_options(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    gpu_memory_growth = runtime_cfg.get("gpu_memory_growth")
    if gpu_memory_growth is None:
        enable_memory_growth = False
    elif isinstance(gpu_memory_growth, bool):
        enable_memory_growth = gpu_memory_growth
    else:
        raise ConfigError("training.runtime.gpu_memory_growth must be a boolean when provided")

    gpu_allocator_raw = runtime_cfg.get("gpu_allocator")
    gpu_allocator: Optional[str]
    if gpu_allocator_raw is None:
        gpu_allocator = None
    else:
        gpu_allocator = str(gpu_allocator_raw).strip().lower()
        if gpu_allocator not in {"default", "cuda_malloc_async"}:
            raise ConfigError("training.runtime.gpu_allocator must be 'default' or 'cuda_malloc_async'")

    gpu_init_stagger_seconds = runtime_cfg.get("gpu_init_stagger_seconds")
    if gpu_init_stagger_seconds is None:
        stagger_seconds = 0.0
    else:
        stagger_seconds = float(gpu_init_stagger_seconds)
        if stagger_seconds < 0:
            raise ConfigError("training.runtime.gpu_init_stagger_seconds must be >= 0")

    return {
        "gpu_memory_growth": enable_memory_growth,
        "gpu_allocator": gpu_allocator,
        "gpu_init_stagger_seconds": stagger_seconds,
    }


def _apply_runtime_device(config: Dict[str, Any], logger: logging.Logger) -> None:
    training_cfg = config.get("training")
    if not isinstance(training_cfg, dict):
        raise ConfigError("training section must be a mapping")
    runtime_cfg = training_cfg.get("runtime")
    if not isinstance(runtime_cfg, dict):
        raise ConfigError("training.runtime must be a mapping")

    device = str(runtime_cfg.get("device", "")).strip().lower()
    if device not in {"cpu", "gpu"}:
        raise ConfigError("training.runtime.device must be 'cpu' or 'gpu'")

    visible_devices = _format_visible_devices(runtime_cfg.get("gpu_visible_devices"))
    tuning = _resolve_runtime_tuning_options(runtime_cfg)

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if tuning["gpu_allocator"] == "default":
            os.environ.pop("TF_GPU_ALLOCATOR", None)
        logger.info("Runtime device set to CPU (CUDA_VISIBLE_DEVICES='').")
        return

    if tuning["gpu_allocator"] == "cuda_malloc_async":
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    elif tuning["gpu_allocator"] == "default":
        os.environ.pop("TF_GPU_ALLOCATOR", None)

    if visible_devices is not None and visible_devices.strip() and visible_devices.lower() != "null":
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        logger.info("Runtime device set to GPU (CUDA_VISIBLE_DEVICES=%s).", visible_devices)
    else:
        logger.info("Runtime device set to GPU (default CUDA_VISIBLE_DEVICES).")

    stagger_seconds = float(tuning["gpu_init_stagger_seconds"])
    if stagger_seconds > 0:
        logger.info("GPU initialization stagger enabled: sleeping %.2fs", stagger_seconds)
        time.sleep(stagger_seconds)


def _validate_runtime_device_availability(config: Dict[str, Any], logger: logging.Logger) -> None:
    training_cfg = config.get("training")
    if not isinstance(training_cfg, dict):
        raise ConfigError("training section must be a mapping")
    runtime_cfg = training_cfg.get("runtime")
    if not isinstance(runtime_cfg, dict):
        raise ConfigError("training.runtime must be a mapping")

    device = str(runtime_cfg.get("device", "")).strip().lower()
    tuning = _resolve_runtime_tuning_options(runtime_cfg)
    if device == "cpu":
        return
    if device != "gpu":
        raise ConfigError("training.runtime.device must be 'cpu' or 'gpu'")

    try:
        import tensorflow as tf  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            "training.runtime.device='gpu' requires TensorFlow to be importable in the current environment"
        ) from exc

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ConfigError(
            "training.runtime.device='gpu' but TensorFlow cannot see any GPU devices. "
            "Install TensorFlow CUDA dependencies (e.g. tensorflow[and-cuda]) and verify NVIDIA driver/runtime setup."
        )

    if bool(tuning["gpu_memory_growth"]):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled TensorFlow GPU memory growth on %s visible device(s).", len(gpus))
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to enable TensorFlow GPU memory growth: {exc}") from exc

    gpu_names = ", ".join(str(getattr(dev, "name", "GPU")) for dev in gpus)
    logger.info("Runtime GPU check passed. Visible TensorFlow devices: %s", gpu_names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance ML Training Platform")
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--schema",
        default="config/validation_schema.yaml",
        help="Path to configuration schema YAML",
    )
    return parser.parse_args()


def _evaluate_snapshot_sequential(config: Dict[str, Any], model: Any, logger: logging.Logger) -> None:
    windows = _resolve_sequential_windows(config)
    if windows is None or len(windows) <= 1:
        evaluate_snapshot_model(config, model)
        return

    sequential_cfg = config["training"].get("sequential_training") or {}
    cleanup_completed = bool(sequential_cfg.get("cleanup_completed_windows", False))
    cleanup_keep_last_windows = int(sequential_cfg.get("cleanup_keep_last_windows", 0))
    if cleanup_keep_last_windows < 0:
        raise ConfigError("training.sequential_training.cleanup_keep_last_windows must be >= 0")

    retained_dirs = []

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        window_config = copy.deepcopy(config)
        window_config["data"]["time_range"]["start_date"] = window_start
        window_config["data"]["time_range"]["end_date"] = window_end

        logger.info(
            "Sequential snapshot evaluation window %s/%s: %s -> %s",
            idx,
            len(windows),
            window_start,
            window_end,
        )
        evaluate_snapshot_model(window_config, model)

        if cleanup_completed:
            context = resolve_snapshot_context(window_config)
            retained_dirs.append(context.snapshot_dir)
            while len(retained_dirs) > cleanup_keep_last_windows:
                old_dir = retained_dirs.pop(0)
                if not os.path.isdir(old_dir):
                    continue
                try:
                    shutil.rmtree(old_dir)
                    logger.info("Removed completed evaluation window snapshot directory: %s", old_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to remove evaluation window snapshot directory %s: %s",
                        old_dir,
                        exc,
                    )


def main() -> int:
    # Initial minimal logging to stderr in case config loading fails
    try:
        args = _parse_args()
        config = load_config(config_path=args.config, schema_path=args.schema)
    except ConfigError as exc:
        # Use a very simple stderr output here; colored logging is not yet available
        sys.stderr.write(f"Configuration error: {exc}\n")
        return 1

    # Configure logging according to loaded config
    logger = setup_colored_logging(config)

    run_log_path = os.environ.get("RUN_LOG_PATH")
    if run_log_path:
        try:
            log_dir = os.path.dirname(run_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(run_log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
            logger.info("File logging enabled. RUN_LOG_PATH=%s", run_log_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to attach file logger to RUN_LOG_PATH=%s: %s", run_log_path, exc)

    writer = None
    try:
        from observability.run_state import get_run_state_writer

        writer = get_run_state_writer()
    except Exception:
        writer = None
    if writer is not None:
        try:
            writer.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to start run-state writer: %s", exc)

    try:
        _apply_runtime_device(config, logger)
        _validate_runtime_device_availability(config, logger)
    except ConfigError as exc:
        logger.error("Runtime device configuration invalid: %s", exc)
        if writer is not None:
            writer.set_error(f"Runtime device configuration invalid: {exc}", traceback_text=None)
        return 1

    try:
        validate_environment(config)
    except ConfigError as exc:
        logger.error(f"Environment validation failed: {exc}")
        if writer is not None:
            writer.set_error(f"Environment validation failed: {exc}", traceback_text=None)
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

    mlflow_run = None
    try:
        mlflow_run = start_run(config, run_name=run_name)
        if writer is not None and mlflow_run is not None:
            try:
                writer.set_run_id(mlflow_run.info.run_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set run_id on run-state writer: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start MLFlow run: %s", exc)
        if writer is not None:
            writer.set_error(f"Failed to start MLFlow run: {exc}", traceback_text=traceback.format_exc())
        return 1

    snapshot_cfg = config["snapshot"]  # Required by schema
    snapshot_enabled = bool(snapshot_cfg["enabled"])  # Required by schema

    try:
        if not snapshot_enabled:
            message = "Legacy in-memory pipeline is disabled. Set snapshot.enabled=true to use the snapshot pipeline."
            logger.error(message)
            if writer is not None:
                writer.set_error(message, traceback_text=None)
            return 1

        data_object = None
        logger.info(
            "Snapshot mode enabled; skipping in-memory data pipeline and diagnostics.",
        )

        config_for_training = config

        hpo_cfg = config["hyperparameter_optimization"]  # Required by schema
        if bool(hpo_cfg["enabled"]):
            if mode == "trial":
                windows = _resolve_sequential_windows(config_for_training)
                if windows is None or len(windows) <= 1:
                    message = (
                        "Snapshot HPO in trial mode requires bounded sequential windows. "
                        "Set training.sequential_training.enabled=true and choose a positive window_days."
                    )
                    logger.error(message)
                    if writer is not None:
                        writer.set_error(message, traceback_text=None)
                    return 1

            logger.info("Starting snapshot-compatible hyperparameter optimization.")
            best_config = run_hyperparameter_search(config_for_training, data_object)
            if best_config is not None:
                config_for_training = best_config

        if mode == "trial":
            logger.info(
                "run_mode.mode='trial'; skipping final production training and evaluation after hyperparameter search.",
            )
            if writer is not None:
                writer.set_stage("trial")
                writer.complete()
            return 0

        if writer is not None:
            writer.set_stage("training")
        model = run_training_pipeline(config_for_training, data_object)

        if model is None:
            logger.info("Training pipeline returned no model; skipping evaluation.")
            if writer is not None:
                writer.complete()
            return 0

        logger.info("Snapshot mode enabled; invoking snapshot evaluation pipeline.")

        if writer is not None:
            writer.set_stage("evaluation")
        _evaluate_snapshot_sequential(config_for_training, model, logger)

        if writer is not None:
            writer.complete()
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Run failed: %s", exc)
        if writer is not None:
            writer.set_error(str(exc), traceback_text=traceback.format_exc())
        return 1
    finally:
        if mlflow_run is not None:
            try:
                end_run()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to end MLFlow run cleanly: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
