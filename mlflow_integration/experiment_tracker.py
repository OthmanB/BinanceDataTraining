"""Experiment tracking helpers using MLFlow.

This module provides thin wrappers around MLFlow to:
- Set the tracking URI and experiment based on YAML configuration
- Start and end runs with consistent logging

It assumes that environment variables required for authentication are
validated at startup by the env_validator utilities.
"""

from typing import Any, Dict, Optional
import logging
import os
from pathlib import Path
import yaml


logger = logging.getLogger(__name__)


def _import_mlflow():
    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MLFlow is required for experiment tracking but could not be imported."
        ) from exc
    return mlflow


def start_run(config: Dict[str, Any], run_name: Optional[str] = None):
    """Start an MLFlow run using configuration settings.

    This function:
    - Sets the MLFlow tracking URI from config.mlflow.tracking_uri
    - Sets/creates the experiment named in config.mlflow.experiment_name
    - Starts a run with the optional run_name

    Returns
    -------
    mlflow.entities.Run
        The active MLFlow run object.
    """

    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name")
    local_tmp_dir = mlflow_cfg.get("local_tmp_dir")

    if not tracking_uri:
        raise ValueError("mlflow.tracking_uri must be set in configuration")
    if not experiment_name:
        raise ValueError("mlflow.experiment_name must be set in configuration")
    if not local_tmp_dir:
        raise ValueError("mlflow.local_tmp_dir must be set in configuration")

    mlflow = _import_mlflow()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Configure the local temporary directory for MLflow client-side operations.
    # This directory is only used as a staging area on the training machine;
    # the authoritative artifact store remains the server-side default_artifact_root.
    tmp_path = Path(local_tmp_dir).expanduser().resolve()
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        os.chdir(tmp_path)
        logger.info("Set MLFlow local temporary directory to %s", tmp_path)
    except OSError as exc:  # noqa: BLE001
        logger.warning("Failed to change working directory to MLFlow local_tmp_dir %s: %s", tmp_path, exc)

    # Enable TensorFlow/Keras autologging so that training metrics, parameters,
    # and model artifacts are automatically captured in MLFlow.
    # Autologging is intentionally disabled; metrics and model artifacts are
    # logged explicitly from the training pipeline for greater control.
    logger.info("TensorFlow autologging is disabled; using explicit MLFlow logging.")

    logger.info(
        "Starting MLFlow run. tracking_uri=%s, experiment_name=%s, run_name=%s",
        tracking_uri,
        experiment_name,
        run_name,
    )

    run = mlflow.start_run(run_name=run_name)

    try:
        snapshot_path = tmp_path / "training_config_effective.yaml"
        with snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        try:
            mlflow.log_artifact(str(snapshot_path), artifact_path="config")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log configuration snapshot artifact to MLFlow: %s", exc)
        else:
            logger.info(
                "Configuration snapshot written to %s and logged to MLFlow under artifact path 'config'.",
                snapshot_path,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write configuration snapshot for MLFlow logging: %s", exc)

    # Log a few high-level configuration parameters for convenience.
    data_cfg = config.get("data", {})
    asset_pairs_cfg = data_cfg.get("asset_pairs", {})
    model_cfg = config.get("model", {})
    compilation_cfg = model_cfg.get("compilation", {})
    training_cfg = config.get("training", {})

    params = {
        "target_asset": asset_pairs_cfg.get("target_asset"),
        "model_architecture": model_cfg.get("architecture"),
        "training_epochs": training_cfg.get("epochs"),
        "training_batch_size": training_cfg.get("batch_size"),
        "training_debug_max_samples": training_cfg.get("debug_max_samples"),
        "optimizer": compilation_cfg.get("optimizer"),
        "learning_rate": compilation_cfg.get("learning_rate"),
        "loss_function": compilation_cfg.get("loss"),
    }

    for name, value in params.items():
        if value is not None:
            try:
                mlflow.log_param(name, value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log MLFlow parameter %s: %s", name, exc)

    return run


def end_run() -> None:
    """End the active MLFlow run."""

    mlflow = _import_mlflow()
    logger.info("Ending MLFlow run.")
    mlflow.end_run()


__all__ = ["start_run", "end_run"]
