"""Safe wrappers around the MLflow fluent API.

The MLflow fluent API will implicitly start a run when logging if no active run
exists. In this repository we want MLflow runs to be managed explicitly by
``mlflow_integration.experiment_tracker``.

These helpers:
- Only log when an active MLflow run exists.
- Never raise on MLflow import/logging failures (best-effort).
"""

from __future__ import annotations

from typing import Any, Optional
import logging


logger = logging.getLogger(__name__)


def get_mlflow_if_active() -> Optional[Any]:
    """Return the imported mlflow module if a run is active; else None."""

    try:
        import mlflow  # type: ignore[import]
    except Exception:  # noqa: BLE001
        return None

    try:
        active = mlflow.active_run()
    except Exception:  # noqa: BLE001
        return None

    if active is None:
        return None
    return mlflow


def log_keras_model_to_active_run(
    model: Any,
    *,
    artifact_path: str = "model",
    signature: Any = None,
) -> bool:
    """Log a Keras model to the active run.

    Returns True when the model artifact logging succeeds.
    """

    mlflow = get_mlflow_if_active()
    if mlflow is None:
        return False

    try:
        mlflow_tf = __import__("mlflow.tensorflow", fromlist=["log_model"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow TensorFlow integration for model logging: %s", exc)
        return False

    try:
        if signature is not None:
            mlflow_tf.log_model(model, str(artifact_path), signature=signature)
        else:
            mlflow_tf.log_model(model, str(artifact_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log model to MLFlow. artifact_path=%s error=%s", artifact_path, exc)
        return False

    return True


def log_keras_model_and_register_if_enabled(
    model: Any,
    *,
    model_name: str,
    register_enabled: bool,
    artifact_path: str = "model",
    signature: Any = None,
) -> bool:
    """Log a model and optionally register it.

    Registry registration is only attempted if model artifact logging succeeds.
    """

    model_logged = log_keras_model_to_active_run(
        model,
        artifact_path=artifact_path,
        signature=signature,
    )

    if not register_enabled:
        return model_logged

    if not model_logged:
        logger.warning("MLFlow model registry registration skipped because model logging failed.")
        return False

    try:
        from mlflow_integration.model_registry import register_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow model registry helper: %s", exc)
        return model_logged

    try:
        register_model(str(model_name), artifact_path=str(artifact_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to register model '%s' in MLFlow model registry: %s", model_name, exc)

    return model_logged


__all__ = [
    "get_mlflow_if_active",
    "log_keras_model_to_active_run",
    "log_keras_model_and_register_if_enabled",
]
