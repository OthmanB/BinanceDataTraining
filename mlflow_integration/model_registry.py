"""Model registry helpers (MLFlow integration)."""

from __future__ import annotations

from typing import Optional
import logging


logger = logging.getLogger(__name__)


def _artifact_path_exists(client: object, run_id: str, artifact_path: str) -> bool:
    try:
        list_artifacts = getattr(client, "list_artifacts")
    except Exception:
        return False

    # Try listing the parent directory first.
    parent = ""
    child = artifact_path
    try:
        infos = list_artifacts(run_id, path=parent)
    except Exception:  # noqa: BLE001
        infos = None

    if infos:
        for info in infos:
            try:
                if str(getattr(info, "path", "")) == child:
                    return True
            except Exception:  # noqa: BLE001
                continue

    # Fall back to trying to list the child path.
    try:
        infos_child = list_artifacts(run_id, path=child)
    except Exception:  # noqa: BLE001
        infos_child = None
    return bool(infos_child)


def register_model(name: str, *, run_id: Optional[str] = None, artifact_path: str = "model") -> None:
    """Register a model in the MLFlow model registry.

    This helper assumes that a model has already been logged to the current
    MLFlow run under ``artifact_path`` using ``mlflow.tensorflow.log_model``.
    It then registers that logged model under the provided registry name.
    """

    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLFlow is required for model registry operations but could not be imported: %s",
            exc,
        )
        return

    active_run = mlflow.active_run()
    if active_run is None and not run_id:
        logger.warning(
            "No active MLFlow run detected; skipping model registry registration for name=%s",
            name,
        )
        return

    resolved_run_id = str(run_id) if run_id else str(active_run.info.run_id)

    try:
        from mlflow.tracking import MlflowClient  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MlflowClient for artifact checks: %s", exc)
        return

    client = MlflowClient()
    if not _artifact_path_exists(client, resolved_run_id, str(artifact_path)):
        logger.warning(
            "Skipping model registry registration: artifact path not found. run_id=%s artifact_path=%s name=%s",
            resolved_run_id,
            artifact_path,
            name,
        )
        return

    model_uri = f"runs:/{resolved_run_id}/{artifact_path}"

    logger.info(
        "Registering model in MLFlow registry. name=%s, model_uri=%s",
        name,
        model_uri,
    )

    try:
        result = mlflow.register_model(model_uri=model_uri, name=name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to register model in MLFlow registry. name=%s, uri=%s, error=%s",
            name,
            model_uri,
            exc,
        )
        return

    logger.info(
        "Model registration request submitted. name=%s, model_uri=%s, version=%s",
        name,
        model_uri,
        getattr(result, "version", None),
    )


__all__ = ["register_model"]
