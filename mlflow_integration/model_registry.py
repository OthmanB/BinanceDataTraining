"""Model registry helpers (MLFlow integration)."""

from __future__ import annotations

from typing import Optional
import logging


logger = logging.getLogger(__name__)


def _artifact_path_exists(client: object, run_id: str, artifact_path: str) -> bool:
    try:
        list_artifacts = getattr(client, "list_artifacts")
    except AttributeError:
        logger.debug("MlflowClient has no list_artifacts method; cannot verify artifact path.")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to access MlflowClient.list_artifacts: %s", exc)
        return False

    # Try listing the parent directory first.
    parent = ""
    child = artifact_path
    try:
        infos = list_artifacts(run_id, path=parent)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Failed listing MLFlow artifacts at parent path while validating artifact path. run_id=%s path=%s error=%s",
            run_id,
            parent,
            exc,
        )
        infos = None

    if infos:
        for info in infos:
            if str(getattr(info, "path", "")) == child:
                return True

    # Fall back to trying to list the child path.
    try:
        infos_child = list_artifacts(run_id, path=child)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Failed listing MLFlow artifacts at child path while validating artifact path. run_id=%s path=%s error=%s",
            run_id,
            child,
            exc,
        )
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
    except ImportError as exc:
        logger.debug(
            "MLFlow import unavailable; skipping model registry registration. name=%s error=%s",
            name,
            exc,
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unexpected failure importing MLFlow while registering model. name=%s error=%s",
            name,
            exc,
        )
        return

    try:
        active_run = mlflow.active_run()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to query active MLFlow run; skipping model registry registration. name=%s error=%s",
            name,
            exc,
        )
        return

    if active_run is None and not run_id:
        logger.warning(
            "No active MLFlow run detected; skipping model registry registration for name=%s",
            name,
        )
        return

    if run_id:
        resolved_run_id = str(run_id)
    else:
        if active_run is None:
            logger.warning("No run_id resolved for model registry registration. name=%s", name)
            return
        assert active_run is not None
        resolved_run_id = str(getattr(getattr(active_run, "info", None), "run_id", ""))
        if not resolved_run_id:
            logger.warning("Active MLFlow run did not expose a run_id; skipping registration. name=%s", name)
            return

    try:
        from mlflow.tracking import MlflowClient  # type: ignore[import]
    except ImportError as exc:
        logger.debug("MlflowClient import unavailable; skipping model registry registration: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected failure importing MlflowClient for artifact checks: %s", exc)
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
