"""Model registry helpers (MLFlow integration)."""

from typing import Any
import logging


logger = logging.getLogger(__name__)


def register_model(model: Any, name: str) -> None:
    """Register a model in the MLFlow model registry.

    This helper assumes that the Keras model has already been logged to the
    current MLFlow run under the default artifact path ``"model"`` using
    ``mlflow.tensorflow.log_model``. It then registers that logged model under
    the provided registry name.
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
    if active_run is None:
        logger.warning(
            "No active MLFlow run detected; skipping model registry registration for name=%s",
            name,
        )
        return

    run_id = active_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

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
