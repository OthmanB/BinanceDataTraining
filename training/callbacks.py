"""Training callbacks utilities.

Creates Keras callbacks based on configuration when TensorFlow is available.
"""

from typing import Any, Dict, List
import logging


logger = logging.getLogger(__name__)


def create_callbacks(config: Dict[str, Any]) -> List[Any]:
    """Create a list of training callbacks."""

    callbacks_cfg = config["training"]["callbacks"]
    es_cfg = callbacks_cfg["early_stopping"]
    rl_cfg = callbacks_cfg["reduce_lr"]
    nan_cfg = callbacks_cfg.get("terminate_on_nan") or {}

    logger.info(
        "Creating callbacks. early_stopping.enabled=%s, reduce_lr.enabled=%s",
        es_cfg["enabled"],
        rl_cfg["enabled"],
    )

    try:
        from tensorflow.keras.callbacks import (  # type: ignore[import]
            EarlyStopping,
            ReduceLROnPlateau,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import Keras callbacks: %s", exc)
        return []

    TerminateOnNaN = None
    try:
        from tensorflow.keras.callbacks import TerminateOnNaN as _TerminateOnNaN  # type: ignore[import]

        TerminateOnNaN = _TerminateOnNaN
    except Exception:
        TerminateOnNaN = None

    created_callbacks: List[Any] = []

    # EarlyStopping callback
    if bool(es_cfg["enabled"]):
        es_monitor = str(es_cfg["monitor"])
        es_patience = int(es_cfg["patience"])
        es_restore_best = bool(es_cfg["restore_best_weights"])
        created_callbacks.append(
            EarlyStopping(
                monitor=es_monitor,
                patience=es_patience,
                restore_best_weights=es_restore_best,
            ),
        )
        logger.info(
            "Created EarlyStopping callback. monitor=%s, patience=%s, restore_best_weights=%s",
            es_monitor,
            es_patience,
            es_restore_best,
        )

    # ReduceLROnPlateau callback
    if bool(rl_cfg["enabled"]):
        rl_monitor = str(rl_cfg["monitor"])
        rl_factor = float(rl_cfg["factor"])
        rl_patience = int(rl_cfg["patience"])
        rl_min_lr = float(rl_cfg["min_lr"])
        created_callbacks.append(
            ReduceLROnPlateau(
                monitor=rl_monitor,
                factor=rl_factor,
                patience=rl_patience,
                min_lr=rl_min_lr,
            ),
        )
        logger.info(
            "Created ReduceLROnPlateau callback. monitor=%s, factor=%s, patience=%s, min_lr=%s",
            rl_monitor,
            rl_factor,
            rl_patience,
            rl_min_lr,
        )

    # TerminateOnNaN callback (fail-fast on numerical instability)
    if bool(nan_cfg.get("enabled", False)):
        if TerminateOnNaN is None:
            logger.warning("TerminateOnNaN requested but callback not available in this TensorFlow build.")
        else:
            created_callbacks.append(TerminateOnNaN())
            logger.info("Created TerminateOnNaN callback.")

    return created_callbacks


__all__ = ["create_callbacks"]
