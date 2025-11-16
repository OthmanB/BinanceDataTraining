"""Training callbacks skeleton.

Phase 3 defines the interface for creating Keras callbacks based on the
configuration but does not instantiate any real callbacks yet.
"""

from typing import Any, Dict, List
import logging


logger = logging.getLogger(__name__)


def create_callbacks(config: Dict[str, Any]) -> List[Any]:
    """Create a list of training callbacks.

    Phase 3: returns an empty list and logs the configuration options that
    would be used to create callbacks in later phases.
    """

    callbacks_cfg = config.get("training", {}).get("callbacks", {})

    logger.info(
        "create_callbacks called (Phase 3 skeleton). early_stopping.enabled=%s, reduce_lr.enabled=%s",
        callbacks_cfg.get("early_stopping", {}).get("enabled"),
        callbacks_cfg.get("reduce_lr", {}).get("enabled"),
    )

    try:
        from tensorflow.keras.callbacks import (  # type: ignore[import]
            EarlyStopping,
            ReduceLROnPlateau,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import Keras callbacks: %s", exc)
        return []

    created_callbacks: List[Any] = []

    # EarlyStopping callback
    es_cfg = callbacks_cfg.get("early_stopping", {})
    if bool(es_cfg.get("enabled")):
        try:
            es_monitor = str(es_cfg["monitor"])
            es_patience = int(es_cfg["patience"])
            es_restore_best = bool(es_cfg["restore_best_weights"])
        except KeyError as exc:  # noqa: BLE001
            logger.warning("Missing early_stopping config key: %s", exc)
        else:
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
    rl_cfg = callbacks_cfg.get("reduce_lr", {})
    if bool(rl_cfg.get("enabled")):
        try:
            rl_monitor = str(rl_cfg["monitor"])
            rl_factor = float(rl_cfg["factor"])
            rl_patience = int(rl_cfg["patience"])
            rl_min_lr = float(rl_cfg["min_lr"])
        except KeyError as exc:  # noqa: BLE001
            logger.warning("Missing reduce_lr config key: %s", exc)
        else:
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

    return created_callbacks


__all__ = ["create_callbacks"]
