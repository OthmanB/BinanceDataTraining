"""Model registry helpers (skeleton)."""

from typing import Any
import logging


logger = logging.getLogger(__name__)


def register_model(model: Any, name: str) -> None:
    """Register a model in the MLFlow model registry.

    Phase 3: placeholder that logs invocation only.
    """

    logger.info("register_model called (Phase 3 skeleton). name=%s", name)


__all__ = ["register_model"]
