"""Artifact management helpers (skeleton)."""

from typing import Any
import logging


logger = logging.getLogger(__name__)


def log_artifact(obj: Any, description: str) -> None:
    """Log an artifact to MLFlow.

    Phase 3: placeholder that logs invocation only.
    """

    logger.info("log_artifact called (Phase 3 skeleton). description=%s", description)


__all__ = ["log_artifact"]
