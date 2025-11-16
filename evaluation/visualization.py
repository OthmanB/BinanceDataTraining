"""Visualization utilities skeleton for evaluation and diagnostics."""

from typing import Any
import logging


logger = logging.getLogger(__name__)


def log_confusion_matrix(confusion_matrix: Any) -> None:
    """Log or plot a confusion matrix.

    Phase 3: placeholder that logs invocation only.
    """

    logger.info("log_confusion_matrix called (Phase 3 skeleton).")


__all__ = ["log_confusion_matrix"]
