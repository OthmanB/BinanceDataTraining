"""Calibration analysis skeleton.

Phase 3 provides function signatures only; implementation will follow the
calibration strategy defined in the technical specifications.
"""

from typing import Any
import logging


logger = logging.getLogger(__name__)


def compute_calibration_metrics(y_true: Any, y_prob: Any) -> None:
    """Compute calibration metrics.

    Phase 3: placeholder that logs invocation only.
    """

    logger.info("compute_calibration_metrics called (Phase 3 skeleton).")


__all__ = ["compute_calibration_metrics"]
