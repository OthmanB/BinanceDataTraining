"""Training metrics utilities (skeleton)."""

from typing import Any, Dict, List
import logging


logger = logging.getLogger(__name__)


def get_metric_names_from_config(config: Dict[str, Any]) -> List[str]:
    """Extract metric names from the model configuration.

    Phase 3: helper only; no framework-specific metric objects are created.
    """

    model_cfg = config.get("model", {})
    compilation_cfg = model_cfg.get("compilation", {})
    metrics = compilation_cfg.get("metrics", []) or []

    logger.info("Configured metrics: %s", metrics)

    return [str(m) for m in metrics]


__all__ = ["get_metric_names_from_config"]
