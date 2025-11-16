"""External data source integration stubs.

This module provides placeholders for future integrations with external data
sources (e.g., macro indices, sentiment feeds). Phase 2 does not implement any
actual external fetching logic.
"""

from typing import Any, Dict
import logging


logger = logging.getLogger(__name__)


def load_external_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load external data according to the configuration.

    Phase 2: returns an empty dictionary and logs that no external sources are
    configured yet.
    """

    logger.info(
        "External data loading requested but not configured; returning empty payload."
    )
    return {}


__all__ = ["load_external_data"]
