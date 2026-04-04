"""Custom Keras layers.

This module hosts custom layers when defined.
"""

from typing import Dict


def get_custom_layers() -> Dict[str, object]:
    """Return a mapping of custom layer names to their classes."""

    return {}


__all__ = ["get_custom_layers"]
