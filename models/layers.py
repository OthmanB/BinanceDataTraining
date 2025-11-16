"""Custom Keras layers (placeholder).

This module will host custom layers such as MaxRescaling in later phases.
For now, it only documents the expected structure.
"""

from typing import Dict


def get_custom_layers() -> Dict[str, object]:
    """Return a mapping of custom layer names to their classes.

    Phase 3 returns an empty mapping; custom layers will be added in future
    phases as needed by the model architectures.
    """

    return {}


__all__ = ["get_custom_layers"]
