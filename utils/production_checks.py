"""Shared runtime checks for production-mode configuration constraints."""

from __future__ import annotations

from typing import Any, Dict

from utils.config_loader import ConfigError


def enforce_production_sample_cap(config: Dict[str, Any], n_samples: int) -> None:
    run_mode_cfg = config["run_mode"]
    mode = str(run_mode_cfg["mode"])
    if mode != "production":
        return

    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])
    if debug_max_samples < n_samples:
        raise ConfigError(
            "training.debug_max_samples must be >= metadata.num_samples when run_mode.mode='production'. "
            f"debug_max_samples={debug_max_samples}, num_samples={n_samples}."
        )


__all__ = ["enforce_production_sample_cap"]
