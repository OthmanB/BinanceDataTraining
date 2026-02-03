"""Long-term context integration helpers for training pipeline (TD-019).

This module provides utilities for integrating long-term context features
into the training pipeline. It bridges the preprocessing.long_term_features
module with the snapshot-based training generators.

Usage:
    1. Compute long-term features for the entire dataset once:
       lt_features = compute_long_term_features_for_dataset(config, snapshot_dataset)

    2. Use the augmented generator for dual-input training:
       gen = build_dual_input_generator(snapshot_dataset, lt_features, ...)

    3. Or wrap an existing generator:
       dual_gen = wrap_generator_with_long_term(single_gen, lt_features, indices)

Note: Full integration into run_snapshot_based_training requires modifying
the generator to yield [x_short, x_long] instead of just x_short. This
module provides the building blocks for that integration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


def compute_long_term_features_for_dataset(
    config: Dict[str, Any],
    snapshot_dataset: Any,
    cadence_seconds: int = 10,
) -> Optional[np.ndarray]:
    """Compute long-term features for all samples in a snapshot dataset.

    Parameters
    ----------
    config:
        Full configuration dictionary with model.long_term settings.
    snapshot_dataset:
        SnapshotDataset instance containing the training data.
    cadence_seconds:
        Time interval between snapshots in seconds.

    Returns
    -------
    Optional[np.ndarray]:
        2D array of shape (n_samples, lt_input_dim) containing long-term
        features for each sample. Returns None if long_term is disabled.

    Notes
    -----
    This function extracts mid_prices and timestamps from the snapshot dataset
    and computes long-term features using the preprocessing module.
    """
    from preprocessing.long_term_features import (
        LongTermConfig,
        compute_long_term_features,
    )

    lt_config = LongTermConfig.from_config(config)
    if not lt_config.enabled:
        logger.info("Long-term features disabled in config")
        return None

    # Extract data from snapshot dataset
    # The snapshot dataset should have methods/properties to access raw data
    try:
        mid_prices = snapshot_dataset.mid_prices
        timestamps = snapshot_dataset.timestamps
        anchor_timestamps = snapshot_dataset.anchor_timestamps
    except AttributeError:
        # Fallback: try to extract from chunks
        logger.warning(
            "SnapshotDataset does not expose mid_prices/timestamps directly. "
            "Long-term feature computation requires dataset modification."
        )
        return None

    # Extract volumes if available
    volumes = getattr(snapshot_dataset, "volumes", None)

    logger.info(
        "Computing long-term features for dataset: n_samples=%d, lt_input_dim=%d",
        len(anchor_timestamps),
        lt_config.input_dim,
    )

    lt_features = compute_long_term_features(
        config=config,
        mid_prices=mid_prices,
        timestamps=timestamps,
        anchor_timestamps=anchor_timestamps,
        cadence_seconds=cadence_seconds,
        volumes=volumes,
    )

    return lt_features


def wrap_generator_with_long_term(
    base_generator: Iterator[Tuple[Any, ...]],
    long_term_features: np.ndarray,
    start_index: int,
    batch_size: int,
) -> Iterator[Tuple[Any, ...]]:
    """Wrap a training generator to include long-term features.

    Parameters
    ----------
    base_generator:
        Original generator yielding (x, y) or (x, y, sample_weight).
    long_term_features:
        Precomputed long-term features array of shape (n_samples, lt_dim).
    start_index:
        Starting sample index for this generator.
    batch_size:
        Batch size for indexing into long_term_features.

    Yields
    ------
    Tuple:
        ([x_short, x_long], y) or ([x_short, x_long], y, sample_weight)
        depending on whether the base generator yields sample weights.

    Notes
    -----
    This wrapper assumes that the base generator yields batches in order
    starting from start_index. The long-term features are sliced accordingly.
    """
    current_idx = start_index

    for batch_data in base_generator:
        batch_len = batch_data[0].shape[0]
        end_idx = current_idx + batch_len

        lt_batch = long_term_features[current_idx:end_idx]
        current_idx = end_idx

        # Replace x with [x, lt_batch]
        x_short = batch_data[0]
        x_dual = [x_short, lt_batch]

        if len(batch_data) == 2:
            # (x, y) -> ([x, lt], y)
            yield (x_dual, batch_data[1])
        elif len(batch_data) == 3:
            # (x, y, sw) -> ([x, lt], y, sw)
            yield (x_dual, batch_data[1], batch_data[2])
        else:
            # Unknown format, pass through with modified x
            yield (x_dual,) + batch_data[1:]


def get_long_term_input_dim(config: Dict[str, Any]) -> int:
    """Get the long-term input dimension from configuration.

    Parameters
    ----------
    config:
        Full configuration dictionary.

    Returns
    -------
    int:
        Long-term input dimension, or 0 if disabled.
    """
    from preprocessing.long_term_features import LongTermConfig

    lt_config = LongTermConfig.from_config(config)
    return lt_config.input_dim


def is_long_term_enabled(config: Dict[str, Any]) -> bool:
    """Check if long-term context is enabled in configuration.

    Parameters
    ----------
    config:
        Full configuration dictionary.

    Returns
    -------
    bool:
        True if model.long_term.enabled is True.
    """
    model_cfg = config.get("model", {})
    lt_cfg = model_cfg.get("long_term", {})
    return bool(lt_cfg.get("enabled", False))


__all__ = [
    "compute_long_term_features_for_dataset",
    "wrap_generator_with_long_term",
    "get_long_term_input_dim",
    "is_long_term_enabled",
]
