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

Note: Long-term features are cached per snapshot dataset and require
the snapshot series files (timestamps, mid_prices, volumes) generated
alongside chunk data.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterator, Optional, Tuple
import hashlib
import json
import logging
import os

import numpy as np

from utils.config_loader import ConfigError


logger = logging.getLogger(__name__)

LONG_TERM_FEATURES_FILENAME = "long_term_features.npz"
ANCHOR_TIMESTAMPS_FILENAME = "anchor_timestamps.npy"


def _load_series_from_snapshot(snapshot_dataset: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = getattr(snapshot_dataset, "manifest", {}) or {}
    series_meta = manifest.get("series", {})
    if not isinstance(series_meta, dict):
        series_meta = {}
    series_chunks = series_meta.get("chunks", []) or []
    if not series_chunks:
        raise ConfigError(
            "Snapshot series data is missing. Rebuild the snapshot dataset to enable long-term features."
        )

    series_chunks = sorted(series_chunks, key=lambda item: item.get("start") or "")
    timestamps_list = []
    mid_prices_list = []
    volumes_list = []

    for entry in series_chunks:
        file_rel = entry.get("file")
        if not file_rel:
            raise ConfigError("Snapshot series entry missing file path in manifest")
        file_path = os.path.join(snapshot_dataset.snapshot_dir, file_rel)
        if not os.path.exists(file_path):
            raise ConfigError(f"Snapshot series file missing: {file_path}")

        with np.load(file_path) as npz:
            timestamps = np.asarray(npz["timestamps"])
            mid_prices = np.asarray(npz["mid_prices"])
            volumes = np.asarray(npz["volumes"])

        if timestamps.ndim != 1 or mid_prices.ndim != 1 or volumes.ndim != 1:
            raise ConfigError("Snapshot series arrays must be 1D")
        if not (len(timestamps) == len(mid_prices) == len(volumes)):
            raise ConfigError("Snapshot series arrays have inconsistent lengths")

        timestamps_list.append(timestamps.astype("int64"))
        mid_prices_list.append(mid_prices.astype("float64"))
        volumes_list.append(volumes.astype("float64"))

    return (
        np.concatenate(timestamps_list) if timestamps_list else np.array([], dtype="int64"),
        np.concatenate(mid_prices_list) if mid_prices_list else np.array([], dtype="float64"),
        np.concatenate(volumes_list) if volumes_list else np.array([], dtype="float64"),
    )


def load_snapshot_series(snapshot_dataset: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load target-asset series (timestamps, mid_prices, volumes) from snapshot data."""
    return _load_series_from_snapshot(snapshot_dataset)


def _load_anchor_timestamps(snapshot_dataset: Any) -> np.ndarray:
    cache_path = os.path.join(snapshot_dataset.snapshot_dir, ANCHOR_TIMESTAMPS_FILENAME)
    if os.path.exists(cache_path):
        anchor_ts = np.load(cache_path)
        if anchor_ts.ndim != 1:
            raise ConfigError("Cached anchor timestamps must be a 1D array")
        return anchor_ts.astype("int64")

    anchor_list = []
    for chunk in snapshot_dataset.chunks:
        with np.load(chunk.file_path) as npz:
            anchor_ts = np.asarray(npz["anchor_ts"], dtype="int64")
        if anchor_ts.ndim != 1:
            raise ConfigError("Anchor timestamps array must be 1D")
        anchor_list.append(anchor_ts)

    if not anchor_list:
        return np.array([], dtype="int64")

    anchor_ts = np.concatenate(anchor_list)
    np.save(cache_path, anchor_ts)
    return anchor_ts


def load_anchor_timestamps(snapshot_dataset: Any) -> np.ndarray:
    """Load anchor timestamps for snapshot samples (cached if available)."""
    return _load_anchor_timestamps(snapshot_dataset)


def _build_long_term_cache_metadata(
    snapshot_dataset: Any,
    lt_config: Any,
    cadence_seconds: int,
    num_samples: int,
) -> Dict[str, Any]:
    payload = {
        "snapshot_config_hash": getattr(snapshot_dataset, "config_hash", ""),
        "cadence_seconds": cadence_seconds,
        "long_term_config": asdict(lt_config),
        "input_dim": int(lt_config.input_dim),
        "num_samples": num_samples,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    payload["cache_hash"] = payload_hash
    return payload


def _load_long_term_cache(cache_path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    if not os.path.exists(cache_path):
        return None
    with np.load(cache_path, allow_pickle=False) as npz:
        if "features" not in npz or "metadata" not in npz:
            return None
        features = np.asarray(npz["features"], dtype="float32")
        meta_raw = npz["metadata"].item()
    try:
        metadata = json.loads(str(meta_raw))
    except json.JSONDecodeError:
        return None
    return features, metadata


def _save_long_term_cache(cache_path: str, features: np.ndarray, metadata: Dict[str, Any]) -> None:
    metadata_json = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(
        cache_path,
        features=features.astype("float32"),
        metadata=np.array([metadata_json]),
    )


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
    This function loads target-asset series data from the snapshot manifest and
    caches computed long-term features under the snapshot directory.
    """
    from preprocessing.long_term_features import (
        LongTermConfig,
        compute_long_term_features,
    )

    lt_config = LongTermConfig.from_config(config)
    if not lt_config.enabled:
        logger.info("Long-term features disabled in config")
        return None

    if cadence_seconds <= 0:
        raise ConfigError("data.time_range.cadence_seconds must be positive for long-term features")

    cache_path = os.path.join(snapshot_dataset.snapshot_dir, LONG_TERM_FEATURES_FILENAME)
    anchor_timestamps = _load_anchor_timestamps(snapshot_dataset)
    num_samples = int(anchor_timestamps.shape[0])
    expected_samples = int(snapshot_dataset.total_samples)
    if num_samples != expected_samples:
        raise ConfigError(
            "Anchor timestamps length does not match snapshot dataset sample count: "
            f"anchors={num_samples}, total_samples={expected_samples}"
        )

    cache_meta = _build_long_term_cache_metadata(
        snapshot_dataset,
        lt_config,
        cadence_seconds,
        num_samples,
    )

    cached = _load_long_term_cache(cache_path)
    if cached is not None:
        cached_features, cached_meta = cached
        if cached_meta.get("cache_hash") == cache_meta.get("cache_hash"):
            if cached_features.shape[0] == num_samples and cached_features.shape[1] == lt_config.input_dim:
                logger.info("Loaded cached long-term features from %s", cache_path)
                return cached_features

    timestamps, mid_prices, volumes = _load_series_from_snapshot(snapshot_dataset)

    logger.info(
        "Computing long-term features for dataset: n_samples=%d, lt_input_dim=%d",
        num_samples,
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

    if lt_features.shape[0] != num_samples:
        raise ConfigError(
            "Computed long-term feature count does not match anchor timestamps length: "
            f"features={lt_features.shape[0]}, anchors={num_samples}"
        )

    _save_long_term_cache(cache_path, lt_features, cache_meta)
    return lt_features


def wrap_generator_with_long_term(
    base_generator: Iterator[Tuple[Any, ...]],
    long_term_features: np.ndarray,
    start_index: int,
    end_index: Optional[int] = None,
) -> Iterator[Tuple[Any, ...]]:
    """Wrap a training generator to include long-term features.

    Parameters
    ----------
    base_generator:
        Original generator yielding (x, y) or (x, y, sample_weight).
        Must loop infinitely (``while True``).
    long_term_features:
        Precomputed long-term features array of shape (n_samples, lt_dim).
    start_index:
        Starting sample index for this generator.
    end_index:
        End sample index (exclusive) for this generator.  When the internal
        cursor reaches *end_index* it resets to *start_index*, keeping the
        wrapper in sync with the base generator's epoch-boundary reset.
        Defaults to ``long_term_features.shape[0]`` when not provided.

    Yields
    ------
    Tuple:
        ([x_short, x_long], y) or ([x_short, x_long], y, sample_weight)
        depending on whether the base generator yields sample weights.

    Notes
    -----
    This wrapper assumes that the base generator yields batches in order
    starting from start_index. The long-term features are sliced according
    to the actual batch size, so final partial batches are supported.
    """
    if end_index is None:
        end_index = int(long_term_features.shape[0])

    current_idx = start_index

    for batch_data in base_generator:
        batch_len = batch_data[0].shape[0]

        # Detect epoch boundary: the base generator looped back to the
        # beginning of its range while current_idx is still at the end.
        if current_idx >= end_index:
            current_idx = start_index

        end_idx = current_idx + batch_len

        lt_batch = long_term_features[current_idx:end_idx]
        if lt_batch.shape[0] != batch_len:
            raise ValueError(
                "Long-term feature batch size mismatch: "
                f"features={lt_batch.shape[0]}, expected={batch_len}. "
                f"current_idx={current_idx}, end_index={end_index}, "
                f"lt_features_len={long_term_features.shape[0]}"
            )
        current_idx = end_idx

        # Replace x with [x, lt_batch]
        x_short = batch_data[0]
        x_dual = (x_short, lt_batch)

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
    model_cfg = config["model"]
    lt_cfg = model_cfg["long_term"]
    return bool(lt_cfg["enabled"])


__all__ = [
    "compute_long_term_features_for_dataset",
    "load_anchor_timestamps",
    "load_snapshot_series",
    "wrap_generator_with_long_term",
    "get_long_term_input_dim",
    "is_long_term_enabled",
]
