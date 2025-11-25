from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def build_top_of_book_sequence_tensor(
    config: Dict[str, Any],
    snapshot_features: List[List[float]] | List[Iterable[float]],
    anchor_indices: List[int],
    sample_indices: Iterable[int],
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    """Build a temporal sequence tensor from top-of-book snapshot features.

    The returned array has shape (N, T, H, W, C), where:
    - N is the number of samples (len(sample_indices)),
    - T is derived from targets.visible_window_seconds and data.time_range.cadence_seconds,
    - (H, W, C) are the spatial dimensions required by the CNN.

    Each sample index i corresponds to anchor_indices[i], which is the index of
    the snapshot used as the temporal anchor for that sample. For each sample,
    the builder collects T snapshots ending at the anchor (inclusive) and maps
    the top-of-book features into the top-left 2x2 patch of the spatial grid
    for each time step.
    """

    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    cadence_seconds = int(time_range_cfg["cadence_seconds"])

    targets_cfg = config["targets"]
    visible_window_seconds = int(targets_cfg["visible_window_seconds"])

    if cadence_seconds <= 0:
        raise ValueError("data.time_range.cadence_seconds must be positive")
    if visible_window_seconds <= 0:
        raise ValueError("targets.visible_window_seconds must be positive")
    if visible_window_seconds % cadence_seconds != 0:
        raise ValueError(
            "targets.visible_window_seconds must be an integer multiple of data.time_range.cadence_seconds",
        )

    window_steps = visible_window_seconds // cadence_seconds
    if window_steps <= 0:
        raise ValueError(
            "Derived visible window length in steps must be at least one snapshot; "
            f"visible_window_seconds={visible_window_seconds}, cadence_seconds={cadence_seconds}",
        )

    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("height, width, and channels must be positive integers")

    num_snapshots = len(snapshot_features)
    if num_snapshots == 0:
        raise ValueError("snapshot_features must be non-empty to build sequence tensors")

    anchor_arr = np.asarray(anchor_indices, dtype="int64")
    if anchor_arr.ndim != 1:
        raise ValueError("anchor_indices must be a one-dimensional list of integers")

    sample_idx_arr = np.asarray(list(sample_indices), dtype="int64")
    if sample_idx_arr.ndim != 1:
        raise ValueError("sample_indices must be a one-dimensional iterable of integers")

    if sample_idx_arr.size == 0:
        return np.zeros((0, window_steps, height, width, channels), dtype="float32")

    if anchor_arr.min() < 0 or anchor_arr.max() >= num_snapshots:
        raise ValueError(
            "anchor_indices must be valid snapshot indices; "
            f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_snapshots={num_snapshots}",
        )

    # Ensure that sampled indices are valid sample indices relative to anchor_indices.
    if sample_idx_arr.min() < 0 or sample_idx_arr.max() >= anchor_arr.shape[0]:
        raise ValueError(
            "sample_indices must be valid indices into anchor_indices; "
            f"got min={sample_idx_arr.min()}, max={sample_idx_arr.max()}, num_samples={anchor_arr.shape[0]}",
        )

    n_samples = int(sample_idx_arr.shape[0])
    x_seq = np.zeros((n_samples, window_steps, height, width, channels), dtype="float32")

    for s_idx, sample_i in enumerate(sample_idx_arr):
        anchor_snapshot_idx = int(anchor_arr[int(sample_i)])

        # Temporal positions are ordered from oldest to newest within the window.
        for tau in range(window_steps):
            # Offset from the start of the window to the current time step.
            # Oldest snapshot index is anchor_snapshot_idx - (window_steps - 1).
            snapshot_idx = anchor_snapshot_idx - (window_steps - 1 - tau)

            if snapshot_idx < 0 or snapshot_idx >= num_snapshots:
                continue

            features = snapshot_features[snapshot_idx]
            if not isinstance(features, (list, tuple)) or len(features) < 4:
                continue

            bid_price, bid_qty, ask_price, ask_qty = features[:4]

            try:
                bid_price_f = float(bid_price)
                bid_qty_f = float(bid_qty)
                ask_price_f = float(ask_price)
                ask_qty_f = float(ask_qty)
            except (TypeError, ValueError):
                # Leave this time step as zeros if parsing fails.
                continue

            # Map into top-left 2x2 patch of the spatial grid for this time step.
            # Remaining cells stay zero for this baseline implementation.
            x_seq[s_idx, tau, 0, 0, 0] = bid_price_f
            if width > 1:
                x_seq[s_idx, tau, 0, 1, 0] = bid_qty_f
            if height > 1:
                x_seq[s_idx, tau, 1, 0, 0] = ask_price_f
            if height > 1 and width > 1:
                x_seq[s_idx, tau, 1, 1, 0] = ask_qty_f

    return x_seq
