"""Build snapshot sequences for training from order book rows."""

from typing import Any, Dict, Iterable, List

import numpy as np

from .depth_aggregator import aggregate_snapshot_to_hybrid, get_hybrid_output_shape


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
        raise ValueError("Model input height/width/channels must be positive")

    anchor_arr = np.asarray(anchor_indices, dtype="int64")
    if anchor_arr.ndim != 1:
        raise ValueError("anchor_indices must be a one-dimensional list of integers")

    sample_idx_arr = np.asarray(list(sample_indices), dtype="int64")
    if sample_idx_arr.ndim != 1:
        raise ValueError("sample_indices must be a one-dimensional iterable of integers")

    if sample_idx_arr.size == 0:
        return np.zeros((0, window_steps, height, width, channels), dtype="float32")

    if anchor_arr.min() < 0 or anchor_arr.max() >= len(snapshot_features):
        raise ValueError(
            "anchor_indices must be valid snapshot indices; "
            f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_snapshots={len(snapshot_features)}",
        )

    if sample_idx_arr.min() < 0 or sample_idx_arr.max() >= anchor_arr.shape[0]:
        raise ValueError(
            "sample_indices must be valid indices into anchor_indices; "
            f"got min={sample_idx_arr.min()}, max={sample_idx_arr.max()}, num_samples={anchor_arr.shape[0]}",
        )

    n_samples = int(sample_idx_arr.shape[0])
    x_seq = np.zeros((n_samples, window_steps, height, width, channels), dtype="float32")

    for out_idx, sample_idx in enumerate(sample_idx_arr):
        anchor_idx = int(anchor_arr[sample_idx])
        start_idx = anchor_idx - window_steps + 1
        end_idx = anchor_idx + 1
        if start_idx < 0:
            raise ValueError("Not enough snapshots to build full window for sample index")

        window = snapshot_features[start_idx:end_idx]
        if len(window) != window_steps:
            raise ValueError("Snapshot window length mismatch")

        for t_idx, features in enumerate(window):
            if len(features) < 4:
                raise ValueError("snapshot_features entries must include bid/ask price and quantity")
            bid_price, bid_qty, ask_price, ask_qty = features[:4]
            x_seq[out_idx, t_idx, 0, 0, 0] = float(bid_price)
            if width > 1:
                x_seq[out_idx, t_idx, 0, 1, 0] = float(bid_qty)
            if height > 1:
                x_seq[out_idx, t_idx, 1, 0, 0] = float(ask_price)
            if height > 1 and width > 1:
                x_seq[out_idx, t_idx, 1, 1, 0] = float(ask_qty)

    return x_seq


def build_hybrid_depth_sequence_tensor(
    config: Dict[str, Any],
    snapshot_depth_data: List[Dict[str, Any]],
    anchor_indices: List[int],
    sample_indices: Iterable[int],
) -> np.ndarray:
    """Build hybrid depth sequence tensor for CNN input.

    Output shape is (N, T, L, 4, 1), where:
    - N is sample count,
    - T is window length in steps,
    - L is depth levels + aggregated bins,
    - 4 is [bid_price, bid_quantity, ask_price, ask_quantity].
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

    num_snapshots = len(snapshot_depth_data)
    if num_snapshots == 0:
        raise ValueError("snapshot_depth_data must be non-empty to build sequence tensors")

    effective_levels = get_hybrid_output_shape(config)

    anchor_arr = np.asarray(anchor_indices, dtype="int64")
    if anchor_arr.ndim != 1:
        raise ValueError("anchor_indices must be a one-dimensional list of integers")

    sample_idx_arr = np.asarray(list(sample_indices), dtype="int64")
    if sample_idx_arr.ndim != 1:
        raise ValueError("sample_indices must be a one-dimensional iterable of integers")

    if sample_idx_arr.size == 0:
        return np.zeros((0, window_steps, effective_levels, 4, 1), dtype="float32")

    if anchor_arr.min() < 0 or anchor_arr.max() >= num_snapshots:
        raise ValueError(
            "anchor_indices must be valid snapshot indices; "
            f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_snapshots={num_snapshots}",
        )

    if sample_idx_arr.min() < 0 or sample_idx_arr.max() >= anchor_arr.shape[0]:
        raise ValueError(
            "sample_indices must be valid indices into anchor_indices; "
            f"got min={sample_idx_arr.min()}, max={sample_idx_arr.max()}, num_samples={anchor_arr.shape[0]}",
        )

    n_samples = int(sample_idx_arr.shape[0])
    x_seq = np.zeros((n_samples, window_steps, effective_levels, 4, 1), dtype="float32")

    for out_idx, sample_idx in enumerate(sample_idx_arr):
        anchor_idx = int(anchor_arr[sample_idx])
        start_idx = anchor_idx - window_steps + 1
        end_idx = anchor_idx + 1
        if start_idx < 0:
            raise ValueError("Not enough snapshots to build full window for sample index")

        window = snapshot_depth_data[start_idx:end_idx]
        if len(window) != window_steps:
            raise ValueError("Snapshot window length mismatch")

        for t_idx, snapshot in enumerate(window):
            aggregated = aggregate_snapshot_to_hybrid(config, snapshot)
            if aggregated.shape[0] != effective_levels:
                raise ValueError("Hybrid snapshot depth levels mismatch")
            x_seq[out_idx, t_idx, :, :, 0] = aggregated

    return x_seq


__all__ = ["build_top_of_book_sequence_tensor", "build_hybrid_depth_sequence_tensor"]
