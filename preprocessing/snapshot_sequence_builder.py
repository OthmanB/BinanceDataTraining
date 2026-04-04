"""Build snapshot sequences for training from order book rows."""

from collections.abc import Iterable, Sequence
from typing import Any, SupportsFloat, cast

import numpy as np
from numpy.typing import NDArray

from .depth_aggregator import aggregate_snapshot_to_hybrid, get_hybrid_output_shape


def _validate_anchor_and_sample_indices(
    *,
    anchor_arr: NDArray[np.int64],
    sample_idx_arr: NDArray[np.int64],
    num_snapshots: int,
    source_name: str,
) -> None:
    if anchor_arr.ndim != 1:
        raise ValueError("anchor_indices must be a one-dimensional list of integers")

    if sample_idx_arr.ndim != 1:
        raise ValueError("sample_indices must be a one-dimensional iterable of integers")

    if sample_idx_arr.size == 0:
        return

    if anchor_arr.size == 0:
        raise ValueError("anchor_indices must not be empty when sample_indices is non-empty")

    if anchor_arr.min() < 0 or anchor_arr.max() >= num_snapshots:
        raise ValueError(
            "anchor_indices must be valid snapshot indices; "
            f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_{source_name}={num_snapshots}",
        )

    if sample_idx_arr.min() < 0 or sample_idx_arr.max() >= anchor_arr.shape[0]:
        raise ValueError(
            "sample_indices must be valid indices into anchor_indices; "
            f"got min={sample_idx_arr.min()}, max={sample_idx_arr.max()}, num_samples={anchor_arr.shape[0]}",
        )


def _coerce_top_of_book_features(
    *,
    features: Iterable[object],
    out_idx: int,
    t_idx: int,
    anchor_idx: int,
) -> tuple[float, float, float, float]:
    feature_list = list(features)
    if len(feature_list) < 4:
        raise ValueError("snapshot_features entries must include bid/ask price and quantity")

    try:
        bid_price = float(cast(SupportsFloat | str, feature_list[0]))
        bid_qty = float(cast(SupportsFloat | str, feature_list[1]))
        ask_price = float(cast(SupportsFloat | str, feature_list[2]))
        ask_qty = float(cast(SupportsFloat | str, feature_list[3]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "snapshot_features values must be numeric for top-of-book tensor construction; "
            f"sample_out_idx={out_idx}, time_idx={t_idx}, anchor_idx={anchor_idx}, features={feature_list!r}"
        ) from exc

    return bid_price, bid_qty, ask_price, ask_qty


def build_top_of_book_sequence_tensor(
    config: dict[str, Any],
    snapshot_features: Sequence[Iterable[object]],
    anchor_indices: list[int],
    sample_indices: Iterable[int],
    height: int,
    width: int,
    channels: int,
) -> NDArray[np.float32]:
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
    sample_idx_arr = np.asarray(list(sample_indices), dtype="int64")

    if sample_idx_arr.size == 0:
        return np.zeros((0, window_steps, height, width, channels), dtype="float32")

    _validate_anchor_and_sample_indices(
        anchor_arr=anchor_arr,
        sample_idx_arr=sample_idx_arr,
        num_snapshots=len(snapshot_features),
        source_name="snapshots",
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
            bid_price, bid_qty, ask_price, ask_qty = _coerce_top_of_book_features(
                features=features,
                out_idx=out_idx,
                t_idx=t_idx,
                anchor_idx=anchor_idx,
            )
            x_seq[out_idx, t_idx, 0, 0, 0] = bid_price
            if width > 1:
                x_seq[out_idx, t_idx, 0, 1, 0] = bid_qty
            if height > 1:
                x_seq[out_idx, t_idx, 1, 0, 0] = ask_price
            if height > 1 and width > 1:
                x_seq[out_idx, t_idx, 1, 1, 0] = ask_qty

    return x_seq


def build_hybrid_depth_sequence_tensor(
    config: dict[str, Any],
    snapshot_depth_data: list[dict[str, Any]],
    anchor_indices: list[int],
    sample_indices: Iterable[int],
) -> NDArray[np.float32]:
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
    sample_idx_arr = np.asarray(list(sample_indices), dtype="int64")

    if sample_idx_arr.size == 0:
        return np.zeros((0, window_steps, effective_levels, 4, 1), dtype="float32")

    _validate_anchor_and_sample_indices(
        anchor_arr=anchor_arr,
        sample_idx_arr=sample_idx_arr,
        num_snapshots=num_snapshots,
        source_name="snapshots",
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
            aggregated = aggregate_snapshot_to_hybrid(
                bid_prices=np.asarray(snapshot["bid_prices"], dtype="float32"),
                bid_quantities=np.asarray(snapshot["bid_quantities"], dtype="float32"),
                ask_prices=np.asarray(snapshot["ask_prices"], dtype="float32"),
                ask_quantities=np.asarray(snapshot["ask_quantities"], dtype="float32"),
                config=config,
            )
            if aggregated.shape[0] != effective_levels:
                raise ValueError("Hybrid snapshot depth levels mismatch")
            x_seq[out_idx, t_idx, :, :, 0] = aggregated

    return x_seq


__all__ = ["build_top_of_book_sequence_tensor", "build_hybrid_depth_sequence_tensor"]
