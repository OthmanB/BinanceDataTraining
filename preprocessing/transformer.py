"""Preprocessing pipeline entrypoint (Phase 2 skeleton).

Later phases will implement normalization, feature engineering, and other
transformations based on the YAML configuration.
"""

from __future__ import annotations

from typing import Any, Dict
import logging

import numpy as np

from .validator import validate_data_object
from .time_utils import normalize_timestamp_array


logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(config: Dict[str, Any], data_object: Dict[str, Any]) -> Dict[str, Any]:
    """Run the preprocessing pipeline on a DataObject.

    Phase 2 implementation only performs structural validation and returns the
    DataObject unchanged. Future phases will add numeric transformations and
    feature engineering steps informed by the configuration.
    """

    validate_data_object(data_object)
    logger.info("Preprocessing pipeline executed (Phase 2: validation only).")

    # Build classification targets from order book rows for the target asset
    # using the configuration-defined labeling scheme.
    _build_targets_from_order_book(config, data_object)

    return data_object


def _validate_snapshot_timestamps_for_target_asset(
    data_cfg: Dict[str, Any],
    target_asset: str,
    snapshot_timestamps: list[Any],
) -> None:
    validation_cfg = data_cfg["validation"]
    check_missing_data = bool(validation_cfg["check_missing_data"])
    if not check_missing_data:
        return

    max_gap_seconds = int(validation_cfg["max_gap_seconds"])
    if max_gap_seconds <= 0:
        raise ValueError(
            "data.validation.max_gap_seconds must be positive when data.validation.check_missing_data is true",
        )

    fail_on_invalid = bool(validation_cfg["fail_on_invalid"])

    if not snapshot_timestamps:
        return

    ts_array = np.asarray(snapshot_timestamps, dtype="datetime64[ns]")
    if ts_array.size <= 1:
        return

    diffs = np.diff(ts_array.astype("datetime64[s]")).astype("int64")
    if diffs.size == 0:
        return

    min_diff = int(diffs.min())
    if min_diff < 0:
        message = (
            "Non-monotonic snapshot timestamps detected for target asset "
            f"{target_asset}: minimum_diff_seconds={min_diff}"
        )
        if fail_on_invalid:
            raise ValueError(message)
        logger.warning(message)
        return

    max_gap_observed = int(diffs.max())
    if max_gap_observed <= max_gap_seconds:
        return

    message = (
        "Snapshot timestamp gaps for target asset "
        f"{target_asset} exceed data.validation.max_gap_seconds; "
        f"max_gap_seconds={max_gap_seconds}, observed_max_gap_seconds={max_gap_observed}"
    )
    if fail_on_invalid:
        raise ValueError(message)
    logger.warning(message)


def _apply_gap_handling_to_snapshots(
    data_cfg: Dict[str, Any],
    targets_cfg: Dict[str, Any],
    snapshot_timestamps: list[Any],
    snapshot_features: list[list[float]],
    mid_price_values: list[float],
) -> tuple[list[Any], list[list[float]], list[float]]:
    labeling_cfg = targets_cfg["labeling"]
    handle_gaps = str(labeling_cfg["handle_gaps"])

    if handle_gaps not in {"skip", "forward_fill", "interpolate"}:
        raise ValueError(
            "Unsupported targets.labeling.handle_gaps value; expected 'skip', 'forward_fill', or 'interpolate'",
        )

    if not snapshot_timestamps:
        return snapshot_timestamps, snapshot_features, mid_price_values

    time_range_cfg = data_cfg["time_range"]
    cadence_seconds = int(time_range_cfg["cadence_seconds"])
    if cadence_seconds <= 0:
        raise ValueError("data.time_range.cadence_seconds must be positive")

    validation_cfg = data_cfg["validation"]
    max_gap_seconds = int(validation_cfg["max_gap_seconds"])
    if max_gap_seconds <= 0:
        raise ValueError("data.validation.max_gap_seconds must be positive")

    ts_array = np.asarray(snapshot_timestamps, dtype="datetime64[s]")
    mid_arr = np.asarray(mid_price_values, dtype="float64")

    n = int(ts_array.shape[0])
    if n <= 1:
        return list(ts_array.astype("datetime64[ns]")), snapshot_features, list(mid_arr.astype("float64"))

    new_ts: list[Any] = []
    new_mid: list[float] = []
    new_feat: list[list[float]] = []

    new_ts.append(ts_array[0].astype("datetime64[ns]"))
    new_mid.append(float(mid_arr[0]))
    new_feat.append(snapshot_features[0])

    for i in range(n - 1):
        t0 = ts_array[i]
        t1 = ts_array[i + 1]
        m0 = float(mid_arr[i])
        m1 = float(mid_arr[i + 1])
        f0 = snapshot_features[i]
        f1 = snapshot_features[i + 1]

        gap_td = (t1 - t0).astype("timedelta64[s]")
        gap_secs = int(gap_td.astype("int64"))

        if gap_secs > cadence_seconds and gap_secs <= max_gap_seconds:
            missing_steps = gap_secs // cadence_seconds - 1
            if missing_steps > 0:
                if handle_gaps == "forward_fill":
                    for j in range(1, missing_steps + 1):
                        t_new = t0 + np.timedelta64(j * cadence_seconds, "s")
                        new_ts.append(t_new.astype("datetime64[ns]"))
                        new_mid.append(m0)
                        new_feat.append(f0)
                elif handle_gaps == "interpolate":
                    for j in range(1, missing_steps + 1):
                        t_new = t0 + np.timedelta64(j * cadence_seconds, "s")
                        alpha = j / float(missing_steps + 1)
                        mid_new = (1.0 - alpha) * m0 + alpha * m1
                        f_new: list[float] = []
                        for k in range(len(f0)):
                            try:
                                v0 = float(f0[k])
                                v1 = float(f1[k])
                            except (TypeError, ValueError):
                                v0 = 0.0
                                v1 = 0.0
                            f_new.append((1.0 - alpha) * v0 + alpha * v1)
                        new_ts.append(t_new.astype("datetime64[ns]"))
                        new_mid.append(mid_new)
                        new_feat.append(f_new)

        new_ts.append(t1.astype("datetime64[ns]"))
        new_mid.append(m1)
        new_feat.append(f1)

    return new_ts, new_feat, new_mid


def _build_targets_from_order_book(config: Dict[str, Any], data_object: Dict[str, Any]) -> None:
    """Populate data_object['targets'] and metadata['num_samples'] with labels.

    Labels are derived from percentage price changes over the configured prediction
    horizon, using the target asset's mid-price (bid/ask midpoint) from the
    order book rows.
    """

    metadata = data_object.get("metadata", {})
    n_samples = int(metadata.get("num_samples", 0))

    if n_samples <= 0:
        logger.info("Target construction skipped: num_samples=0.")
        return

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    order_books = data_object.get("order_books", {})
    target_book = order_books.get(target_asset, {})
    target_rows = target_book.get("rows") or []

    if not target_rows:
        logger.info("Target construction skipped: no order book rows for target asset %s.", target_asset)
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    targets_cfg = config["targets"]
    price_classes_cfg = targets_cfg["price_classes"]
    labeling_cfg = targets_cfg["labeling"]

    definition_type = str(price_classes_cfg["definition_type"])
    if definition_type != "percentage":
        raise ValueError("Only percentage-based price_classes.definition_type is supported in this phase")

    labeling_scheme = str(labeling_cfg["scheme"])
    if labeling_scheme != "two_head_intensity":
        raise ValueError("Only targets.labeling.scheme='two_head_intensity' is supported in this phase")

    use_midpoint = bool(labeling_cfg["use_midpoint"])
    if not use_midpoint:
        raise ValueError("Only labeling.use_midpoint=true is supported in this phase")

    boundaries = price_classes_cfg["boundaries"]
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("targets.price_classes.boundaries must be a non-empty list in configuration")

    boundaries_float = [float(b) for b in boundaries]

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in this phase")

    num_classes = int(output_cfg["num_classes"])

    expected_intensity_classes = len(boundaries_float) + 1
    if num_classes != expected_intensity_classes:
        raise ValueError(
            "For two_head_intensity, model.output.num_classes must equal len(targets.price_classes.boundaries) + 1",
        )

    time_range_cfg = data_cfg["time_range"]
    cadence_seconds = int(time_range_cfg["cadence_seconds"])
    if cadence_seconds <= 0:
        raise ValueError("data.time_range.cadence_seconds must be positive")

    prediction_horizon_seconds = int(targets_cfg["prediction_horizon_seconds"])
    if prediction_horizon_seconds <= 0:
        raise ValueError("targets.prediction_horizon_seconds must be positive")

    horizon_steps = prediction_horizon_seconds // cadence_seconds
    if horizon_steps <= 0:
        raise ValueError(
            "prediction_horizon_seconds must be at least data.time_range.cadence_seconds to define a future step",
        )

    visible_window_seconds = int(targets_cfg["visible_window_seconds"])
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

    # Build mid-prices at the snapshot level by aggregating rows that share the
    # same timestamp. The batch_id column encodes a coarse time bucket and must
    # not be used to define snapshot identity.
    snapshots = {}
    snapshot_features = []
    snapshot_timestamps = []
    for row_index, row in enumerate(target_rows):
        if len(row) < 6:
            raise ValueError(
                f"Order book row {row_index} has insufficient columns for snapshot reconstruction: "
                f"expected at least 6, got {len(row)}",
            )

        ts_value = row[0]
        key = ts_value

        snapshot_state = snapshots.get(key)
        if snapshot_state is None:
            snapshot_state = {
                "best_bid_price": None,
                "best_bid_qty": None,
                "best_ask_price": None,
                "best_ask_qty": None,
            }
            snapshots[key] = snapshot_state

        try:
            bid_price = float(row[1])
            bid_qty = float(row[2])
            ask_price = float(row[3])
            ask_qty = float(row[4])
        except (TypeError, ValueError, IndexError) as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse bid/ask price or quantity from order book row {row_index}: {exc}") from exc

        if bid_price > 0.0 and bid_qty >= 0.0:
            current_bid_price = snapshot_state["best_bid_price"]
            if current_bid_price is None or bid_price > current_bid_price:
                snapshot_state["best_bid_price"] = bid_price
                snapshot_state["best_bid_qty"] = bid_qty

        if ask_price > 0.0 and ask_qty >= 0.0:
            current_ask_price = snapshot_state["best_ask_price"]
            if current_ask_price is None or ask_price < current_ask_price:
                snapshot_state["best_ask_price"] = ask_price
                snapshot_state["best_ask_qty"] = ask_qty

    if not snapshots:
        logger.info(
            "Target construction skipped: no snapshots could be reconstructed for target asset %s.",
            target_asset,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    sorted_keys = sorted(snapshots.keys())

    # Normalize snapshot timestamps once so that all downstream consumers
    # (diagnostics, training, evaluation) share the same datetime64[ns]
    # representation.
    normalized_ts = normalize_timestamp_array(sorted_keys)

    mid_price_values = []
    for key, ts_norm in zip(sorted_keys, normalized_ts):
        snapshot_state = snapshots[key]
        best_bid_price = snapshot_state["best_bid_price"]
        best_bid_qty = snapshot_state["best_bid_qty"]
        best_ask_price = snapshot_state["best_ask_price"]
        best_ask_qty = snapshot_state["best_ask_qty"]

        if best_bid_price is None or best_ask_price is None:
            continue

        if best_bid_price <= 0.0 or best_ask_price <= 0.0:
            continue

        mid_price_values.append(0.5 * (best_bid_price + best_ask_price))
        snapshot_features.append([
            best_bid_price,
            best_bid_qty if best_bid_qty is not None else 0.0,
            best_ask_price,
            best_ask_qty if best_ask_qty is not None else 0.0,
        ])
        snapshot_timestamps.append(ts_norm)

    if not mid_price_values:
        logger.info(
            "Target construction skipped: no snapshots with valid mid-prices for target asset %s.",
            target_asset,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    _validate_snapshot_timestamps_for_target_asset(data_cfg, target_asset, snapshot_timestamps)

    snapshot_timestamps, snapshot_features, mid_price_values = _apply_gap_handling_to_snapshots(
        data_cfg,
        targets_cfg,
        snapshot_timestamps,
        snapshot_features,
        mid_price_values,
    )

    mid_prices = np.asarray(mid_price_values, dtype="float64")
    num_snapshots = int(mid_prices.shape[0])

    target_book["snapshot_features"] = snapshot_features
    target_book["snapshot_timestamps"] = snapshot_timestamps
    order_books[target_asset] = target_book
    data_object["order_books"] = order_books

    if horizon_steps >= num_snapshots:
        logger.info(
            "Target construction skipped: horizon_steps=%s >= available_snapshots=%s.",
            horizon_steps,
            num_snapshots,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    # Require that each effective sample has both a full visible window in the past
    # and a full prediction horizon in the future.
    if num_snapshots <= horizon_steps + (window_steps - 1):
        logger.info(
            "Target construction skipped: insufficient snapshots for visible_window_seconds=%s "
            "and horizon_steps=%s: available_snapshots=%s.",
            visible_window_seconds,
            horizon_steps,
            num_snapshots,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    effective_samples = num_snapshots - horizon_steps - (window_steps - 1)
    if effective_samples <= 0:
        logger.info(
            "Target construction resulted in no effective samples after applying visible window: "
            "rows=%s, horizon_steps=%s, window_steps=%s.",
            len(target_rows),
            horizon_steps,
            window_steps,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    # Anchor indices map each sample index to the underlying snapshot index used as
    # the reference for label construction.
    anchor_start = window_steps - 1
    anchor_indices = list(range(anchor_start, anchor_start + effective_samples))

    labels = np.zeros(effective_samples, dtype="int64")
    price_changes = np.zeros(effective_samples, dtype="float64")
    max_up_moves = np.zeros(effective_samples, dtype="float64")
    max_down_moves = np.zeros(effective_samples, dtype="float64")
    labels_up_intensity = np.zeros(effective_samples, dtype="int64")
    labels_down_intensity = np.zeros(effective_samples, dtype="int64")

    for i, anchor_idx in enumerate(anchor_indices):
        p0 = mid_prices[anchor_idx]
        future_window = mid_prices[anchor_idx + 1 : anchor_idx + horizon_steps + 1]

        if future_window.size == 0:
            max_up = 0.0
            max_down = 0.0
        else:
            rel_moves = (future_window - p0) / p0 * 100.0
            max_up = float(np.max(rel_moves))
            max_down = float(np.min(rel_moves))

        max_up_moves[i] = max_up
        max_down_moves[i] = max_down

        # Single-head scalar used for current multi-class labels: choose the
        # move (up or down) with the largest absolute magnitude within the
        # horizon.
        if abs(max_up) >= abs(max_down):
            change_pct = max_up
        else:
            change_pct = max_down

        price_changes[i] = change_pct

        # Derive a diagnostic single-head class index from the signed
        # change_pct and the magnitude thresholds in boundaries_float.
        # The central bin represents small moves; bins farther from the
        # center represent larger moves in either direction.
        magnitude = abs(change_pct)
        mag_bin = 0
        for idx, boundary in enumerate(boundaries_float):
            if magnitude <= boundary:
                mag_bin = idx
                break
        else:
            mag_bin = len(boundaries_float)

        center_idx = len(boundaries_float)
        if mag_bin == 0 or change_pct == 0.0:
            cls_idx = center_idx
        else:
            if change_pct > 0.0:
                cls_idx = center_idx + mag_bin
            else:
                cls_idx = center_idx - mag_bin

        labels[i] = cls_idx

        # Two-head intensity labels based on the maximum upward and downward
        # moves within the prediction horizon, using the same magnitude
        # thresholds for both heads.
        up_intensity = max(max_up, 0.0)
        down_intensity = max(-max_down, 0.0)

        up_bin = 0
        for idx, boundary in enumerate(boundaries_float):
            if up_intensity <= boundary:
                up_bin = idx
                break
        else:
            up_bin = len(boundaries_float)

        down_bin = 0
        for idx, boundary in enumerate(boundaries_float):
            if down_intensity <= boundary:
                down_bin = idx
                break
        else:
            down_bin = len(boundaries_float)

        labels_up_intensity[i] = up_bin
        labels_down_intensity[i] = down_bin

    metadata["num_samples"] = int(effective_samples)
    metadata["anchor_indices"] = anchor_indices
    data_object["metadata"] = metadata

    targets = data_object.get("targets", {})
    targets["asset"] = target_asset
    targets["labels"] = labels.tolist()
    targets["price_changes"] = price_changes.tolist()
    targets["max_up_move_pct"] = max_up_moves.tolist()
    targets["max_down_move_pct"] = max_down_moves.tolist()
    targets["labels_up_intensity"] = labels_up_intensity.tolist()
    targets["labels_down_intensity"] = labels_down_intensity.tolist()
    targets["delta_t_seconds"] = prediction_horizon_seconds
    data_object["targets"] = targets

    logger.info(
        "Targets constructed for asset=%s. num_samples=%s, horizon_seconds=%s, num_classes=%s",
        target_asset,
        effective_samples,
        prediction_horizon_seconds,
        num_classes,
    )


__all__ = ["run_preprocessing_pipeline"]
