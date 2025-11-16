"""Preprocessing pipeline entrypoint (Phase 2 skeleton).

Later phases will implement normalization, feature engineering, and other
transformations based on the YAML configuration.
"""

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

    use_midpoint = bool(labeling_cfg["use_midpoint"])
    if not use_midpoint:
        raise ValueError("Only labeling.use_midpoint=true is supported in this phase")

    boundaries = price_classes_cfg["boundaries"]
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("targets.price_classes.boundaries must be a non-empty list in configuration")

    boundaries_float = [float(b) for b in boundaries]

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    num_classes = int(output_cfg["num_classes"])

    expected_classes = len(boundaries_float) + 1
    if num_classes != expected_classes:
        raise ValueError(
            "model.output.num_classes must equal len(targets.price_classes.boundaries) + 1",
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

    effective_samples = num_snapshots - horizon_steps
    if effective_samples <= 0:
        logger.info(
            "Target construction resulted in no effective samples: rows=%s, horizon_steps=%s.",
            len(target_rows),
            horizon_steps,
        )
        metadata["num_samples"] = 0
        data_object["metadata"] = metadata
        return

    labels = np.zeros(effective_samples, dtype="int64")
    price_changes = np.zeros(effective_samples, dtype="float64")

    for i in range(effective_samples):
        p0 = mid_prices[i]
        p1 = mid_prices[i + horizon_steps]

        change_pct = (p1 - p0) / p0 * 100.0
        price_changes[i] = change_pct

        cls_idx = 0
        for idx, boundary in enumerate(boundaries_float):
            if change_pct <= boundary:
                cls_idx = idx
                break
        else:
            cls_idx = len(boundaries_float)

        labels[i] = cls_idx

    metadata["num_samples"] = int(effective_samples)
    data_object["metadata"] = metadata

    targets = data_object.get("targets", {})
    targets["asset"] = target_asset
    targets["labels"] = labels.tolist()
    targets["price_changes"] = price_changes.tolist()
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
