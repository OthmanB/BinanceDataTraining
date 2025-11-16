from typing import Any, Dict, Sequence
import logging
from pathlib import Path
import tempfile
import csv
import math

import numpy as np
import matplotlib  # type: ignore[import]

matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.ticker import AutoMinorLocator, FuncFormatter  # type: ignore[import]
import matplotlib.patches as mpatches  # type: ignore[import]

from preprocessing.time_utils import normalize_timestamp_array


logger = logging.getLogger(__name__)


def run_data_diagnostics(
    config: Dict[str, Any],
    data_object: Dict[str, Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    """Run pre-training data diagnostics on snapshot-level features.

    This stage operates on the same snapshot index space used for targets and
    train/validation/test splits. It computes basic spread statistics on a
    configuration-defined sample of training snapshots and logs metrics and a
    small CSV artifact to MLflow.
    """

    diagnostics_cfg = config["diagnostics"]
    enabled = bool(diagnostics_cfg["enabled"])
    if not enabled:
        logger.info("Data diagnostics disabled via configuration; skipping.")
        return

    sampling_cfg = diagnostics_cfg["sampling"]
    method = str(sampling_cfg["method"])
    num_samples = int(sampling_cfg["num_samples"])
    random_seed = int(sampling_cfg["random_seed"])

    spread_cfg = diagnostics_cfg["spread_checks"]
    suspicious_threshold_pct = float(spread_cfg["suspicious_threshold_pct"])
    high_spread_threshold_pct = float(spread_cfg["high_spread_threshold_pct"])
    max_high_spread_fraction = float(spread_cfg["max_high_spread_fraction"])

    quantity_cfg = diagnostics_cfg["quantity_checks"]
    enable_negative_quantity_checks = bool(quantity_cfg["enable_negative_checks"])
    enable_zero_stats = bool(quantity_cfg["enable_zero_stats"])

    outlier_cfg = diagnostics_cfg["outlier_checks"]
    outlier_checks_enabled = bool(outlier_cfg["enabled"])
    z_score_threshold = float(outlier_cfg["z_score_threshold"])
    min_nonzero_points = int(outlier_cfg["min_nonzero_points"])

    anomaly_cfg = diagnostics_cfg["anomaly_export"]
    anomaly_export_enabled = bool(anomaly_cfg["enabled"])
    anomaly_max_samples = int(anomaly_cfg["max_samples"])

    viz_cfg = diagnostics_cfg["visualization"]
    visualization_enabled = bool(viz_cfg["enabled"])
    time_series_enabled = bool(viz_cfg["time_series"])
    histograms_enabled = bool(viz_cfg["histograms"])
    histogram_bins = int(viz_cfg["histogram_bins"])

    heatmaps_cfg = viz_cfg["heatmaps"]
    heatmaps_enabled = bool(heatmaps_cfg["enabled"])
    heatmap_types = [str(t) for t in heatmaps_cfg["types"]]
    num_time_bins = int(heatmaps_cfg["num_time_bins"])
    num_spread_bins = int(heatmaps_cfg["num_spread_bins"])

    spread_clip_cfg = viz_cfg["spread_clipping"]
    spread_clipping_enabled = bool(spread_clip_cfg["enabled"])
    spread_clipping_num_sigma = float(spread_clip_cfg["num_sigma"])

    depth_heatmap_cfg = viz_cfg["depth_heatmap"]
    depth_heatmap_enabled = bool(depth_heatmap_cfg["enabled"])
    num_price_bins = int(depth_heatmap_cfg["num_price_bins"])

    label_checks_cfg = diagnostics_cfg["label_checks"]
    label_checks_enabled = bool(label_checks_cfg["enabled"])
    label_checks_num_examples = int(label_checks_cfg["num_examples"])
    label_checks_max_per_figure = int(label_checks_cfg["max_examples_per_figure"])

    gap_cfg = diagnostics_cfg["gap_checks"]
    gap_checks_enabled = bool(gap_cfg["enabled"])
    large_gap_multiplier = float(gap_cfg["large_gap_multiplier"])
    very_large_gap_multiplier = float(gap_cfg["very_large_gap_multiplier"])

    metadata = data_object["metadata"]
    n_samples = int(metadata["num_samples"])
    if n_samples <= 0:
        logger.info("Data diagnostics skipped: num_samples=0.")
        return

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    # Cadence is used for gap checks (snapshot_timestamps continuity).
    time_range_cfg = data_cfg["time_range"]
    cadence_seconds = int(time_range_cfg["cadence_seconds"])

    # Targets configuration (used for label-aware diagnostics and horizon/window semantics).
    targets_cfg = config["targets"]
    prediction_horizon_seconds = int(targets_cfg["prediction_horizon_seconds"])
    visible_window_seconds = int(targets_cfg["visible_window_seconds"])

    if prediction_horizon_seconds <= 0:
        raise ValueError("targets.prediction_horizon_seconds must be positive")
    if visible_window_seconds <= 0:
        raise ValueError("targets.visible_window_seconds must be positive")

    price_classes_cfg = targets_cfg["price_classes"]
    price_boundaries_cfg = price_classes_cfg["boundaries"]
    if not isinstance(price_boundaries_cfg, list) or not price_boundaries_cfg:
        raise ValueError("targets.price_classes.boundaries must be a non-empty list in configuration")

    price_boundaries_pct = [float(b) for b in price_boundaries_cfg]

    order_books = data_object.get("order_books", {})
    target_book = order_books.get(target_asset, {})
    snapshot_features = target_book.get("snapshot_features") or []
    snapshot_timestamps = target_book.get("snapshot_timestamps") or []

    if not snapshot_features:
        logger.info(
            "Data diagnostics skipped: no snapshot_features available for target asset %s.",
            target_asset,
        )
        return

    base_indices = np.asarray(list(train_idx), dtype="int64")
    if base_indices.size == 0:
        logger.info("Data diagnostics skipped: empty train indices.")
        return

    # Determine which snapshot indices to sample from the training set.
    if num_samples >= base_indices.size:
        sampled_indices = base_indices
    else:
        if method == "uniform":
            positions = np.linspace(0, base_indices.size - 1, num_samples, dtype="int64")
            sampled_indices = base_indices[positions]
        elif method == "random":
            rng = np.random.default_rng(random_seed)
            sampled_indices = rng.choice(base_indices, size=num_samples, replace=False)
        else:
            raise ValueError(
                "Unsupported diagnostics.sampling.method: expected 'uniform' or 'random', "
                f"got {method!r}",
            )

    # Optional labels for label-aware diagnostics (e.g., spread_vs_label heatmap).
    labels_array = None
    try:
        targets = data_object["targets"]
        labels_array = np.asarray(targets["labels"])
    except KeyError:
        labels_array = None

    # Convert snapshot timestamps to datetime64[ns] if available. The
    # normalization logic is centralized in preprocessing.time_utils so that
    # diagnostics, training, and evaluation all see the same representation.
    timestamps_array = None
    if snapshot_timestamps:
        try:
            timestamps_array = normalize_timestamp_array(snapshot_timestamps)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to convert snapshot_timestamps to datetime64 for diagnostics: %s",
                exc,
            )
            timestamps_array = None

    # Collect per-snapshot diagnostic records for the sampled indices.
    records = []

    for idx in sampled_indices:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(snapshot_features):
            continue

        features_i = snapshot_features[idx_int]
        if not isinstance(features_i, (list, tuple)) or len(features_i) < 4:
            continue

        bid_price, bid_qty, ask_price, ask_qty = features_i[:4]

        try:
            bid_price_f = float(bid_price)
            ask_price_f = float(ask_price)
            bid_qty_f = float(bid_qty)
            ask_qty_f = float(ask_qty)
        except (TypeError, ValueError):
            continue

        record: Dict[str, Any] = {
            "index": idx_int,
            "bid_price": bid_price_f,
            "ask_price": ask_price_f,
            "bid_quantity": bid_qty_f,
            "ask_quantity": ask_qty_f,
        }

        if labels_array is not None and 0 <= idx_int < labels_array.shape[0]:
            record["label"] = labels_array[idx_int]

        if timestamps_array is not None and 0 <= idx_int < timestamps_array.shape[0]:
            record["timestamp"] = timestamps_array[idx_int]

        # Flags for anomalies and quality checks.
        flags = {
            "negative_price": (bid_price_f < 0.0) or (ask_price_f < 0.0),
            "zero_bid_price": bid_price_f == 0.0,
            "zero_ask_price": ask_price_f == 0.0,
            "negative_quantity": (bid_qty_f < 0.0) or (ask_qty_f < 0.0),
            "zero_bid_quantity": bid_qty_f == 0.0,
            "zero_ask_quantity": ask_qty_f == 0.0,
            "inverted_spread": False,
            "high_spread": False,
        }

        # Spread diagnostics: only meaningful when both prices are positive.
        mid_price = float("nan")
        spread_pct = float("nan")
        valid_spread = False

        if bid_price_f > 0.0 and ask_price_f > 0.0:
            mid_price = 0.5 * (bid_price_f + ask_price_f)
            if mid_price > 0.0:
                spread_pct = (ask_price_f - bid_price_f) / mid_price * 100.0
                valid_spread = True

                if bid_price_f >= ask_price_f:
                    flags["inverted_spread"] = True

                if spread_pct > high_spread_threshold_pct:
                    flags["high_spread"] = True

        record["mid_price"] = mid_price
        record["spread_pct"] = spread_pct
        record["valid_spread"] = valid_spread
        record["flags"] = flags

        records.append(record)

    if not records:
        logger.info("Data diagnostics skipped: no valid snapshot records could be constructed.")
        return

    # Convert records into numeric arrays for metrics.
    bid_arr = np.asarray([r["bid_price"] for r in records], dtype="float64")
    ask_arr = np.asarray([r["ask_price"] for r in records], dtype="float64")
    bid_qty_arr = np.asarray([r["bid_quantity"] for r in records], dtype="float64")
    ask_qty_arr = np.asarray([r["ask_quantity"] for r in records], dtype="float64")
    spread_arr_all = np.asarray([r["spread_pct"] for r in records], dtype="float64")
    mid_arr_all = np.asarray([r["mid_price"] for r in records], dtype="float64")

    valid_spread_mask = np.isfinite(spread_arr_all)
    spread_arr = spread_arr_all[valid_spread_mask]

    # Mid-price consistency metrics, following the legacy script's spirit.
    mid_from_points = 0.5 * (bid_arr[valid_spread_mask] + ask_arr[valid_spread_mask])
    mean_mid_from_points = float(mid_from_points.mean()) if mid_from_points.size > 0 else 0.0
    if valid_spread_mask.any():
        mean_mid_from_means = 0.5 * float(bid_arr[valid_spread_mask].mean() + ask_arr[valid_spread_mask].mean())
    else:
        mean_mid_from_means = 0.0

    if mean_mid_from_points != 0.0:
        mid_mean_rel_diff = abs(mean_mid_from_points - mean_mid_from_means) / abs(mean_mid_from_points)
    else:
        mid_mean_rel_diff = 0.0

    # Spread-based quality metrics.
    if spread_arr.size > 0:
        fraction_suspicious = float((spread_arr > suspicious_threshold_pct).mean())
        fraction_high_spread = float((spread_arr > high_spread_threshold_pct).mean())
        spread_mean = float(spread_arr.mean())
        spread_std = float(spread_arr.std())
        spread_max = float(spread_arr.max())
    else:
        fraction_suspicious = 0.0
        fraction_high_spread = 0.0
        spread_mean = 0.0
        spread_std = 0.0
        spread_max = 0.0

    # Sigma-based clipping thresholds for visualizations (not used for metrics).
    spread_clip_low = None
    spread_clip_high = None
    if spread_clipping_enabled and spread_arr.size > 0 and spread_std > 0.0:
        spread_clip_low = spread_mean - spread_clipping_num_sigma * spread_std
        spread_clip_high = spread_mean + spread_clipping_num_sigma * spread_std

    # Quantity and price anomaly statistics.
    negative_price_flags = np.asarray([r["flags"]["negative_price"] for r in records], dtype=bool)
    zero_bid_price_flags = np.asarray([r["flags"]["zero_bid_price"] for r in records], dtype=bool)
    zero_ask_price_flags = np.asarray([r["flags"]["zero_ask_price"] for r in records], dtype=bool)
    negative_quantity_flags = np.asarray([r["flags"]["negative_quantity"] for r in records], dtype=bool)
    zero_bid_quantity_flags = np.asarray([r["flags"]["zero_bid_quantity"] for r in records], dtype=bool)
    zero_ask_quantity_flags = np.asarray([r["flags"]["zero_ask_quantity"] for r in records], dtype=bool)
    inverted_spread_flags = np.asarray([r["flags"]["inverted_spread"] for r in records], dtype=bool)
    high_spread_flags = np.asarray([r["flags"]["high_spread"] for r in records], dtype=bool)

    # Outlier detection for non-zero prices.
    bid_nonzero = bid_arr[bid_arr > 0.0]
    ask_nonzero = ask_arr[ask_arr > 0.0]

    bid_outliers_percent = 0.0
    ask_outliers_percent = 0.0

    if outlier_checks_enabled and bid_nonzero.size >= min_nonzero_points:
        bid_mean = float(bid_nonzero.mean())
        bid_std = float(bid_nonzero.std())
        if bid_std > 0.0:
            bid_z = np.abs((bid_nonzero - bid_mean) / bid_std)
            bid_outliers_percent = float((bid_z > z_score_threshold).mean() * 100.0)

    if outlier_checks_enabled and ask_nonzero.size >= min_nonzero_points:
        ask_mean = float(ask_nonzero.mean())
        ask_std = float(ask_nonzero.std())
        if ask_std > 0.0:
            ask_z = np.abs((ask_nonzero - ask_mean) / ask_std)
            ask_outliers_percent = float((ask_z > z_score_threshold).mean() * 100.0)

    # Gap checks on snapshot_timestamps across the entire series for the target asset.
    gap_metrics: Dict[str, float] = {}
    if gap_checks_enabled and timestamps_array is not None and timestamps_array.size > 1:
        try:
            ts_sorted = np.sort(timestamps_array)
            diffs = np.diff(ts_sorted).astype("timedelta64[s]").astype("float64")
            if diffs.size > 0:
                max_gap_seconds = float(diffs.max())
                large_threshold = large_gap_multiplier * float(cadence_seconds)
                very_large_threshold = very_large_gap_multiplier * float(cadence_seconds)

                fraction_large = float((diffs > large_threshold).mean())
                fraction_very_large = float((diffs > very_large_threshold).mean())

                gap_metrics = {
                    "diag_gap_max_seconds": max_gap_seconds,
                    "diag_gap_fraction_large": fraction_large,
                    "diag_gap_fraction_very_large": fraction_very_large,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute gap diagnostics from snapshot_timestamps: %s", exc)

    # Prepare MLflow logging.
    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for data diagnostics logging: %s", exc)
        return

    metrics: Dict[str, float] = {
        "diag_spread_mean_pct": spread_mean,
        "diag_spread_std_pct": spread_std,
        "diag_spread_max_pct": spread_max,
        "diag_spread_fraction_gt_threshold": fraction_suspicious,
        "diag_spread_fraction_high": fraction_high_spread,
        "diag_spread_high_allowed_fraction": max_high_spread_fraction,
        "diag_mid_mean_price": float(mean_mid_from_points),
        "diag_mid_mean_vs_bidask_means_rel_diff": float(mid_mean_rel_diff),
        "diag_bid_price_min": float(bid_arr.min()),
        "diag_bid_price_max": float(bid_arr.max()),
        "diag_bid_price_mean": float(bid_arr.mean()),
        "diag_bid_price_std": float(bid_arr.std()),
        "diag_ask_price_min": float(ask_arr.min()),
        "diag_ask_price_max": float(ask_arr.max()),
        "diag_ask_price_mean": float(ask_arr.mean()),
        "diag_ask_price_std": float(ask_arr.std()),
        "diag_bid_quantity_min": float(bid_qty_arr.min()),
        "diag_bid_quantity_max": float(bid_qty_arr.max()),
        "diag_bid_quantity_mean": float(bid_qty_arr.mean()),
        "diag_bid_quantity_std": float(bid_qty_arr.std()),
        "diag_ask_quantity_min": float(ask_qty_arr.min()),
        "diag_ask_quantity_max": float(ask_qty_arr.max()),
        "diag_ask_quantity_mean": float(ask_qty_arr.mean()),
        "diag_ask_quantity_std": float(ask_qty_arr.std()),
        "diag_bid_price_nonzero_count": float(bid_nonzero.size),
        "diag_ask_price_nonzero_count": float(ask_nonzero.size),
    }

    if enable_negative_quantity_checks:
        metrics.update(
            {
                "diag_negative_price_fraction": float(negative_price_flags.mean()),
                "diag_negative_quantity_fraction": float(negative_quantity_flags.mean()),
            },
        )

    if enable_zero_stats:
        metrics.update(
            {
                "diag_zero_bid_price_fraction": float(zero_bid_price_flags.mean()),
                "diag_zero_ask_price_fraction": float(zero_ask_price_flags.mean()),
                "diag_zero_bid_quantity_fraction": float(zero_bid_quantity_flags.mean()),
                "diag_zero_ask_quantity_fraction": float(zero_ask_quantity_flags.mean()),
            },
        )

    if outlier_checks_enabled:
        metrics.update(
            {
                "diag_bid_price_outliers_percent": bid_outliers_percent,
                "diag_ask_price_outliers_percent": ask_outliers_percent,
            },
        )

    if gap_metrics:
        metrics.update(gap_metrics)

    # Inverted and high-spread fractions are always informative.
    metrics.update(
        {
            "diag_inverted_spread_fraction": float(inverted_spread_flags.mean()),
            "diag_high_spread_fraction": float(high_spread_flags.mean()),
        },
    )

    for name, value in metrics.items():
        try:
            mlflow.log_metric(name, float(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log MLFlow data diagnostics metric %s: %s", name, exc)

    # Prepare artifacts (CSV samples, anomalies, and optional visualizations).
    tmp_dir = Path(tempfile.mkdtemp())

    # Use asset symbol in filenames so diagnostics are pair-specific.
    asset_suffix = target_asset.replace("/", "-")

    # Sample CSV with all records used for diagnostics.
    sample_csv_path = tmp_dir / f"data_diagnostics_sample_{asset_suffix}.csv"
    try:
        with sample_csv_path.open("w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(
                [
                    "sample_index",
                    "timestamp",
                    "label",
                    "bid_price",
                    "ask_price",
                    "bid_quantity",
                    "ask_quantity",
                    "mid_price",
                    "spread_pct",
                ],
            )
            for r in records:
                writer.writerow(
                    [
                        r["index"],
                        str(r.get("timestamp")) if r.get("timestamp") is not None else "",
                        r.get("label", ""),
                        r["bid_price"],
                        r["ask_price"],
                        r["bid_quantity"],
                        r["ask_quantity"],
                        r["mid_price"],
                        r["spread_pct"],
                    ],
                )

        mlflow.log_artifact(str(sample_csv_path), artifact_path="diagnostics")
        logger.info("Logged data diagnostics sample CSV artifact to MLFlow at %s", sample_csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log data diagnostics sample artifact to MLFlow: %s", exc)

    # Anomaly CSV focusing on the most critical issues (subset of records).
    if anomaly_export_enabled:
        anomalies = []
        for r in records:
            flags = r["flags"]
            anomaly_types = []
            if flags["negative_price"]:
                anomaly_types.append("negative_price")
            if flags["negative_quantity"]:
                anomaly_types.append("negative_quantity")
            if flags["inverted_spread"]:
                anomaly_types.append("inverted_spread")
            if flags["high_spread"]:
                anomaly_types.append("high_spread")

            if anomaly_types:
                anomaly_record = dict(r)
                anomaly_record["anomaly_type"] = "|".join(anomaly_types)
                anomalies.append(anomaly_record)

        if anomalies:
            anomalies_limited = anomalies[:anomaly_max_samples]
            anomalies_csv_path = tmp_dir / f"data_diagnostics_anomalies_{asset_suffix}.csv"
            try:
                with anomalies_csv_path.open("w", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(
                        [
                            "sample_index",
                            "timestamp",
                            "label",
                            "bid_price",
                            "ask_price",
                            "bid_quantity",
                            "ask_quantity",
                            "mid_price",
                            "spread_pct",
                            "anomaly_type",
                        ],
                    )
                    for r in anomalies_limited:
                        writer.writerow(
                            [
                                r["index"],
                                str(r.get("timestamp")) if r.get("timestamp") is not None else "",
                                r.get("label", ""),
                                r["bid_price"],
                                r["ask_price"],
                                r["bid_quantity"],
                                r["ask_quantity"],
                                r["mid_price"],
                                r["spread_pct"],
                                r["anomaly_type"],
                            ],
                        )

                mlflow.log_artifact(str(anomalies_csv_path), artifact_path="diagnostics")
                logger.info("Logged data diagnostics anomalies CSV artifact to MLFlow at %s", anomalies_csv_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log data diagnostics anomalies artifact to MLFlow: %s", exc)

    # Visual diagnostics: time series, histograms, and heatmaps.
    if visualization_enabled:
        # Numeric time axis for sampled records (seconds since first sample).
        time_axis = None
        if timestamps_array is not None and timestamps_array.size >= len(records):
            try:
                sample_times = np.asarray(
                    [r.get("timestamp") for r in records],
                    dtype="datetime64[ns]",
                )
                base_time = sample_times.min()
                time_axis = (sample_times - base_time).astype("timedelta64[s]").astype("float64")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to construct numeric time axis for diagnostics: %s", exc)
                time_axis = None

        # Time-series plots: bid/ask prices, spreads (pct and relative), and quantities.
        if time_series_enabled and time_axis is not None:
            try:
                # Prepare spread values for visualization with optional sigma clipping.
                spread_for_plot = spread_arr_all.copy()
                if spread_clip_low is not None and spread_clip_high is not None:
                    spread_for_plot = np.clip(spread_for_plot, spread_clip_low, spread_clip_high)

                # Relative spread as fraction of mid-price.
                relative_spread_for_plot = spread_for_plot / 100.0

                fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

                # Panel 1: best bid/ask prices.
                axes[0].plot(time_axis, bid_arr, label="best_bid_price")
                axes[0].plot(time_axis, ask_arr, label="best_ask_price")
                axes[0].set_ylabel("Price")
                axes[0].legend()

                # Panel 2: spread in percent (optionally clipped).
                axes[1].plot(
                    time_axis[valid_spread_mask],
                    spread_for_plot[valid_spread_mask],
                    label="spread_pct (clipped)" if spread_clip_low is not None else "spread_pct",
                    color="orange",
                )
                axes[1].set_ylabel("Spread %")
                axes[1].legend()

                # Panel 3: relative spread (dimensionless fraction of mid-price).
                axes[2].plot(
                    time_axis[valid_spread_mask],
                    relative_spread_for_plot[valid_spread_mask],
                    label="relative_spread",
                    color="green",
                )
                axes[2].set_ylabel("Rel. spread")
                axes[2].legend()

                # Panel 4: bid/ask quantities.
                axes[3].plot(time_axis, bid_qty_arr, label="bid_quantity")
                axes[3].plot(time_axis, ask_qty_arr, label="ask_quantity")
                axes[3].set_ylabel("Quantity")
                axes[3].set_xlabel("Seconds since first sampled snapshot")
                axes[3].legend()

                fig.suptitle(
                    f"Data diagnostics time series for {target_asset} (train sample, N={len(records)})",
                )

                # Secondary y-axis on the price panel in percentage relative to
                # a reference mid-price.
                ref_price = float(mean_mid_from_points) if mean_mid_from_points > 0.0 else 0.0
                if ref_price > 0.0:
                    y_min_price, y_max_price = axes[0].get_ylim()
                    pct_min = (y_min_price / ref_price - 1.0) * 100.0
                    pct_max = (y_max_price / ref_price - 1.0) * 100.0

                    sec_ax_price = axes[0].twinx()
                    sec_ax_price.set_ylim(pct_min, pct_max)
                    sec_ax_price.set_ylabel("Price change % vs ref")
                    sec_ax_price.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
                    sec_ax_price.yaxis.set_minor_locator(AutoMinorLocator())

                # Minor ticks on all primary axes for better resolution.
                for ax in axes:
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
                    ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Secondary x-axis with datetime labels at the top figure.
                # We reuse the same numeric time coordinate but display
                # human-readable timestamps using the base_time reference.
                try:
                    # base_time was defined when constructing time_axis.
                    top_ax = axes[0].twiny()
                    top_ax.set_xlim(axes[0].get_xlim())

                    major_ticks = axes[3].get_xticks()
                    datetime_labels = []
                    for t_val in major_ticks:
                        dt_val = base_time + np.timedelta64(int(t_val), "s")
                        # Compact datetime format for tight spacing.
                        datetime_labels.append(str(dt_val)[:16])  # YYYY-MM-DDTHH:MM

                    top_ax.set_xticks(major_ticks)
                    top_ax.set_xticklabels(datetime_labels)

                    for label in top_ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_fontsize(6)

                    top_ax.xaxis.set_minor_locator(AutoMinorLocator())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to construct secondary datetime x-axis for time series: %s", exc)

                time_series_path = tmp_dir / f"diagnostics_time_series_{asset_suffix}.png"
                fig.tight_layout()
                fig.savefig(time_series_path)
                plt.close(fig)

                mlflow.log_artifact(str(time_series_path), artifact_path="diagnostics")
                logger.info(
                    "Logged data diagnostics time-series plot artifact to MLFlow at %s",
                    time_series_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate/log data diagnostics time-series plot: %s", exc)

        # Histograms of non-zero bid/ask prices.
        if histograms_enabled:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))

                if bid_nonzero.size > 0:
                    ax.hist(
                        bid_nonzero,
                        bins=histogram_bins,
                        alpha=0.6,
                        label="bid_price_nonzero",
                    )
                if ask_nonzero.size > 0:
                    ax.hist(
                        ask_nonzero,
                        bins=histogram_bins,
                        alpha=0.6,
                        label="ask_price_nonzero",
                    )

                ax.set_xlabel("Price")
                ax.set_ylabel("Count")
                ax.set_title(f"Non-zero bid/ask price distribution for {target_asset}")
                ax.legend()

                hist_path = tmp_dir / f"diagnostics_histograms_{asset_suffix}.png"
                fig.tight_layout()
                fig.savefig(hist_path)
                plt.close(fig)

                mlflow.log_artifact(str(hist_path), artifact_path="diagnostics")
                logger.info(
                    "Logged data diagnostics histogram plot artifact to MLFlow at %s",
                    hist_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate/log data diagnostics histograms: %s", exc)

        # Heatmaps: spread vs time and spread vs label (if configured).
        if heatmaps_enabled and time_axis is not None and spread_arr_all.size > 0:
            # Spread vs time heatmap.
            if "spread_vs_time" in heatmap_types:
                try:
                    mask = np.isfinite(spread_arr_all) & np.isfinite(time_axis)
                    if mask.any():
                        t_vals = time_axis[mask]
                        s_vals = spread_arr_all[mask]

                        # Apply sigma-based clipping for heatmap visualization if configured.
                        if spread_clip_low is not None and spread_clip_high is not None:
                            s_vals = np.clip(s_vals, spread_clip_low, spread_clip_high)

                        t_min, t_max = float(t_vals.min()), float(t_vals.max())
                        s_min, s_max = float(s_vals.min()), float(s_vals.max())

                        if t_max > t_min and s_max > s_min:
                            t_edges = np.linspace(t_min, t_max, num_time_bins + 1)
                            s_edges = np.linspace(s_min, s_max, num_spread_bins + 1)

                            hist2d, xedges, yedges = np.histogram2d(
                                t_vals,
                                s_vals,
                                bins=[t_edges, s_edges],
                            )

                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            mesh = ax.pcolormesh(
                                xedges,
                                yedges,
                                hist2d.T,
                                shading="auto",
                            )
                            fig.colorbar(mesh, ax=ax, label="Count")
                            ax.set_xlabel("Seconds since first sampled snapshot")
                            ax.set_ylabel("Spread %")
                            ax.set_title(f"Spread vs time heatmap for {target_asset}")

                            heatmap_time_path = tmp_dir / f"heatmap_spread_vs_time_{asset_suffix}.png"
                            fig.tight_layout()
                            fig.savefig(heatmap_time_path)
                            plt.close(fig)

                            mlflow.log_artifact(str(heatmap_time_path), artifact_path="diagnostics")
                            logger.info(
                                "Logged data diagnostics spread-vs-time heatmap artifact to MLFlow at %s",
                                heatmap_time_path,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to generate/log data diagnostics spread-vs-time heatmap: %s",
                        exc,
                    )

            # Spread vs label heatmap.
            if "spread_vs_label" in heatmap_types and labels_array is not None:
                try:
                    # Assemble spread and label arrays for records that have labels.
                    spreads_for_label = []
                    labels_for_label = []
                    for r in records:
                        if "label" in r and np.isfinite(r["spread_pct"]):
                            spreads_for_label.append(float(r["spread_pct"]))
                            labels_for_label.append(float(r["label"]))

                    if spreads_for_label:
                        spreads_for_label_arr = np.asarray(spreads_for_label, dtype="float64")
                        labels_for_label_arr = np.asarray(labels_for_label, dtype="float64")

                        # Apply sigma-based clipping for label heatmap if configured.
                        if spread_clip_low is not None and spread_clip_high is not None:
                            spreads_for_label_arr = np.clip(
                                spreads_for_label_arr,
                                spread_clip_low,
                                spread_clip_high,
                            )

                        s_min = float(spreads_for_label_arr.min())
                        s_max = float(spreads_for_label_arr.max())

                        if s_max > s_min:
                            # Use model.output.num_classes to determine label axis range.
                            model_cfg = config["model"]
                            output_cfg = model_cfg["output"]
                            num_classes = int(output_cfg["num_classes"])

                            x_edges = np.linspace(-0.5, num_classes - 0.5, num_classes + 1)
                            y_edges = np.linspace(s_min, s_max, num_spread_bins + 1)

                            hist2d, xedges, yedges = np.histogram2d(
                                labels_for_label_arr,
                                spreads_for_label_arr,
                                bins=[x_edges, y_edges],
                            )

                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            mesh = ax.pcolormesh(
                                xedges,
                                yedges,
                                hist2d.T,
                                shading="auto",
                            )
                            fig.colorbar(mesh, ax=ax, label="Count")
                            ax.set_xlabel("Label index")
                            ax.set_ylabel("Spread %")
                            ax.set_title(f"Spread vs label heatmap for {target_asset}")

                            heatmap_label_path = tmp_dir / f"heatmap_spread_vs_label_{asset_suffix}.png"
                            fig.tight_layout()
                            fig.savefig(heatmap_label_path)
                            plt.close(fig)

                            mlflow.log_artifact(str(heatmap_label_path), artifact_path="diagnostics")
                            logger.info(
                                "Logged data diagnostics spread-vs-label heatmap artifact to MLFlow at %s",
                                heatmap_label_path,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to generate/log data diagnostics spread-vs-label heatmap: %s",
                        exc,
                    )

        # Depth volume heatmap: time vs price with volume as color, based on
        # rebinned depth for the sampled timestamps.
        if depth_heatmap_enabled and time_axis is not None:
            try:
                target_rows_all = target_book.get("rows") or []
                if target_rows_all and timestamps_array is not None and timestamps_array.size > 0:
                    # Map normalized snapshot timestamps to numeric time coordinates
                    # derived from the sampled records.
                    ts_to_timecoord: Dict[Any, float] = {}
                    for record, t_coord in zip(records, time_axis):
                        idx_int = record["index"]
                        if 0 <= idx_int < timestamps_array.shape[0]:
                            ts_key = timestamps_array[idx_int]
                            ts_to_timecoord[ts_key] = t_coord

                    if ts_to_timecoord:
                        sample_ts_keys = set(ts_to_timecoord.keys())

                        time_values = []
                        price_values = []
                        volume_values = []

                        for row in target_rows_all:
                            if len(row) < 6:
                                continue

                            ts_raw = row[0]
                            try:
                                ts_norm = normalize_timestamp_array([ts_raw])[0]
                            except Exception:  # noqa: BLE001
                                continue

                            if ts_norm not in sample_ts_keys:
                                continue

                            t_coord = ts_to_timecoord.get(ts_norm)
                            if t_coord is None:
                                continue

                            try:
                                bid_price_row = float(row[1])
                                bid_qty_row = float(row[2])
                                ask_price_row = float(row[3])
                                ask_qty_row = float(row[4])
                            except (TypeError, ValueError):
                                continue

                            if bid_price_row > 0.0 and bid_qty_row > 0.0:
                                time_values.append(t_coord)
                                price_values.append(bid_price_row)
                                volume_values.append(bid_qty_row)

                            if ask_price_row > 0.0 and ask_qty_row > 0.0:
                                time_values.append(t_coord)
                                price_values.append(ask_price_row)
                                volume_values.append(ask_qty_row)

                        if time_values:
                            t_arr = np.asarray(time_values, dtype="float64")
                            p_arr = np.asarray(price_values, dtype="float64")
                            v_arr = np.asarray(volume_values, dtype="float64")

                            t_min, t_max = float(t_arr.min()), float(t_arr.max())
                            p_min, p_max = float(p_arr.min()), float(p_arr.max())

                            if t_max > t_min and p_max > p_min:
                                t_edges = np.linspace(t_min, t_max, num_time_bins + 1)
                                p_edges = np.linspace(p_min, p_max, num_price_bins + 1)

                                vol_grid, t_edges, p_edges = np.histogram2d(
                                    t_arr,
                                    p_arr,
                                    bins=[t_edges, p_edges],
                                    weights=v_arr,
                                )

                                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                                mesh = ax.pcolormesh(
                                    t_edges,
                                    p_edges,
                                    vol_grid.T,
                                    shading="auto",
                                )
                                fig.colorbar(mesh, ax=ax, label="Volume")

                                # Overlay mid-price line if available (no legend label needed).
                                mid_valid_mask = np.isfinite(mid_arr_all)
                                if mid_valid_mask.any():
                                    ax.plot(
                                        time_axis[mid_valid_mask],
                                        mid_arr_all[mid_valid_mask],
                                        color="white",
                                        linewidth=1.0,
                                    )
                                ax.set_xlabel("Seconds since first sampled snapshot")
                                ax.set_ylabel("Price")
                                ax.set_title(f"Depth volume heatmap for {target_asset}")

                                # Minor ticks on both axes.
                                ax.xaxis.set_minor_locator(AutoMinorLocator())
                                ax.yaxis.set_minor_locator(AutoMinorLocator())

                                # Secondary datetime x-axis at the top.
                                try:
                                    top_ax = ax.twiny()
                                    top_ax.set_xlim(ax.get_xlim())

                                    major_ticks = ax.get_xticks()
                                    datetime_labels = []
                                    for t_val in major_ticks:
                                        dt_val = base_time + np.timedelta64(int(t_val), "s")
                                        datetime_labels.append(str(dt_val)[:16])  # YYYY-MM-DDTHH:MM

                                    top_ax.set_xticks(major_ticks)
                                    top_ax.set_xticklabels(datetime_labels)

                                    for label in top_ax.get_xticklabels():
                                        label.set_rotation(45)
                                        label.set_fontsize(6)

                                    top_ax.xaxis.set_minor_locator(AutoMinorLocator())
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning(
                                        "Failed to construct secondary datetime x-axis for depth heatmap: %s",
                                        exc,
                                    )

                                depth_heatmap_path = tmp_dir / f"depth_volume_heatmap_{asset_suffix}.png"
                                fig.tight_layout()
                                fig.savefig(depth_heatmap_path)

                                # Optional label-check overlay on top of the depth heatmap.
                                if label_checks_enabled and labels_array is not None:
                                    try:
                                        ts_full = np.asarray(timestamps_array, dtype="datetime64[ns]")
                                        n_snapshots_full = min(len(snapshot_features), ts_full.shape[0])
                                        n_labels_full = int(labels_array.shape[0])

                                        horizon_steps_overlay = prediction_horizon_seconds // cadence_seconds
                                        visible_steps_overlay = visible_window_seconds // cadence_seconds

                                        if horizon_steps_overlay > 0 and visible_steps_overlay > 0:
                                            mid_full_overlay: list[float] = []
                                            for features in snapshot_features[:n_snapshots_full]:
                                                if not isinstance(features, (list, tuple)) or len(features) < 4:
                                                    mid_full_overlay.append(float("nan"))
                                                    continue

                                                bid_p, _, ask_p, _ = features[:4]
                                                try:
                                                    bid_p_f = float(bid_p)
                                                    ask_p_f = float(ask_p)
                                                except (TypeError, ValueError):
                                                    mid_full_overlay.append(float("nan"))
                                                    continue

                                                if bid_p_f <= 0.0 or ask_p_f <= 0.0:
                                                    mid_full_overlay.append(float("nan"))
                                                    continue

                                                mid_full_overlay.append(0.5 * (bid_p_f + ask_p_f))

                                            mid_full_overlay_arr = np.asarray(mid_full_overlay, dtype="float64")

                                            train_indices_arr = np.asarray(list(train_idx), dtype="int64")
                                            valid_anchors_overlay: list[int] = []
                                            for idx_anchor in train_indices_arr:
                                                if idx_anchor < 0 or idx_anchor >= n_labels_full:
                                                    continue

                                                start_idx_overlay = idx_anchor - visible_steps_overlay
                                                end_idx_overlay = idx_anchor + horizon_steps_overlay

                                                if start_idx_overlay < 0 or end_idx_overlay >= n_snapshots_full:
                                                    continue

                                                if not math.isfinite(mid_full_overlay_arr[idx_anchor]):
                                                    continue

                                                valid_anchors_overlay.append(int(idx_anchor))

                                            if valid_anchors_overlay:
                                                rng_overlay = np.random.default_rng(random_seed)
                                                idx_anchor_overlay = int(
                                                    rng_overlay.choice(
                                                        np.asarray(valid_anchors_overlay, dtype="int64"),
                                                    ),
                                                )

                                                mid_anchor_overlay = float(
                                                    mid_full_overlay_arr[idx_anchor_overlay],
                                                )
                                                if math.isfinite(mid_anchor_overlay) and mid_anchor_overlay > 0.0:
                                                    ts_anchor_overlay = ts_full[idx_anchor_overlay]
                                                    t_anchor_coord = (
                                                        ts_anchor_overlay - base_time
                                                    ) / np.timedelta64(1, "s")
                                                    t_anchor_coord = float(t_anchor_coord)

                                                    x_start = t_anchor_coord
                                                    x_end = t_anchor_coord + float(prediction_horizon_seconds)

                                                    y_min_hm, y_max_hm = ax.get_ylim()

                                                    box_overlay = mpatches.Rectangle(
                                                        (x_start, y_min_hm),
                                                        x_end - x_start,
                                                        y_max_hm - y_min_hm,
                                                        linewidth=1.0,
                                                        edgecolor="red",
                                                        facecolor="none",
                                                    )
                                                    ax.add_patch(box_overlay)

                                                    price_levels_overlay = [
                                                        mid_anchor_overlay * (1.0 + b / 100.0)
                                                        for b in price_boundaries_pct
                                                    ]
                                                    for level in price_levels_overlay:
                                                        if y_min_hm <= level <= y_max_hm:
                                                            ax.hlines(
                                                                level,
                                                                x_start,
                                                                x_end,
                                                                colors="red",
                                                                linestyles="dashed",
                                                                linewidth=0.8,
                                                            )

                                                    label_depth_heatmap_path = (
                                                        tmp_dir
                                                        / f"label_checks_depth_volume_heatmap_{asset_suffix}.png"
                                                    )
                                                    fig.tight_layout()
                                                    fig.savefig(label_depth_heatmap_path)

                                                    mlflow.log_artifact(
                                                        str(label_depth_heatmap_path),
                                                        artifact_path="diagnostics",
                                                    )
                                                    logger.info(
                                                        "Logged label-check depth volume heatmap artifact to MLFlow at %s",
                                                        label_depth_heatmap_path,
                                                    )
                                    except Exception as exc:  # noqa: BLE001
                                        logger.warning(
                                            "Failed to generate/log label-check depth volume heatmap: %s",
                                            exc,
                                        )

                                plt.close(fig)

                                mlflow.log_artifact(str(depth_heatmap_path), artifact_path="diagnostics")
                                logger.info(
                                    "Logged depth volume heatmap artifact to MLFlow at %s",
                                    depth_heatmap_path,
                                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate/log depth volume heatmap: %s", exc)

        # Label-check visualization: for a few randomly selected anchor times
        # in the training set, show past/future windows and class boxes based
        # on price_classes around the anchor mid-price. This visualization is
        # independent from the spread diagnostics and focuses on verifying
        # that labels are consistent with future price paths.
        if label_checks_enabled and timestamps_array is not None and labels_array is not None:
            try:
                # Use the already-normalized timestamps_array (datetime64[ns])
                # to avoid unit inconsistencies when computing relative times.
                ts_full = np.asarray(timestamps_array, dtype="datetime64[ns]")
                if ts_full.size == 0:
                    raise ValueError("No snapshot_timestamps available for label checks")

                # Ensure we stay within bounds of both snapshot_features and ts_full.
                n_snapshots = min(len(snapshot_features), ts_full.shape[0])
                n_labels = int(labels_array.shape[0])

                # Horizon and visible window in snapshot index steps.
                horizon_steps = prediction_horizon_seconds // cadence_seconds
                visible_steps = visible_window_seconds // cadence_seconds

                if horizon_steps <= 0 or visible_steps <= 0:
                    logger.warning(
                        "Label checks skipped: horizon_steps=%s, visible_steps=%s (must both be positive)",
                        horizon_steps,
                        visible_steps,
                    )
                else:
                    # Mid-price for all snapshots (aligned with snapshot_timestamps).
                    mid_full = []
                    for idx_snap, features in enumerate(snapshot_features):
                        if not isinstance(features, (list, tuple)) or len(features) < 4:
                            mid_full.append(float("nan"))
                            continue

                        bid_p, _, ask_p, _ = features[:4]
                        try:
                            bid_p_f = float(bid_p)
                            ask_p_f = float(ask_p)
                        except (TypeError, ValueError):
                            mid_full.append(float("nan"))
                            continue

                        if bid_p_f <= 0.0 or ask_p_f <= 0.0:
                            mid_full.append(float("nan"))
                            continue

                        mid_full.append(0.5 * (bid_p_f + ask_p_f))

                    mid_full_arr = np.asarray(mid_full, dtype="float64")

                    # Candidate anchors: training anchors where both full
                    # past and future windows fall inside the available
                    # snapshots.
                    train_indices_arr = np.asarray(list(train_idx), dtype="int64")
                    valid_anchors = []
                    for idx_anchor in train_indices_arr:
                        if idx_anchor < 0 or idx_anchor >= n_labels:
                            continue

                        start_idx = idx_anchor - visible_steps
                        end_idx = idx_anchor + horizon_steps

                        if start_idx < 0 or end_idx >= n_snapshots:
                            continue

                        if not math.isfinite(mid_full_arr[idx_anchor]):
                            continue

                        valid_anchors.append(int(idx_anchor))

                    if not valid_anchors:
                        logger.info("Label checks skipped: no valid anchor indices found for label visualization.")
                    else:
                        rng_label = np.random.default_rng(random_seed)

                        num_examples = min(label_checks_num_examples, len(valid_anchors))
                        selected_indices = rng_label.choice(
                            np.asarray(valid_anchors, dtype="int64"),
                            size=num_examples,
                            replace=False,
                        )

                        # Decide whether to group into a single figure or
                        # create one PNG per anchor.
                        if num_examples <= label_checks_max_per_figure:
                            # Single figure with one axis per example.
                            fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4 * num_examples), sharex=True)
                            if num_examples == 1:
                                axes = [axes]

                            for ax, idx_anchor in zip(axes, selected_indices):
                                _plot_label_check_for_anchor(
                                    ax=ax,
                                    idx_anchor=int(idx_anchor),
                                    ts_full=ts_full,
                                    mid_full_arr=mid_full_arr,
                                    visible_window_seconds=visible_window_seconds,
                                    prediction_horizon_seconds=prediction_horizon_seconds,
                                    cadence_seconds=cadence_seconds,
                                    price_boundaries_pct=price_boundaries_pct,
                                )

                            axes[-1].set_xlabel("Seconds relative to anchor t0 (0 = anchor)")

                            label_checks_path = tmp_dir / f"label_checks_{asset_suffix}.png"
                            fig.tight_layout()
                            fig.savefig(label_checks_path)
                            plt.close(fig)

                            mlflow.log_artifact(str(label_checks_path), artifact_path="diagnostics")
                            logger.info(
                                "Logged label-check diagnostics figure to MLFlow at %s",
                                label_checks_path,
                            )
                        else:
                            # One figure per anchor to avoid overcrowding.
                            for idx_anchor in selected_indices:
                                fig, ax = plt.subplots(1, 1, figsize=(12, 4))

                                _plot_label_check_for_anchor(
                                    ax=ax,
                                    idx_anchor=int(idx_anchor),
                                    ts_full=ts_full,
                                    mid_full_arr=mid_full_arr,
                                    visible_window_seconds=visible_window_seconds,
                                    prediction_horizon_seconds=prediction_horizon_seconds,
                                    cadence_seconds=cadence_seconds,
                                    price_boundaries_pct=price_boundaries_pct,
                                )

                                ax.set_xlabel("Seconds relative to anchor t0 (0 = anchor)")

                                label_checks_path = tmp_dir / f"label_check_t0_{int(idx_anchor)}_{asset_suffix}.png"
                                fig.tight_layout()
                                fig.savefig(label_checks_path)
                                plt.close(fig)

                                mlflow.log_artifact(str(label_checks_path), artifact_path="diagnostics")
                                logger.info(
                                    "Logged label-check diagnostics figure for anchor %s to MLFlow at %s",
                                    int(idx_anchor),
                                    label_checks_path,
                                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate/log label-check diagnostics: %s", exc)


def _plot_label_check_for_anchor(
    *,
    ax: Any,
    idx_anchor: int,
    ts_full: np.ndarray,
    mid_full_arr: np.ndarray,
    visible_window_seconds: int,
    prediction_horizon_seconds: int,
    cadence_seconds: int,
    price_boundaries_pct: Sequence[float],
) -> None:
    """Plot label-check visualization for a single anchor index.

    The x-axis is time relative to the anchor t0 (in seconds). The past
    window [t0 - visible_window_seconds, t0] and the future window
    [t0, t0 + prediction_horizon_seconds] are shaded differently, and the
    price_classes boundaries around the anchor mid-price are drawn as
    subdivisions within the future window.
    """

    n_snapshots = mid_full_arr.shape[0]

    horizon_steps = prediction_horizon_seconds // cadence_seconds
    visible_steps = visible_window_seconds // cadence_seconds

    if horizon_steps <= 0 or visible_steps <= 0:
        return

    start_idx = idx_anchor - visible_steps
    end_idx = idx_anchor + horizon_steps

    if start_idx < 0 or end_idx >= n_snapshots:
        return

    mid_anchor = float(mid_full_arr[idx_anchor])
    if not math.isfinite(mid_anchor) or mid_anchor <= 0.0:
        return

    ts_anchor = ts_full[idx_anchor]
    ts_window = ts_full[start_idx : end_idx + 1]
    mid_window = mid_full_arr[start_idx : end_idx + 1]

    # Relative time axis in seconds: anchor t0 at 0.
    t_offsets = (ts_window - ts_anchor) / np.timedelta64(1, "s")
    t_offsets = t_offsets.astype("float64")

    # Shading for past and future windows.
    ax.axvspan(-float(visible_window_seconds), 0.0, facecolor="lightblue", alpha=0.2)
    ax.axvspan(0.0, float(prediction_horizon_seconds), facecolor="lightgray", alpha=0.2)

    # Mid-price trajectory over the window and autoscaled y-limits.
    ax.plot(t_offsets, mid_window, color="blue", linewidth=1.0, label="mid_price")
    ax.relim()
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()

    # Future window rectangle constrained to the autoscaled y-range.
    box = mpatches.Rectangle(
        (0.0, y_min),
        float(prediction_horizon_seconds),
        y_max - y_min,
        linewidth=1.0,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(box)

    # Horizontal subdivision lines only for price boundaries inside the
    # visible y-range.
    price_levels = [mid_anchor * (1.0 + b / 100.0) for b in price_boundaries_pct]
    for level in price_levels:
        if y_min <= level <= y_max:
            ax.hlines(
                level,
                0.0,
                float(prediction_horizon_seconds),
                colors="red",
                linestyles="dashed",
                linewidth=0.8,
            )

    # Anchor marker at t=0.
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)

    ax.set_ylabel("Mid price")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    pct_min = (y_min / mid_anchor - 1.0) * 100.0
    pct_max = (y_max / mid_anchor - 1.0) * 100.0
    sec_ax = ax.twinx()
    sec_ax.set_ylim(pct_min, pct_max)
    sec_ax.set_ylabel("Δ price % vs anchor")
    sec_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
    sec_ax.yaxis.set_minor_locator(AutoMinorLocator())

    try:
        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())

        major_ticks = ax.get_xticks()
        datetime_labels = []
        for t_val in major_ticks:
            dt_val = ts_anchor + np.timedelta64(int(t_val), "s")
            datetime_labels.append(str(dt_val)[:16])

        top_ax.set_xticks(major_ticks)
        top_ax.set_xticklabels(datetime_labels)

        for label in top_ax.get_xticklabels():
            label.set_rotation(45)
            label.set_fontsize(6)

        top_ax.xaxis.set_minor_locator(AutoMinorLocator())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to construct secondary datetime x-axis for label-check plot: %s",
            exc,
        )

    ax.set_title(f"Label check around anchor t0={str(ts_anchor)[:19]}")
