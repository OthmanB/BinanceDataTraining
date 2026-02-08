"""Snapshot-compatible diagnostics for data quality and monitoring."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
from pathlib import Path
import tempfile

import numpy as np

from training.long_term_context import load_anchor_timestamps, load_snapshot_series
from training.snapshot_dataset import iter_snapshot_batches, prepare_snapshot_dataset


logger = logging.getLogger(__name__)


def _compute_split_boundaries(n_samples: int, train_ratio: float, validation_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("preprocessing.train_test_split ratios must sum to 1.0")
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * validation_ratio)
    return train_end, min(val_end, n_samples), n_samples


def _sample_indices(base_indices: np.ndarray, method: str, num_samples: int, seed: int) -> np.ndarray:
    if base_indices.size == 0:
        return np.array([], dtype="int64")
    if num_samples >= base_indices.size:
        return base_indices
    if method == "uniform":
        positions = np.linspace(0, base_indices.size - 1, num_samples, dtype="int64")
        return base_indices[positions]
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(base_indices, size=num_samples, replace=False).astype("int64"))
    raise ValueError(f"Unsupported diagnostics.sampling.method: {method!r}")


def _coerce_scalar(value: Any) -> Optional[float]:
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.shape == ():
        return float(arr)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _extract_top_of_book(sample: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if sample.ndim == 4:
        last_step = sample[-1]
        if last_step.shape[2] == 4:
            bid_price = _coerce_scalar(last_step[0, 0, 0])
            bid_qty = _coerce_scalar(last_step[0, 1, 0])
            ask_price = _coerce_scalar(last_step[0, 2, 0])
            ask_qty = _coerce_scalar(last_step[0, 3, 0])
        else:
            bid_price = _coerce_scalar(last_step[0, 0, 0])
            bid_qty = _coerce_scalar(last_step[0, 1, 0])
            ask_price = _coerce_scalar(last_step[1, 0, 0])
            ask_qty = _coerce_scalar(last_step[1, 1, 0])
    elif sample.ndim == 3:
        if sample.shape[1] == 4:
            bid_price = _coerce_scalar(sample[0, 0, 0])
            bid_qty = _coerce_scalar(sample[0, 1, 0])
            ask_price = _coerce_scalar(sample[0, 2, 0])
            ask_qty = _coerce_scalar(sample[0, 3, 0])
        else:
            bid_price = _coerce_scalar(sample[0, 0, 0])
            bid_qty = _coerce_scalar(sample[0, 1, 0])
            ask_price = _coerce_scalar(sample[1, 0, 0])
            ask_qty = _coerce_scalar(sample[1, 1, 0])
    elif sample.ndim == 2:
        bid_price = _coerce_scalar(sample[0, 0])
        bid_qty = _coerce_scalar(sample[0, 1])
        ask_price = _coerce_scalar(sample[0, 2])
        ask_qty = _coerce_scalar(sample[0, 3])
    else:
        return None

    if None in {bid_price, bid_qty, ask_price, ask_qty}:
        return None
    return float(bid_price), float(bid_qty), float(ask_price), float(ask_qty)


def run_snapshot_diagnostics(config: Dict[str, Any]) -> None:
    """Run diagnostics directly from snapshot datasets.

    This keeps diagnostics useful in snapshot-only mode, where legacy in-memory
    diagnostics are not applicable.
    """

    diagnostics_cfg = config["diagnostics"]
    if not bool(diagnostics_cfg["enabled"]):
        logger.info("Snapshot diagnostics disabled via configuration; skipping.")
        return

    snapshot_dataset = prepare_snapshot_dataset(config)
    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Snapshot diagnostics skipped: snapshot dataset has no samples.")
        return

    split_cfg = config["preprocessing"]["train_test_split"]
    train_end, _, _ = _compute_split_boundaries(
        n_samples,
        float(split_cfg["train_ratio"]),
        float(split_cfg["validation_ratio"]),
        float(split_cfg["test_ratio"]),
    )
    if train_end <= 0:
        logger.info("Snapshot diagnostics skipped: no training samples available.")
        return

    sampling_cfg = diagnostics_cfg["sampling"]
    sampled_indices = _sample_indices(
        np.arange(train_end, dtype="int64"),
        str(sampling_cfg["method"]),
        int(sampling_cfg["num_samples"]),
        int(sampling_cfg["random_seed"]),
    )

    duty_cycle_sampled: List[float] = []
    label_up_counts: Dict[int, int] = {}
    label_down_counts: Dict[int, int] = {}
    spread_pct_sampled: List[float] = []
    mid_price_sampled: List[float] = []
    bid_price_sampled: List[float] = []
    ask_price_sampled: List[float] = []
    bid_qty_sampled: List[float] = []
    ask_qty_sampled: List[float] = []

    sampled_set = set(int(i) for i in sampled_indices.tolist())
    warned_bad_feature = False
    cursor = 0
    for x_chunk, y_up_chunk, y_down_chunk, _, duty_cycle_chunk in iter_snapshot_batches(snapshot_dataset, 0, train_end):
        chunk_len = int(y_up_chunk.shape[0])
        for i in range(chunk_len):
            y_up = int(y_up_chunk[i])
            y_down = int(y_down_chunk[i])
            label_up_counts[y_up] = int(label_up_counts.get(y_up, 0)) + 1
            label_down_counts[y_down] = int(label_down_counts.get(y_down, 0)) + 1
            global_idx = cursor + i
            if global_idx in sampled_set:
                duty_cycle_sampled.append(float(duty_cycle_chunk[i]))
                try:
                    features = np.asarray(x_chunk[i])
                    extracted = _extract_top_of_book(features)
                except Exception as exc:  # noqa: BLE001
                    extracted = None
                    if not warned_bad_feature:
                        logger.warning("Snapshot diagnostics failed to parse sample features: %s", exc)
                        warned_bad_feature = True

                if extracted is not None:
                    bid_price, bid_qty, ask_price, ask_qty = extracted
                    bid_price_sampled.append(bid_price)
                    ask_price_sampled.append(ask_price)
                    bid_qty_sampled.append(bid_qty)
                    ask_qty_sampled.append(ask_qty)
                    if bid_price > 0.0 and ask_price > 0.0:
                        mid_price = 0.5 * (bid_price + ask_price)
                        mid_price_sampled.append(mid_price)
                        if mid_price > 0.0:
                            spread_pct_sampled.append((ask_price - bid_price) / mid_price * 100.0)
        cursor += chunk_len

    if duty_cycle_sampled:
        duty_arr = np.asarray(duty_cycle_sampled, dtype="float64")
        logger.info(
            "Snapshot diagnostics duty-cycle stats (sampled): min=%.6f max=%.6f mean=%.6f std=%.6f",
            float(np.min(duty_arr)),
            float(np.max(duty_arr)),
            float(np.mean(duty_arr)),
            float(np.std(duty_arr)),
        )

    if spread_pct_sampled:
        spread_arr = np.asarray(spread_pct_sampled, dtype="float64")
        logger.info(
            "Snapshot diagnostics spread stats (sampled): mean=%.6f std=%.6f max=%.6f",
            float(np.mean(spread_arr)),
            float(np.std(spread_arr)),
            float(np.max(spread_arr)),
        )

    anchor_ts = load_anchor_timestamps(snapshot_dataset)[:train_end]
    if anchor_ts.shape[0] >= 2:
        cadence = int(config["data"]["time_range"]["cadence_seconds"])
        gap_cfg = diagnostics_cfg["gap_checks"]
        large_gap = float(gap_cfg["large_gap_multiplier"]) * float(cadence)
        very_large_gap = float(gap_cfg["very_large_gap_multiplier"]) * float(cadence)
        delta_seconds = np.diff(anchor_ts.astype("int64"))
        large_count = int(np.sum(delta_seconds > large_gap))
        very_large_count = int(np.sum(delta_seconds > very_large_gap))
        logger.info(
            "Snapshot diagnostics gap stats (train): large_gaps=%s very_large_gaps=%s max_gap_seconds=%s",
            large_count,
            very_large_count,
            int(np.max(delta_seconds)) if delta_seconds.size > 0 else 0,
        )

    viz_cfg = diagnostics_cfg["visualization"]
    if not bool(viz_cfg["enabled"]):
        return

    time_series_enabled = bool(viz_cfg["time_series"])
    histograms_enabled = bool(viz_cfg["histograms"])
    histogram_bins = int(viz_cfg["histogram_bins"])
    heatmaps_cfg = viz_cfg["heatmaps"]
    heatmaps_enabled = bool(heatmaps_cfg["enabled"])
    heatmap_types = [str(t) for t in heatmaps_cfg.get("types", [])]
    num_time_bins = int(heatmaps_cfg.get("num_time_bins", 20))
    num_spread_bins = int(heatmaps_cfg.get("num_spread_bins", 20))

    with tempfile.TemporaryDirectory(prefix="snapshot_diagnostics_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_paths: List[Path] = []

        if histograms_enabled and duty_cycle_sampled:
            import matplotlib  # type: ignore[import]

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]

            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            ax.hist(np.asarray(duty_cycle_sampled, dtype="float64"), bins=max(5, histogram_bins), color="#2a7f62")
            ax.set_title("Sampled Duty-Cycle Distribution (Train)")
            ax.set_xlabel("Duty Cycle")
            ax.set_ylabel("Count")
            fig.tight_layout()
            out = tmp_path / "duty_cycle_hist.png"
            fig.savefig(out)
            plt.close(fig)
            artifact_paths.append(out)

        if histograms_enabled and spread_pct_sampled:
            import matplotlib  # type: ignore[import]

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]

            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            ax.hist(np.asarray(spread_pct_sampled, dtype="float64"), bins=max(5, histogram_bins), color="#f59e0b")
            ax.set_title("Sampled Spread % Distribution (Train)")
            ax.set_xlabel("Spread %")
            ax.set_ylabel("Count")
            fig.tight_layout()
            out = tmp_path / "spread_pct_hist.png"
            fig.savefig(out)
            plt.close(fig)
            artifact_paths.append(out)

        if label_up_counts and label_down_counts:
            import matplotlib  # type: ignore[import]

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            up_classes = sorted(label_up_counts)
            down_classes = sorted(label_down_counts)
            axes[0].bar([str(i) for i in up_classes], [label_up_counts[i] for i in up_classes], color="#3b82f6")
            axes[0].set_title("Up-Head Label Distribution (Train)")
            axes[1].bar([str(i) for i in down_classes], [label_down_counts[i] for i in down_classes], color="#ef4444")
            axes[1].set_title("Down-Head Label Distribution (Train)")
            for ax in axes:
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
            fig.tight_layout()
            label_dist_path = tmp_path / "label_distribution.png"
            fig.savefig(label_dist_path)
            plt.close(fig)
            artifact_paths.append(label_dist_path)

        if time_series_enabled and sampled_indices.size > 0:
            series_ts, series_mid, _ = load_snapshot_series(snapshot_dataset)
            sampled_anchor_ts = anchor_ts[np.asarray(sampled_indices, dtype="int64")]
            match_idx = np.searchsorted(series_ts.astype("int64"), sampled_anchor_ts.astype("int64"))
            valid = match_idx < series_ts.shape[0]
            if np.any(valid):
                import matplotlib  # type: ignore[import]

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore[import]

                sampled_times = series_ts[match_idx[valid]]
                sampled_prices = series_mid[match_idx[valid]]
                fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
                ax.plot(sampled_times, sampled_prices, linewidth=1.0, color="#111827")
                ax.set_title("Sampled Mid-Price Trace (Train Anchors)")
                ax.set_xlabel("Timestamp (epoch seconds)")
                ax.set_ylabel("Mid Price")
                fig.tight_layout()
                ts_path = tmp_path / "sampled_mid_price_trace.png"
                fig.savefig(ts_path)
                plt.close(fig)
                artifact_paths.append(ts_path)

        if heatmaps_enabled and "spread_vs_time" in heatmap_types and spread_pct_sampled and sampled_indices.size > 0:
            series_ts, _, _ = load_snapshot_series(snapshot_dataset)
            sampled_anchor_ts = anchor_ts[np.asarray(sampled_indices, dtype="int64")]
            match_idx = np.searchsorted(series_ts.astype("int64"), sampled_anchor_ts.astype("int64"))
            valid = match_idx < series_ts.shape[0]
            if np.any(valid):
                import matplotlib  # type: ignore[import]

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore[import]

                t_vals = series_ts[match_idx[valid]].astype("float64")
                s_vals = np.asarray(spread_pct_sampled, dtype="float64")[: t_vals.shape[0]]
                if t_vals.size == s_vals.size and t_vals.size > 1:
                    t_edges = np.linspace(float(t_vals.min()), float(t_vals.max()), num_time_bins + 1)
                    s_edges = np.linspace(float(s_vals.min()), float(s_vals.max()), num_spread_bins + 1)
                    hist2d, xedges, yedges = np.histogram2d(
                        t_vals,
                        s_vals,
                        bins=[t_edges.tolist(), s_edges.tolist()],
                    )
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    mesh = ax.pcolormesh(xedges, yedges, hist2d.T, shading="auto")
                    fig.colorbar(mesh, ax=ax, label="Count")
                    ax.set_xlabel("Timestamp (epoch seconds)")
                    ax.set_ylabel("Spread %")
                    ax.set_title("Spread vs time heatmap (sampled)")
                    heatmap_path = tmp_path / "heatmap_spread_vs_time.png"
                    fig.tight_layout()
                    fig.savefig(heatmap_path)
                    plt.close(fig)
                    artifact_paths.append(heatmap_path)

        if not artifact_paths:
            return

        try:
            import mlflow  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Snapshot diagnostics skipped MLflow logging: %s", exc)
            return

        for artifact in artifact_paths:
            try:
                mlflow.log_artifact(str(artifact), artifact_path="snapshot_diagnostics")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log snapshot diagnostics artifact %s: %s", artifact, exc)


__all__ = ["run_snapshot_diagnostics"]
