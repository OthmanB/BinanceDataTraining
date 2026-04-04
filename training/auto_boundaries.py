"""Automatic fitting of price-class boundaries.

This module fits targets.price_classes.boundaries from a disk-cached series-only
dataset (timestamps/mid_prices/volumes).

The series cache hash intentionally excludes targets.price_classes.boundaries so
that auto boundary fitting can run before snapshot hashing/building.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging
import os

import numpy as np

from training.series_dataset import prepare_series_dataset
from utils.config_loader import ConfigError


logger = logging.getLogger(__name__)


def _iter_series_files(series_dir: str) -> List[str]:
    chunks_dir = os.path.join(series_dir, "chunks")
    if not os.path.isdir(chunks_dir):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(chunks_dir)):
        if not name.endswith(".npz"):
            continue
        out.append(os.path.join(chunks_dir, name))
    return out


def _resolve_fit_indices(config: Dict[str, Any], n_samples: int, fit_on: str) -> Tuple[int, int]:
    if fit_on == "full":
        return 0, int(n_samples)

    # fit_on == "train" uses preprocessing.train_test_split ratios.
    split_cfg = config["preprocessing"]["train_test_split"]
    method = str(split_cfg.get("method") or "")
    if method != "chronological":
        raise ConfigError("Auto boundaries require preprocessing.train_test_split.method='chronological'")
    train_ratio = float(split_cfg["train_ratio"])
    if not (0.0 < train_ratio <= 1.0):
        raise ConfigError("preprocessing.train_test_split.train_ratio must be in (0, 1]")

    train_end = int(np.floor(float(n_samples) * float(train_ratio)))
    train_end = max(0, min(int(n_samples), train_end))
    return 0, train_end


def _compute_intensity_magnitudes_from_mid_prices(
    mid_prices: np.ndarray,
    *,
    window_steps: int,
    horizon_steps: int,
    max_samples: int,
    random_seed: int,
    labeling_criteria: str,
) -> np.ndarray:
    mid_prices = np.asarray(mid_prices, dtype="float64")
    if mid_prices.ndim != 1:
        raise ValueError("mid_prices must be rank 1")

    n = int(mid_prices.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype="float64")

    min_anchor = window_steps - 1
    max_anchor = n - horizon_steps - 1
    if max_anchor < min_anchor:
        return np.zeros((0,), dtype="float64")

    anchors_all = np.arange(min_anchor, max_anchor + 1, dtype="int64")
    if anchors_all.size == 0:
        return np.zeros((0,), dtype="float64")

    if int(max_samples) < int(anchors_all.size):
        rng = np.random.default_rng(int(random_seed))
        anchors = rng.choice(anchors_all, size=int(max_samples), replace=False)
        anchors = np.sort(anchors.astype("int64"))
    else:
        anchors = anchors_all

    mags = np.zeros((int(anchors.shape[0]),), dtype="float64")
    for i, anchor in enumerate(anchors.tolist()):
        p0 = float(mid_prices[int(anchor)])
        if not (p0 > 0.0):
            mags[i] = 0.0
            continue
        future = mid_prices[int(anchor) + 1 : int(anchor) + int(horizon_steps) + 1]
        if future.size == 0:
            mags[i] = 0.0
            continue
        rel_moves = (future - p0) / p0 * 100.0
        max_up = float(np.max(rel_moves))
        max_down = float(np.min(rel_moves))
        up_intensity = max(max_up, 0.0)
        down_intensity = max(-max_down, 0.0)

        if labeling_criteria == "up_intensity":
            mags[i] = up_intensity
        elif labeling_criteria == "down_intensity":
            mags[i] = down_intensity
        else:
            mags[i] = max(up_intensity, down_intensity)

    mags = mags[np.isfinite(mags)]
    mags = mags[mags > 0.0]
    return np.asarray(mags, dtype="float64")


def _fit_quantile_boundaries(magnitudes: np.ndarray, *, num_classes: int) -> List[float]:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    k = int(num_classes) - 1
    magnitudes = np.asarray(magnitudes, dtype="float64")
    if magnitudes.size == 0:
        raise ConfigError("Auto boundary fitting found no valid intensity magnitudes")

    qs = np.linspace(1.0 / float(num_classes), float(k) / float(num_classes), num=k)
    bounds = np.quantile(magnitudes, qs, method="linear")  # numpy>=1.22
    bounds = np.asarray(bounds, dtype="float64")

    # Enforce strictly increasing, >0 by deduping with a tiny epsilon.
    eps = 1e-9
    out: List[float] = []
    last = 0.0
    for v in bounds.tolist():
        fv = float(v)
        if not np.isfinite(fv):
            continue
        fv = max(fv, eps)
        if out and fv <= last:
            fv = last + eps
        out.append(fv)
        last = fv

    if len(out) != k:
        raise ConfigError(
            "Auto boundary fitting produced invalid boundary count: "
            f"got={len(out)}, expected={k}"
        )
    return out


def fit_price_class_boundaries_from_series_cache(
    config: Dict[str, Any],
    *,
    num_classes: int,
    fit_on: str,
    labeling_criteria: str,
    max_samples: int,
    random_seed: int,
) -> List[float]:
    """Fit boundaries by sampling anchors and computing horizon intensities."""

    if fit_on not in {"train", "full"}:
        raise ConfigError("fit_on must be 'train' or 'full'")
    if labeling_criteria not in {"max_intensity", "up_intensity", "down_intensity"}:
        raise ConfigError("labeling_criteria must be one of: max_intensity, up_intensity, down_intensity")
    if max_samples <= 0:
        raise ConfigError("max_samples must be positive")
    if random_seed < 0:
        raise ConfigError("random_seed must be >= 0")

    targets_cfg = config["targets"]
    visible_window_seconds = int(targets_cfg["visible_window_seconds"])
    prediction_horizon_seconds = int(targets_cfg["prediction_horizon_seconds"])
    cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
    if cadence_seconds <= 0:
        raise ConfigError("data.time_range.cadence_seconds must be positive")
    if visible_window_seconds <= 0 or prediction_horizon_seconds <= 0:
        raise ConfigError("targets.visible_window_seconds and prediction_horizon_seconds must be positive")
    if visible_window_seconds % cadence_seconds != 0:
        raise ConfigError("targets.visible_window_seconds must be a multiple of data.time_range.cadence_seconds")
    window_steps = int(visible_window_seconds // cadence_seconds)
    horizon_steps = int(prediction_horizon_seconds // cadence_seconds)
    if horizon_steps <= 0:
        raise ConfigError("targets.prediction_horizon_seconds must be >= data.time_range.cadence_seconds")

    # Ensure series cache exists so chunk files are present.
    dataset = prepare_series_dataset(config)
    context_dir = str(getattr(dataset, "series_dir", ""))
    if not context_dir:
        raise ConfigError("Series dataset missing series_dir")

    series_files = _iter_series_files(context_dir)
    if not series_files:
        raise ConfigError(
            "Series cache files missing; expected files under series_cache/<name>/chunks. "
            "Rebuild series cache."
        )

    # Aggregate mid_prices from series files (target asset only).
    mid_parts: List[np.ndarray] = []
    total_snapshots = 0
    for path in series_files:
        try:
            with np.load(path) as npz:
                mids = np.asarray(npz["mid_prices"], dtype="float64")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read series file %s: %s", path, exc)
            continue
        if mids.size == 0:
            continue
        mid_parts.append(mids)
        total_snapshots += int(mids.shape[0])

    if total_snapshots <= 0 or not mid_parts:
        raise ConfigError("Auto boundary fitting found no mid_prices in snapshot series")

    mid_prices = np.concatenate(mid_parts, axis=0)
    fit_start, fit_end = _resolve_fit_indices(config, int(mid_prices.shape[0]), fit_on)
    mid_fit = np.asarray(mid_prices[int(fit_start) : int(fit_end)], dtype="float64")

    magnitudes = _compute_intensity_magnitudes_from_mid_prices(
        mid_fit,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        max_samples=int(max_samples),
        random_seed=int(random_seed),
        labeling_criteria=labeling_criteria,
    )

    boundaries = _fit_quantile_boundaries(magnitudes, num_classes=int(num_classes))
    logger.info(
        "Fitted auto boundaries from series cache: num_classes=%s fit_on=%s labeling_criteria=%s samples=%s boundaries=%s",
        int(num_classes),
        fit_on,
        labeling_criteria,
        int(magnitudes.shape[0]),
        [round(float(b), 6) for b in boundaries],
    )
    return boundaries


__all__ = [
    "fit_price_class_boundaries_from_series_cache",
]
