"""Long-term feature computation for dual-channel architecture (TD-019).

This module implements multi-scale long-term context extraction for the
dual-channel CNN+LSTM model. It computes summary statistics over configurable
time windows (e.g., 7, 30, 90 days) to capture weekly, monthly, and quarterly
patterns in price dynamics.

The long-term features are designed to complement the short-term order book
snapshots used by the main model branch, providing broader market context
without excessive input dimensionality.

Features supported:
- mean_return: Average daily return over the window
- volatility: Standard deviation of daily returns
- volume_proxy: Average daily volume (from total order book depth)
- skewness: Skewness of the daily return distribution
- kurtosis: Excess kurtosis of daily returns
- max_up: Maximum single-day upward move
- max_down: Maximum single-day downward move (absolute value)

Summary methods:
- mean: Simple average of daily statistics
- ewma: Exponentially weighted moving average (more weight on recent days)
- last: Use only the most recent day's statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


# Supported long-term features
SUPPORTED_LT_FEATURES: FrozenSet[str] = frozenset({
    "mean_return",
    "volatility",
    "volume_proxy",
    "skewness",
    "kurtosis",
    "max_up",
    "max_down",
})

# Supported summary methods for aggregating daily stats over a window
SUPPORTED_SUMMARY_METHODS: FrozenSet[str] = frozenset({
    "mean",
    "ewma",
    "last",
})


class LongTermFeatureError(Exception):
    """Exception raised for long-term feature computation errors."""


@dataclass
class LongTermConfig:
    """Configuration for long-term feature extraction.

    Attributes
    ----------
    enabled:
        Whether long-term features are enabled.
    windows_days:
        List of window sizes in days (e.g., [7, 30, 90]).
    resolution_days:
        Aggregation resolution in days (typically 1).
    features:
        List of feature names to compute (must be in SUPPORTED_LT_FEATURES).
    summary_method:
        Method to summarize daily stats over a window ("mean", "ewma", "last").
    ewma_halflife_days:
        Half-life for EWMA weighting when summary_method is "ewma".
    """

    enabled: bool = False
    windows_days: List[int] = field(default_factory=lambda: [7, 30, 90])
    resolution_days: int = 1
    features: List[str] = field(
        default_factory=lambda: ["mean_return", "volatility", "volume_proxy", "skewness"]
    )
    summary_method: str = "mean"
    ewma_halflife_days: float = 7.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.enabled:
            return

        # Validate windows
        if not self.windows_days:
            raise LongTermFeatureError("windows_days must be a non-empty list")
        for w in self.windows_days:
            if w < 1:
                raise LongTermFeatureError(f"Window size must be >= 1, got {w}")

        # Validate resolution
        if self.resolution_days < 1:
            raise LongTermFeatureError(
                f"resolution_days must be >= 1, got {self.resolution_days}"
            )

        # Validate features
        for feat in self.features:
            if feat not in SUPPORTED_LT_FEATURES:
                raise LongTermFeatureError(
                    f"Unsupported long-term feature: {feat!r}. "
                    f"Supported: {sorted(SUPPORTED_LT_FEATURES)}"
                )

        # Validate summary method
        if self.summary_method not in SUPPORTED_SUMMARY_METHODS:
            raise LongTermFeatureError(
                f"Unsupported summary_method: {self.summary_method!r}. "
                f"Supported: {sorted(SUPPORTED_SUMMARY_METHODS)}"
            )

        # Validate EWMA half-life
        if self.summary_method == "ewma" and self.ewma_halflife_days <= 0:
            raise LongTermFeatureError(
                f"ewma_halflife_days must be > 0, got {self.ewma_halflife_days}"
            )

    @property
    def input_dim(self) -> int:
        """Compute the total input dimension for the long-term branch.

        Returns the number of windows times the number of features.
        """
        if not self.enabled:
            return 0
        return len(self.windows_days) * len(self.features)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LongTermConfig":
        """Create LongTermConfig from the model.long_term config section.

        Parameters
        ----------
        config:
            Full configuration dictionary. Will look for model.long_term.

        Returns
        -------
        LongTermConfig:
            Configured instance.
        """
        model_cfg = config["model"]
        lt_cfg = model_cfg["long_term"]

        try:
            enabled = bool(lt_cfg["enabled"])
            windows_days = list(lt_cfg["windows_days"])
            resolution_days = int(lt_cfg["resolution_days"])
            features = list(lt_cfg["features"])
            summary_method = str(lt_cfg["summary_method"])
            ewma_halflife_days = float(lt_cfg["ewma_halflife_days"])
        except KeyError as exc:
            raise LongTermFeatureError(
                f"Missing required long-term config key: model.long_term.{exc.args[0]}"
            ) from exc

        return cls(
            enabled=enabled,
            windows_days=windows_days,
            resolution_days=resolution_days,
            features=features,
            summary_method=summary_method,
            ewma_halflife_days=ewma_halflife_days,
        )


def compute_daily_aggregates(
    mid_prices: np.ndarray,
    timestamps: np.ndarray,
    volumes: Optional[np.ndarray],
    cadence_seconds: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate intraday data into daily statistics.

    Parameters
    ----------
    mid_prices:
        1D array of mid-prices for all snapshots.
    timestamps:
        1D array of Unix timestamps (seconds) for each snapshot.
    volumes:
        1D array of volume proxy values, or None if not available.
    cadence_seconds:
        Time interval between snapshots in seconds.

    Returns
    -------
    daily_timestamps:
        1D array of Unix timestamps for the start of each day.
    daily_returns:
        1D array of daily returns (close_i - close_{i-1}) / close_{i-1}.
        First day has return 0.
    daily_volumes:
        1D array of average daily volumes. If volumes is None, returns zeros.

    Notes
    -----
    - Days are defined as UTC calendar days (86400 seconds).
    - Each day's close price is the last mid_price within that day.
    - Handles missing days by not including them in output.
    """
    if len(mid_prices) == 0 or len(timestamps) == 0:
        return np.array([]), np.array([]), np.array([])

    if len(mid_prices) != len(timestamps):
        raise LongTermFeatureError(
            f"mid_prices and timestamps must have same length. "
            f"Got {len(mid_prices)} and {len(timestamps)}"
        )

    # Define day boundaries (86400 seconds per day)
    seconds_per_day = 86400
    day_indices = (timestamps // seconds_per_day).astype(np.int64)
    unique_days = np.unique(day_indices)

    n_days = len(unique_days)
    if n_days == 0:
        return np.array([]), np.array([]), np.array([])

    daily_timestamps = np.zeros(n_days, dtype=np.float64)
    daily_closes = np.zeros(n_days, dtype=np.float64)
    daily_volumes = np.zeros(n_days, dtype=np.float64)

    for i, day_idx in enumerate(unique_days):
        mask = day_indices == day_idx
        day_prices = mid_prices[mask]
        daily_timestamps[i] = float(day_idx * seconds_per_day)

        # Use last price of the day as close
        if len(day_prices) > 0:
            daily_closes[i] = day_prices[-1]

            # Average volume for the day
            if volumes is not None:
                day_volumes = volumes[mask]
                daily_volumes[i] = np.mean(day_volumes)

    # Compute daily returns
    daily_returns = np.zeros(n_days, dtype=np.float64)
    for i in range(1, n_days):
        if daily_closes[i - 1] > 0:
            daily_returns[i] = (daily_closes[i] - daily_closes[i - 1]) / daily_closes[i - 1]

    return daily_timestamps, daily_returns, daily_volumes


def _compute_window_stats(
    daily_returns: np.ndarray,
    daily_volumes: np.ndarray,
    features: List[str],
) -> np.ndarray:
    """Compute statistics for a single window of daily data.

    Parameters
    ----------
    daily_returns:
        1D array of daily returns for the window.
    daily_volumes:
        1D array of daily volumes for the window.
    features:
        List of feature names to compute.

    Returns
    -------
    np.ndarray:
        1D array of feature values, one per feature in order.
    """
    n_features = len(features)
    result = np.zeros(n_features, dtype=np.float64)

    n_days = len(daily_returns)

    for i, feat in enumerate(features):
        if n_days == 0:
            result[i] = 0.0
            continue

        if feat == "mean_return":
            result[i] = np.mean(daily_returns)
        elif feat == "volatility":
            result[i] = np.std(daily_returns, ddof=1) if n_days > 1 else 0.0
        elif feat == "volume_proxy":
            result[i] = np.mean(daily_volumes) if len(daily_volumes) > 0 else 0.0
        elif feat == "skewness":
            result[i] = _compute_skewness(daily_returns)
        elif feat == "kurtosis":
            result[i] = _compute_kurtosis(daily_returns)
        elif feat == "max_up":
            result[i] = np.max(daily_returns) if n_days > 0 else 0.0
        elif feat == "max_down":
            result[i] = abs(np.min(daily_returns)) if n_days > 0 else 0.0

    return result


def _compute_skewness(values: np.ndarray) -> float:
    """Compute Fisher-Pearson skewness coefficient.

    Returns 0 if there are fewer than 3 values or std is zero.
    """
    n = len(values)
    if n < 3:
        return 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return 0.0

    # Fisher-Pearson coefficient of skewness
    m3 = np.mean((values - mean) ** 3)
    return m3 / (std ** 3)


def _compute_kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher's definition).

    Returns 0 if there are fewer than 4 values or std is zero.
    """
    n = len(values)
    if n < 4:
        return 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        return 0.0

    # Excess kurtosis (subtract 3 for normal distribution)
    m4 = np.mean((values - mean) ** 4)
    return (m4 / (std ** 4)) - 3.0


def _apply_summary_method(
    daily_stats: np.ndarray,
    summary_method: str,
    ewma_halflife_days: float = 7.0,
) -> np.ndarray:
    """Apply summary method to reduce window stats to a single vector.

    Parameters
    ----------
    daily_stats:
        2D array of shape (n_days, n_features).
    summary_method:
        One of "mean", "ewma", "last".
    ewma_halflife_days:
        Half-life for EWMA weighting.

    Returns
    -------
    np.ndarray:
        1D array of shape (n_features,).
    """
    if len(daily_stats) == 0:
        return np.zeros(daily_stats.shape[1] if daily_stats.ndim == 2 else 0)

    if summary_method == "last":
        return daily_stats[-1]

    if summary_method == "mean":
        return np.mean(daily_stats, axis=0)

    if summary_method == "ewma":
        # Compute exponentially weighted mean
        n_days = len(daily_stats)
        # Weight = 0.5^(age / half_life), where age = n_days - 1 - i for i-th day
        decay = np.log(2) / ewma_halflife_days
        weights = np.exp(-decay * np.arange(n_days - 1, -1, -1, dtype=np.float64))
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            return np.sum(daily_stats * weights[:, np.newaxis], axis=0) / weights_sum
        return np.mean(daily_stats, axis=0)

    # Fallback to mean
    return np.mean(daily_stats, axis=0)


def build_long_term_features_multiscale(
    daily_returns: np.ndarray,
    daily_volumes: np.ndarray,
    daily_timestamps: np.ndarray,
    anchor_timestamp: float,
    config: LongTermConfig,
) -> np.ndarray:
    """Build multi-scale long-term feature vector for a single sample.

    Parameters
    ----------
    daily_returns:
        1D array of all available daily returns.
    daily_volumes:
        1D array of all available daily volumes.
    daily_timestamps:
        1D array of day-start timestamps (Unix seconds).
    anchor_timestamp:
        Timestamp of the anchor snapshot for which to compute features.
    config:
        LongTermConfig with window sizes and feature specifications.

    Returns
    -------
    np.ndarray:
        1D array of shape (n_windows * n_features,) containing concatenated
        features for all windows.

    Notes
    -----
    - Features are computed for each window separately, then concatenated.
    - If a window has insufficient history, zeros are returned for that window.
    - The anchor day is NOT included in the window (to avoid lookahead).
    """
    n_windows = len(config.windows_days)
    n_features = len(config.features)
    result = np.zeros(n_windows * n_features, dtype=np.float32)

    if len(daily_timestamps) == 0:
        return result

    # Find the anchor day index (exclusive - we want days BEFORE anchor)
    seconds_per_day = 86400
    anchor_day_start = (anchor_timestamp // seconds_per_day) * seconds_per_day

    # Find index of first day that is strictly before anchor
    valid_mask = daily_timestamps < anchor_day_start
    if not np.any(valid_mask):
        # No history before anchor
        return result

    valid_indices = np.where(valid_mask)[0]
    end_idx = valid_indices[-1] + 1  # Exclusive end index

    for w_idx, window_days in enumerate(config.windows_days):
        # Calculate start index for this window
        start_idx = max(0, end_idx - window_days)

        if start_idx >= end_idx:
            # No data for this window
            continue

        window_returns = daily_returns[start_idx:end_idx]
        window_volumes = daily_volumes[start_idx:end_idx]

        # Compute per-day stats then summarize
        # For now, compute stats directly on the window
        window_stats = _compute_window_stats(
            window_returns, window_volumes, config.features
        )

        # Place in result array
        feat_start = w_idx * n_features
        feat_end = feat_start + n_features
        result[feat_start:feat_end] = window_stats

    return result


def compute_long_term_features(
    config: Dict[str, Any],
    mid_prices: np.ndarray,
    timestamps: np.ndarray,
    anchor_timestamps: np.ndarray,
    cadence_seconds: int,
    volumes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Main entry point: compute long-term features for all samples.

    Parameters
    ----------
    config:
        Full configuration dictionary (uses model.long_term section).
    mid_prices:
        1D array of mid-prices for all snapshots in the dataset.
    timestamps:
        1D array of Unix timestamps (seconds) for each snapshot.
    anchor_timestamps:
        1D array of anchor timestamps (one per sample).
    cadence_seconds:
        Time interval between snapshots in seconds.
    volumes:
        Optional 1D array of volume proxy values for each snapshot.

    Returns
    -------
    np.ndarray:
        2D array of shape (n_samples, input_dim) containing long-term features.
        If long_term is disabled, returns array of shape (n_samples, 0).

    Raises
    ------
    LongTermFeatureError:
        If configuration is invalid or computation fails.
    """
    lt_config = LongTermConfig.from_config(config)

    n_samples = len(anchor_timestamps)

    if not lt_config.enabled:
        logger.info("Long-term features disabled; returning empty array")
        return np.zeros((n_samples, 0), dtype=np.float32)

    logger.info(
        "Computing long-term features: windows=%s, features=%s, summary=%s",
        lt_config.windows_days,
        lt_config.features,
        lt_config.summary_method,
    )

    # Compute daily aggregates from the full dataset
    daily_timestamps, daily_returns, daily_volumes = compute_daily_aggregates(
        mid_prices=mid_prices,
        timestamps=timestamps,
        volumes=volumes,
        cadence_seconds=cadence_seconds,
    )

    logger.info(
        "Daily aggregates computed: n_days=%d, date_range=[%s, %s]",
        len(daily_timestamps),
        _ts_to_date_str(daily_timestamps[0]) if len(daily_timestamps) > 0 else "N/A",
        _ts_to_date_str(daily_timestamps[-1]) if len(daily_timestamps) > 0 else "N/A",
    )

    input_dim = lt_config.input_dim
    result = np.zeros((n_samples, input_dim), dtype=np.float32)

    for i, anchor_ts in enumerate(anchor_timestamps):
        result[i] = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=float(anchor_ts),
            config=lt_config,
        )

    # Log summary statistics
    nonzero_mask = np.any(result != 0, axis=1)
    n_nonzero = np.sum(nonzero_mask)
    logger.info(
        "Long-term features computed: n_samples=%d, input_dim=%d, "
        "samples_with_history=%d (%.1f%%)",
        n_samples,
        input_dim,
        n_nonzero,
        100.0 * n_nonzero / n_samples if n_samples > 0 else 0.0,
    )

    return result


def _ts_to_date_str(timestamp: float) -> str:
    """Convert Unix timestamp to YYYY-MM-DD string."""
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


__all__ = [
    "LongTermConfig",
    "LongTermFeatureError",
    "SUPPORTED_LT_FEATURES",
    "SUPPORTED_SUMMARY_METHODS",
    "compute_daily_aggregates",
    "build_long_term_features_multiscale",
    "compute_long_term_features",
]
