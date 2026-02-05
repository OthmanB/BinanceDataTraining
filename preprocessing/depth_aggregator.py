"""Order book depth aggregation for hybrid representation.

This module implements functions to validate hybrid configuration and aggregate
full order book depth levels into a hybrid representation combining:
- High-resolution near market (top N levels kept as-is)
- Aggregated deeper levels (remaining levels binned)

All parameters must be provided via YAML configuration; no implicit defaults are used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


def validate_hybrid_config(config: Dict[str, Any]) -> None:
    """Validate hybrid order book configuration.

    Checks that:
    - raw_levels is positive and less than depth_levels
    - aggregated_bins is positive
    - raw_levels + aggregated_bins <= depth_levels
    - bin_strategy is one of ["equal_width", "log_spaced"]

    Parameters
    ----------
    config:
        Full configuration dictionary. Expects data.order_book section
        with depth_levels and hybrid subsection.

    Raises
    ------
    ValueError:
        If any validation constraint is violated.
    KeyError:
        If required configuration keys are missing.
    """
    data_cfg = config["data"]
    order_book_cfg = data_cfg["order_book"]

    depth_levels = int(order_book_cfg["depth_levels"])
    representation = str(order_book_cfg["representation"])

    if representation != "hybrid":
        # No validation needed for non-hybrid representations
        return

    hybrid_cfg = order_book_cfg["hybrid"]

    raw_levels = int(hybrid_cfg["raw_levels"])
    aggregated_bins = int(hybrid_cfg["aggregated_bins"])
    bin_strategy = str(hybrid_cfg["bin_strategy"])

    # Validate raw_levels
    if raw_levels <= 0:
        raise ValueError(
            f"data.order_book.hybrid.raw_levels must be positive; got {raw_levels}",
        )
    if raw_levels >= depth_levels:
        raise ValueError(
            f"data.order_book.hybrid.raw_levels must be less than depth_levels; "
            f"got raw_levels={raw_levels}, depth_levels={depth_levels}",
        )

    # Validate aggregated_bins
    if aggregated_bins <= 0:
        raise ValueError(
            f"data.order_book.hybrid.aggregated_bins must be positive; got {aggregated_bins}",
        )

    # Validate combined constraint
    if raw_levels + aggregated_bins > depth_levels:
        raise ValueError(
            f"data.order_book.hybrid constraint violated: raw_levels + aggregated_bins must be <= depth_levels; "
            f"got raw_levels={raw_levels}, aggregated_bins={aggregated_bins}, depth_levels={depth_levels}",
        )

    # Validate bin_strategy
    valid_strategies = ("equal_width", "log_spaced")
    if bin_strategy not in valid_strategies:
        raise ValueError(
            f"data.order_book.hybrid.bin_strategy must be one of {valid_strategies}; "
            f"got {bin_strategy!r}",
        )

    logger.info(
        "Hybrid config validated: raw_levels=%d, aggregated_bins=%d, bin_strategy=%s, depth_levels=%d",
        raw_levels,
        aggregated_bins,
        bin_strategy,
        depth_levels,
    )


def get_hybrid_output_shape(config: Dict[str, Any]) -> int:
    """Compute the number of effective levels in hybrid representation.

    Parameters
    ----------
    config:
        Full configuration dictionary.

    Returns
    -------
    effective_levels:
        Total number of levels in hybrid output = raw_levels + aggregated_bins.
    """
    data_cfg = config["data"]
    order_book_cfg = data_cfg["order_book"]
    hybrid_cfg = order_book_cfg["hybrid"]

    raw_levels = int(hybrid_cfg["raw_levels"])
    aggregated_bins = int(hybrid_cfg["aggregated_bins"])

    return raw_levels + aggregated_bins


def compute_bin_boundaries(
    depth_levels: int,
    raw_levels: int,
    aggregated_bins: int,
    bin_strategy: str,
) -> np.ndarray:
    """Compute bin boundaries for aggregating deeper order book levels.

    Parameters
    ----------
    depth_levels:
        Total number of depth levels available.
    raw_levels:
        Number of levels to keep as-is (near market).
    aggregated_bins:
        Number of bins to create for remaining levels.
    bin_strategy:
        Either "equal_width" for uniform bins or "log_spaced" for
        logarithmically spaced bins (more granular near market).

    Returns
    -------
    boundaries:
        Array of shape (aggregated_bins + 1,) containing bin edges.
        boundaries[i] and boundaries[i+1] define the range for bin i.
    """
    remaining_levels = depth_levels - raw_levels

    if remaining_levels <= 0:
        return np.array([raw_levels], dtype=np.int64)

    if bin_strategy == "equal_width":
        # Uniform bin boundaries
        boundaries = np.linspace(
            raw_levels,
            depth_levels,
            aggregated_bins + 1,
            dtype=np.float64,
        )
    elif bin_strategy == "log_spaced":
        # Logarithmically spaced boundaries (more granular near market)
        # Use log scale starting from 1 to avoid log(0)
        log_boundaries = np.logspace(
            0,
            np.log10(remaining_levels),
            aggregated_bins + 1,
        )
        # Shift to start at raw_levels
        boundaries = raw_levels + log_boundaries - 1
        # Ensure last boundary is exactly depth_levels
        boundaries[-1] = depth_levels
    else:
        raise ValueError(f"Unknown bin_strategy: {bin_strategy!r}")

    return np.round(boundaries).astype(np.int64)


def aggregate_depth_levels(
    depth_prices: np.ndarray,
    depth_quantities: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate full depth levels into hybrid representation.

    Takes the full order book depth and produces a hybrid representation:
    - Top raw_levels are kept as-is
    - Remaining levels are aggregated into bins using volume-weighted average price
      and sum of quantities

    Parameters
    ----------
    depth_prices:
        Array of shape (n_levels,) containing prices for each level.
    depth_quantities:
        Array of shape (n_levels,) containing quantities for each level.
    config:
        Full configuration dictionary with data.order_book.hybrid settings.

    Returns
    -------
    hybrid_prices:
        Array of shape (raw_levels + aggregated_bins,) with hybrid prices.
    hybrid_quantities:
        Array of shape (raw_levels + aggregated_bins,) with hybrid quantities.
    """
    data_cfg = config["data"]
    order_book_cfg = data_cfg["order_book"]
    hybrid_cfg = order_book_cfg["hybrid"]

    depth_levels = int(order_book_cfg["depth_levels"])
    raw_levels = int(hybrid_cfg["raw_levels"])
    aggregated_bins = int(hybrid_cfg["aggregated_bins"])
    bin_strategy = str(hybrid_cfg["bin_strategy"])

    n_input_levels = len(depth_prices)

    # Pad input if shorter than expected
    if n_input_levels < depth_levels:
        pad_size = depth_levels - n_input_levels
        depth_prices = np.pad(depth_prices, (0, pad_size), constant_values=0.0)
        depth_quantities = np.pad(depth_quantities, (0, pad_size), constant_values=0.0)

    effective_levels = raw_levels + aggregated_bins

    hybrid_prices = np.zeros(effective_levels, dtype=np.float64)
    hybrid_quantities = np.zeros(effective_levels, dtype=np.float64)

    # Keep top raw_levels as-is
    hybrid_prices[:raw_levels] = depth_prices[:raw_levels]
    hybrid_quantities[:raw_levels] = depth_quantities[:raw_levels]

    # Aggregate remaining levels into bins
    boundaries = compute_bin_boundaries(
        depth_levels=depth_levels,
        raw_levels=raw_levels,
        aggregated_bins=aggregated_bins,
        bin_strategy=bin_strategy,
    )

    for bin_idx in range(aggregated_bins):
        start_idx = int(boundaries[bin_idx])
        end_idx = int(boundaries[bin_idx + 1])

        if start_idx >= end_idx:
            continue

        bin_prices = depth_prices[start_idx:end_idx]
        bin_quantities = depth_quantities[start_idx:end_idx]

        total_qty = np.sum(bin_quantities)
        if total_qty > 0:
            # Volume-weighted average price
            vwap = np.sum(bin_prices * bin_quantities) / total_qty
            hybrid_prices[raw_levels + bin_idx] = vwap
            hybrid_quantities[raw_levels + bin_idx] = total_qty
        else:
            # No volume in this bin
            hybrid_prices[raw_levels + bin_idx] = np.mean(bin_prices) if len(bin_prices) > 0 else 0.0
            hybrid_quantities[raw_levels + bin_idx] = 0.0

    return hybrid_prices, hybrid_quantities


def aggregate_snapshot_to_hybrid(
    bid_prices: np.ndarray,
    bid_quantities: np.ndarray,
    ask_prices: np.ndarray,
    ask_quantities: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Aggregate a full order book snapshot into hybrid representation.

    Parameters
    ----------
    bid_prices:
        Array of bid prices ordered from best (highest) to worst.
    bid_quantities:
        Array of bid quantities corresponding to bid_prices.
    ask_prices:
        Array of ask prices ordered from best (lowest) to worst.
    ask_quantities:
        Array of ask quantities corresponding to ask_prices.
    config:
        Full configuration dictionary.

    Returns
    -------
    hybrid_snapshot:
        Array of shape (effective_levels, 4) where:
        - column 0: bid_price
        - column 1: bid_quantity
        - column 2: ask_price
        - column 3: ask_quantity
    """
    effective_levels = get_hybrid_output_shape(config)

    hybrid_bid_prices, hybrid_bid_qtys = aggregate_depth_levels(
        bid_prices, bid_quantities, config
    )
    hybrid_ask_prices, hybrid_ask_qtys = aggregate_depth_levels(
        ask_prices, ask_quantities, config
    )

    # Stack into (effective_levels, 4) array
    hybrid_snapshot = np.stack(
        [hybrid_bid_prices, hybrid_bid_qtys, hybrid_ask_prices, hybrid_ask_qtys],
        axis=1,
    )

    return hybrid_snapshot.astype(np.float32)


__all__ = [
    "validate_hybrid_config",
    "get_hybrid_output_shape",
    "compute_bin_boundaries",
    "aggregate_depth_levels",
    "aggregate_snapshot_to_hybrid",
]
