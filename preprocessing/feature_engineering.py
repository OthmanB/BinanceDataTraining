"""Feature engineering pipeline for order book data.

This module implements the FeatureEngineer class that computes derived features
from order book snapshots as defined in the YAML configuration under
preprocessing.feature_engineering.

Features include:
- Order book features: bid_ask_spread, volume_imbalance, depth_imbalance, weighted_mid_price
- Derived features: price_momentum, volume_momentum (with edge decay for partial windows)

All parameters must come from configuration; no implicit defaults are used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute derived features from order book data.

    This class follows a two-phase pattern:
    - compute_order_book_features(): per-snapshot features from depth data
    - compute_momentum_features(): temporal features requiring historical context

    All configuration must come from YAML; this class does not define defaults.
    """

    SUPPORTED_ORDER_BOOK_FEATURES = (
        "bid_ask_spread",
        "volume_imbalance",
        "depth_imbalance",
        "weighted_mid_price",
    )

    SUPPORTED_DERIVED_FEATURES = (
        "price_momentum",
        "volume_momentum",
    )

    SUPPORTED_VOLUME_PROXY_METHODS = ("top_of_book", "total_depth")
    SUPPORTED_EDGE_DECAY_METHODS = ("linear", "exponential")

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize FeatureEngineer with configuration.

        Parameters
        ----------
        config:
            Full configuration dict. Must contain preprocessing.feature_engineering
            with enabled, order_book_features, derived_features, etc.

        Raises
        ------
        ValueError:
            If configuration is invalid or contains unsupported feature names.
        """
        self._config = config

        fe_cfg = config["preprocessing"]["feature_engineering"]
        self._enabled = bool(fe_cfg["enabled"])

        if not self._enabled:
            self._order_book_features: List[str] = []
            self._derived_features: List[str] = []
            self._momentum_window_seconds = 0
            self._volume_proxy_method = "top_of_book"
            self._edge_decay_enabled = False
            self._edge_decay_method = "linear"
            return

        self._order_book_features = list(fe_cfg["order_book_features"])
        self._derived_features = list(fe_cfg["derived_features"])
        self._momentum_window_seconds = int(fe_cfg["momentum_window_seconds"])
        self._volume_proxy_method = str(fe_cfg["volume_proxy_method"])

        edge_decay_cfg = fe_cfg["edge_decay"]
        self._edge_decay_enabled = bool(edge_decay_cfg["enabled"])
        self._edge_decay_method = str(edge_decay_cfg["method"])

        # Validate configuration
        for feat in self._order_book_features:
            if feat not in self.SUPPORTED_ORDER_BOOK_FEATURES:
                raise ValueError(
                    f"Unsupported order_book_feature: {feat!r}. "
                    f"Must be one of: {self.SUPPORTED_ORDER_BOOK_FEATURES}",
                )

        for feat in self._derived_features:
            if feat not in self.SUPPORTED_DERIVED_FEATURES:
                raise ValueError(
                    f"Unsupported derived_feature: {feat!r}. "
                    f"Must be one of: {self.SUPPORTED_DERIVED_FEATURES}",
                )

        if self._volume_proxy_method not in self.SUPPORTED_VOLUME_PROXY_METHODS:
            raise ValueError(
                f"Unsupported volume_proxy_method: {self._volume_proxy_method!r}. "
                f"Must be one of: {self.SUPPORTED_VOLUME_PROXY_METHODS}",
            )

        if self._edge_decay_enabled:
            if self._edge_decay_method not in self.SUPPORTED_EDGE_DECAY_METHODS:
                raise ValueError(
                    f"Unsupported edge_decay.method: {self._edge_decay_method!r}. "
                    f"Must be one of: {self.SUPPORTED_EDGE_DECAY_METHODS}",
                )

        # Validate momentum window constraint
        targets_cfg = config["targets"]
        visible_window_seconds = int(targets_cfg["visible_window_seconds"])
        if self._momentum_window_seconds > visible_window_seconds:
            raise ValueError(
                f"preprocessing.feature_engineering.momentum_window_seconds "
                f"({self._momentum_window_seconds}) must be <= targets.visible_window_seconds "
                f"({visible_window_seconds})",
            )

    @property
    def enabled(self) -> bool:
        """Return whether feature engineering is enabled."""
        return self._enabled

    @property
    def num_order_book_features(self) -> int:
        """Return the number of order book features to compute."""
        return len(self._order_book_features)

    @property
    def num_derived_features(self) -> int:
        """Return the number of derived features to compute."""
        return len(self._derived_features)

    @property
    def num_total_features(self) -> int:
        """Return the total number of features to compute."""
        return self.num_order_book_features + self.num_derived_features

    def compute_order_book_features(
        self,
        snapshot_depth: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute per-snapshot order book features.

        Parameters
        ----------
        snapshot_depth:
            Dict with keys: bid_prices, bid_quantities, ask_prices, ask_quantities.
            Each is a 1D numpy array of shape (depth_levels,).

        Returns
        -------
        Dict[str, float]:
            Mapping from feature name to computed value.
        """
        if not self._enabled:
            return {}

        bid_prices = snapshot_depth["bid_prices"]
        bid_quantities = snapshot_depth["bid_quantities"]
        ask_prices = snapshot_depth["ask_prices"]
        ask_quantities = snapshot_depth["ask_quantities"]

        features: Dict[str, float] = {}

        # Compute mid_price from top-of-book (needed for several features)
        best_bid = float(bid_prices[0]) if bid_prices[0] > 0 else 0.0
        best_ask = float(ask_prices[0]) if ask_prices[0] > 0 else 0.0

        if best_bid > 0 and best_ask > 0:
            mid_price = 0.5 * (best_bid + best_ask)
        else:
            mid_price = 0.0

        for feat in self._order_book_features:
            if feat == "bid_ask_spread":
                features[feat] = self._compute_bid_ask_spread(
                    best_bid, best_ask, mid_price
                )
            elif feat == "volume_imbalance":
                features[feat] = self._compute_volume_imbalance(
                    bid_quantities, ask_quantities
                )
            elif feat == "depth_imbalance":
                features[feat] = self._compute_depth_imbalance(
                    bid_prices, bid_quantities, ask_prices, ask_quantities, mid_price
                )
            elif feat == "weighted_mid_price":
                features[feat] = self._compute_weighted_mid_price(
                    best_bid, best_ask,
                    float(bid_quantities[0]),
                    float(ask_quantities[0]),
                )

        return features

    def compute_volume_proxy(
        self,
        snapshot_depth: Dict[str, np.ndarray],
    ) -> float:
        """Compute volume proxy using configured method.

        Parameters
        ----------
        snapshot_depth:
            Dict with keys: bid_prices, bid_quantities, ask_prices, ask_quantities.

        Returns
        -------
        float:
            Volume proxy value.
        """
        bid_quantities = snapshot_depth["bid_quantities"]
        ask_quantities = snapshot_depth["ask_quantities"]

        if self._volume_proxy_method == "top_of_book":
            # Use top-of-book quantities as proxy for matched order volume
            return float(bid_quantities[0]) + float(ask_quantities[0])
        else:
            # total_depth: sum all quantities
            return float(np.sum(bid_quantities)) + float(np.sum(ask_quantities))

    def compute_momentum_features(
        self,
        mid_prices: np.ndarray,
        volumes: np.ndarray,
        anchor_idx: int,
        momentum_window_steps: int,
    ) -> Dict[str, float]:
        """Compute momentum features with edge decay for partial windows.

        Parameters
        ----------
        mid_prices:
            1D array of mid-prices for all snapshots.
        volumes:
            1D array of volume proxy values for all snapshots.
        anchor_idx:
            Index of the current anchor snapshot.
        momentum_window_steps:
            Number of steps in the momentum window.

        Returns
        -------
        Dict[str, float]:
            Mapping from feature name to computed value.
        """
        if not self._enabled:
            return {}

        features: Dict[str, float] = {}

        for feat in self._derived_features:
            if feat == "price_momentum":
                features[feat] = self._compute_momentum(
                    mid_prices, anchor_idx, momentum_window_steps
                )
            elif feat == "volume_momentum":
                features[feat] = self._compute_momentum(
                    volumes, anchor_idx, momentum_window_steps
                )

        return features

    def _compute_bid_ask_spread(
        self,
        best_bid: float,
        best_ask: float,
        mid_price: float,
    ) -> float:
        """Compute bid-ask spread as percentage of mid-price.

        Formula: (ask - bid) / mid_price
        Returns 0.0 if mid_price is zero or non-positive.
        """
        if mid_price <= 0 or best_bid <= 0 or best_ask <= 0:
            return 0.0
        spread = (best_ask - best_bid) / mid_price
        return float(spread)

    def _compute_volume_imbalance(
        self,
        bid_quantities: np.ndarray,
        ask_quantities: np.ndarray,
    ) -> float:
        """Compute volume imbalance across all depth levels.

        Formula: (sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty))
        Returns 0.0 if total volume is zero.
        Result is in range [-1, 1].
        """
        bid_sum = float(np.sum(bid_quantities))
        ask_sum = float(np.sum(ask_quantities))
        total = bid_sum + ask_sum

        if total <= 0:
            return 0.0

        imbalance = (bid_sum - ask_sum) / total
        return float(imbalance)

    def _compute_depth_imbalance(
        self,
        bid_prices: np.ndarray,
        bid_quantities: np.ndarray,
        ask_prices: np.ndarray,
        ask_quantities: np.ndarray,
        mid_price: float,
    ) -> float:
        """Compute depth imbalance weighted by distance from mid-price.

        Formula: sum(bid_qty * distance) - sum(ask_qty * distance)
                 normalized by total weighted volume.

        Distance is computed as |price - mid_price| / mid_price.
        Returns 0.0 if mid_price is zero or total weighted volume is zero.
        """
        if mid_price <= 0:
            return 0.0

        # Compute distances from mid-price (normalized)
        bid_distances = np.abs(bid_prices - mid_price) / mid_price
        ask_distances = np.abs(ask_prices - mid_price) / mid_price

        # Weight quantities by distance
        bid_weighted = bid_quantities * bid_distances
        ask_weighted = ask_quantities * ask_distances

        bid_weighted_sum = float(np.sum(bid_weighted))
        ask_weighted_sum = float(np.sum(ask_weighted))
        total_weighted = bid_weighted_sum + ask_weighted_sum

        if total_weighted <= 0:
            return 0.0

        imbalance = (bid_weighted_sum - ask_weighted_sum) / total_weighted
        return float(imbalance)

    def _compute_weighted_mid_price(
        self,
        best_bid: float,
        best_ask: float,
        bid_qty: float,
        ask_qty: float,
    ) -> float:
        """Compute volume-weighted mid-price.

        Formula: (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
        Returns simple mid-price if total quantity is zero.
        """
        total_qty = bid_qty + ask_qty
        if total_qty <= 0:
            if best_bid > 0 and best_ask > 0:
                return 0.5 * (best_bid + best_ask)
            return 0.0

        weighted_mid = (best_bid * ask_qty + best_ask * bid_qty) / total_qty
        return float(weighted_mid)

    def _compute_momentum(
        self,
        values: np.ndarray,
        anchor_idx: int,
        window_steps: int,
    ) -> float:
        """Compute momentum with edge decay for partial windows.

        Formula: (value[anchor] - value[anchor - window]) / value[anchor - window]
        With edge decay when anchor_idx < window_steps.
        """
        if anchor_idx < 0 or anchor_idx >= len(values):
            return 0.0

        current_value = float(values[anchor_idx])

        # Compute effective window (clamped to available history)
        effective_window = min(anchor_idx, window_steps)
        if effective_window <= 0:
            return 0.0

        past_idx = anchor_idx - effective_window
        past_value = float(values[past_idx])

        if past_value <= 0:
            return 0.0

        raw_momentum = (current_value - past_value) / past_value

        # Apply edge decay if enabled and window is partial
        if self._edge_decay_enabled and effective_window < window_steps:
            decay_weight = self._compute_decay_weight(effective_window, window_steps)
            return float(raw_momentum * decay_weight)

        return float(raw_momentum)

    def _compute_decay_weight(
        self,
        effective_window: int,
        full_window: int,
    ) -> float:
        """Compute edge decay weight for partial windows.

        Parameters
        ----------
        effective_window:
            Actual window size available.
        full_window:
            Full configured window size.

        Returns
        -------
        float:
            Decay weight in range (0, 1].
        """
        if full_window <= 0:
            return 1.0

        ratio = float(effective_window) / float(full_window)

        if self._edge_decay_method == "linear":
            return ratio
        else:
            # Exponential decay: 1 - exp(-effective / (full / 3))
            decay_constant = float(full_window) / 3.0
            if decay_constant <= 0:
                return ratio
            return float(1.0 - np.exp(-float(effective_window) / decay_constant))

    def compute_all_features(
        self,
        snapshot_depth_data: List[Dict[str, np.ndarray]],
        mid_prices: np.ndarray,
        anchor_indices: List[int],
        cadence_seconds: int,
    ) -> Optional[np.ndarray]:
        """Compute all features for all samples.

        Parameters
        ----------
        snapshot_depth_data:
            List of snapshot depth dicts, one per snapshot.
        mid_prices:
            1D array of mid-prices for all snapshots.
        anchor_indices:
            List of anchor indices for each sample.
        cadence_seconds:
            Cadence in seconds between snapshots.

        Returns
        -------
        Optional[np.ndarray]:
            Feature array of shape (N, num_total_features), or None if disabled.
        """
        if not self._enabled:
            return None

        if not anchor_indices:
            return None

        n_samples = len(anchor_indices)
        n_features = self.num_total_features

        if n_features == 0:
            return None

        features = np.zeros((n_samples, n_features), dtype="float32")

        # Compute momentum window in steps
        momentum_window_steps = self._momentum_window_seconds // cadence_seconds

        # Precompute volume proxy for all snapshots
        volumes = np.zeros(len(snapshot_depth_data), dtype="float64")
        for i, depth in enumerate(snapshot_depth_data):
            volumes[i] = self.compute_volume_proxy(depth)

        # Compute features for each sample
        for sample_idx, anchor_idx in enumerate(anchor_indices):
            feature_idx = 0

            # Order book features from anchor snapshot
            if anchor_idx < len(snapshot_depth_data):
                ob_features = self.compute_order_book_features(
                    snapshot_depth_data[anchor_idx]
                )
                for feat_name in self._order_book_features:
                    features[sample_idx, feature_idx] = ob_features.get(feat_name, 0.0)
                    feature_idx += 1

            # Momentum features
            momentum_features = self.compute_momentum_features(
                mid_prices, volumes, anchor_idx, momentum_window_steps
            )
            for feat_name in self._derived_features:
                features[sample_idx, feature_idx] = momentum_features.get(feat_name, 0.0)
                feature_idx += 1

        logger.info(
            "Computed all features: n_samples=%s, n_features=%s, "
            "order_book_features=%s, derived_features=%s",
            n_samples,
            n_features,
            self._order_book_features,
            self._derived_features,
        )

        return features


def create_feature_engineer(config: Dict[str, Any]) -> FeatureEngineer:
    """Factory function to create a FeatureEngineer from configuration.

    Parameters
    ----------
    config:
        Full configuration dict.

    Returns
    -------
    FeatureEngineer:
        Configured feature engineer instance.
    """
    return FeatureEngineer(config)


__all__ = ["FeatureEngineer", "create_feature_engineer"]
