"""Unit tests for feature engineering.

Tests cover:
- Order book features: bid_ask_spread, volume_imbalance, depth_imbalance, weighted_mid_price
- Derived features: price_momentum, volume_momentum
- Edge decay: linear and exponential methods
- Property-based tests for value ranges
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from preprocessing.feature_engineering import FeatureEngineer


def _build_minimal_config(
    enabled: bool = True,
    order_book_features: list = None,
    derived_features: list = None,
    momentum_window_seconds: int = 300,
    volume_proxy_method: str = "top_of_book",
    edge_decay_enabled: bool = True,
    edge_decay_method: str = "linear",
) -> dict:
    """Build minimal valid config for FeatureEngineer."""
    if order_book_features is None:
        order_book_features = ["bid_ask_spread", "volume_imbalance"]
    if derived_features is None:
        derived_features = ["price_momentum"]

    return {
        "preprocessing": {
            "feature_engineering": {
                "enabled": enabled,
                "order_book_features": order_book_features,
                "derived_features": derived_features,
                "momentum_window_seconds": momentum_window_seconds,
                "volume_proxy_method": volume_proxy_method,
                "edge_decay": {
                    "enabled": edge_decay_enabled,
                    "method": edge_decay_method,
                },
            },
        },
        "targets": {
            "visible_window_seconds": 3600,  # Must be >= momentum_window_seconds
        },
    }


def _build_snapshot_depth(
    best_bid: float = 100.0,
    best_ask: float = 101.0,
    best_bid_qty: float = 10.0,
    best_ask_qty: float = 15.0,
    depth_levels: int = 5,
) -> dict:
    """Build a minimal snapshot depth dict for testing."""
    bid_prices = np.zeros(depth_levels, dtype="float64")
    bid_quantities = np.zeros(depth_levels, dtype="float64")
    ask_prices = np.zeros(depth_levels, dtype="float64")
    ask_quantities = np.zeros(depth_levels, dtype="float64")

    bid_prices[0] = best_bid
    bid_quantities[0] = best_bid_qty
    ask_prices[0] = best_ask
    ask_quantities[0] = best_ask_qty

    # Add some depth levels
    for i in range(1, min(3, depth_levels)):
        bid_prices[i] = best_bid - i * 0.1
        bid_quantities[i] = best_bid_qty * (1.0 - 0.1 * i)
        ask_prices[i] = best_ask + i * 0.1
        ask_quantities[i] = best_ask_qty * (1.0 - 0.1 * i)

    return {
        "bid_prices": bid_prices,
        "bid_quantities": bid_quantities,
        "ask_prices": ask_prices,
        "ask_quantities": ask_quantities,
    }


class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""

    def test_disabled_engineer(self) -> None:
        """When disabled, no features are computed."""
        config = _build_minimal_config(enabled=False)
        engineer = FeatureEngineer(config)

        assert not engineer.enabled
        assert engineer.num_order_book_features == 0
        assert engineer.num_derived_features == 0

    def test_valid_config(self) -> None:
        """Valid config creates working engineer."""
        config = _build_minimal_config(
            order_book_features=["bid_ask_spread", "volume_imbalance"],
            derived_features=["price_momentum"],
        )
        engineer = FeatureEngineer(config)

        assert engineer.enabled
        assert engineer.num_order_book_features == 2
        assert engineer.num_derived_features == 1
        assert engineer.num_total_features == 3

    def test_invalid_order_book_feature_raises(self) -> None:
        """Invalid order book feature name raises ValueError."""
        config = _build_minimal_config(order_book_features=["invalid_feature"])
        with pytest.raises(ValueError, match="Unsupported order_book_feature"):
            FeatureEngineer(config)

    def test_invalid_derived_feature_raises(self) -> None:
        """Invalid derived feature name raises ValueError."""
        config = _build_minimal_config(derived_features=["invalid_feature"])
        with pytest.raises(ValueError, match="Unsupported derived_feature"):
            FeatureEngineer(config)

    def test_invalid_volume_proxy_method_raises(self) -> None:
        """Invalid volume proxy method raises ValueError."""
        config = _build_minimal_config(volume_proxy_method="invalid_method")
        with pytest.raises(ValueError, match="Unsupported volume_proxy_method"):
            FeatureEngineer(config)

    def test_momentum_window_exceeds_visible_window_raises(self) -> None:
        """momentum_window_seconds > visible_window_seconds raises ValueError."""
        config = _build_minimal_config(momentum_window_seconds=4000)
        # visible_window_seconds is 3600
        with pytest.raises(ValueError, match="must be <="):
            FeatureEngineer(config)


class TestOrderBookFeatures:
    """Tests for order book feature computations."""

    def test_bid_ask_spread(self) -> None:
        """Test bid-ask spread calculation."""
        config = _build_minimal_config(order_book_features=["bid_ask_spread"])
        engineer = FeatureEngineer(config)

        # best_bid=100, best_ask=101, mid=100.5
        # spread = (101 - 100) / 100.5 ≈ 0.00995
        depth = _build_snapshot_depth(best_bid=100.0, best_ask=101.0)
        features = engineer.compute_order_book_features(depth)

        assert "bid_ask_spread" in features
        expected = (101.0 - 100.0) / 100.5
        assert features["bid_ask_spread"] == pytest.approx(expected)

    def test_bid_ask_spread_nonnegative(self) -> None:
        """Spread should always be >= 0."""
        config = _build_minimal_config(order_book_features=["bid_ask_spread"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid=100.0, best_ask=100.5)
        features = engineer.compute_order_book_features(depth)

        assert features["bid_ask_spread"] >= 0

    def test_volume_imbalance(self) -> None:
        """Test volume imbalance calculation."""
        config = _build_minimal_config(order_book_features=["volume_imbalance"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=20.0, best_ask_qty=10.0)
        features = engineer.compute_order_book_features(depth)

        assert "volume_imbalance" in features
        # More bids than asks -> positive imbalance
        assert features["volume_imbalance"] > 0

    def test_volume_imbalance_range(self) -> None:
        """Volume imbalance should be in [-1, 1]."""
        config = _build_minimal_config(order_book_features=["volume_imbalance"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=100.0, best_ask_qty=1.0)
        features = engineer.compute_order_book_features(depth)

        assert -1.0 <= features["volume_imbalance"] <= 1.0

    def test_weighted_mid_price(self) -> None:
        """Test weighted mid-price calculation."""
        config = _build_minimal_config(order_book_features=["weighted_mid_price"])
        engineer = FeatureEngineer(config)

        # bid=100, ask=102, bid_qty=10, ask_qty=30
        # weighted = (100*30 + 102*10) / 40 = 4020/40 = 100.5
        depth = _build_snapshot_depth(
            best_bid=100.0, best_ask=102.0,
            best_bid_qty=10.0, best_ask_qty=30.0,
        )
        features = engineer.compute_order_book_features(depth)

        assert "weighted_mid_price" in features
        expected = (100.0 * 30.0 + 102.0 * 10.0) / 40.0
        assert features["weighted_mid_price"] == pytest.approx(expected)


class TestVolumeProxy:
    """Tests for volume proxy computation."""

    def test_top_of_book_proxy(self) -> None:
        """Top-of-book volume proxy uses only top level."""
        config = _build_minimal_config(volume_proxy_method="top_of_book")
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=10.0, best_ask_qty=15.0)
        proxy = engineer.compute_volume_proxy(depth)

        assert proxy == pytest.approx(25.0)

    def test_total_depth_proxy(self) -> None:
        """Total depth volume proxy sums all levels."""
        config = _build_minimal_config(volume_proxy_method="total_depth")
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=10.0, best_ask_qty=15.0)
        proxy = engineer.compute_volume_proxy(depth)

        # With default setup, should be sum of all quantities
        assert proxy >= 25.0  # At least top level


class TestMomentumFeatures:
    """Tests for momentum feature computations."""

    def test_price_momentum_full_window(self) -> None:
        """Price momentum with full window available."""
        config = _build_minimal_config(
            derived_features=["price_momentum"],
            momentum_window_seconds=100,
            edge_decay_enabled=False,
        )
        engineer = FeatureEngineer(config)

        # 10 second cadence -> momentum window = 10 steps
        mid_prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
        volumes = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        features = engineer.compute_momentum_features(
            mid_prices, volumes, anchor_idx=5, momentum_window_steps=5,
        )

        assert "price_momentum" in features
        # (110 - 100) / 100 = 0.1
        assert features["price_momentum"] == pytest.approx(0.1)

    def test_price_momentum_partial_window_with_decay(self) -> None:
        """Price momentum with partial window applies edge decay."""
        config = _build_minimal_config(
            derived_features=["price_momentum"],
            edge_decay_enabled=True,
            edge_decay_method="linear",
        )
        engineer = FeatureEngineer(config)

        mid_prices = np.array([100.0, 110.0, 120.0])  # Only 3 snapshots
        volumes = np.array([10.0, 10.0, 10.0])

        # anchor_idx=1, window=5 -> effective=1 (partial)
        features = engineer.compute_momentum_features(
            mid_prices, volumes, anchor_idx=1, momentum_window_steps=5,
        )

        # Raw momentum: (110 - 100) / 100 = 0.1
        # Decay: 1/5 = 0.2
        # Result: 0.1 * 0.2 = 0.02
        assert features["price_momentum"] == pytest.approx(0.02)

    def test_decay_weight_linear(self) -> None:
        """Linear decay weight is proportional to window coverage."""
        config = _build_minimal_config(edge_decay_method="linear")
        engineer = FeatureEngineer(config)

        # Full window -> 1.0
        assert engineer._compute_decay_weight(10, 10) == pytest.approx(1.0)
        # Half window -> 0.5
        assert engineer._compute_decay_weight(5, 10) == pytest.approx(0.5)
        # Quarter window -> 0.25
        assert engineer._compute_decay_weight(25, 100) == pytest.approx(0.25)

    def test_decay_weight_exponential(self) -> None:
        """Exponential decay approaches 1.0 for full windows."""
        config = _build_minimal_config(edge_decay_method="exponential")
        engineer = FeatureEngineer(config)

        # Full window should be close to 1.0
        weight_full = engineer._compute_decay_weight(10, 10)
        assert weight_full > 0.9

        # Partial window should be less than full
        weight_partial = engineer._compute_decay_weight(5, 10)
        assert weight_partial < weight_full


class TestComputeAllFeatures:
    """Tests for compute_all_features integration method."""

    def test_compute_all_features_shape(self) -> None:
        """Output shape is (N, num_features)."""
        config = _build_minimal_config(
            order_book_features=["bid_ask_spread"],
            derived_features=["price_momentum"],
        )
        engineer = FeatureEngineer(config)

        snapshot_depth_data = [
            _build_snapshot_depth() for _ in range(10)
        ]
        mid_prices = np.linspace(100.0, 110.0, 10)
        anchor_indices = [3, 5, 7]

        features = engineer.compute_all_features(
            snapshot_depth_data=snapshot_depth_data,
            mid_prices=mid_prices,
            anchor_indices=anchor_indices,
            cadence_seconds=10,
        )

        assert features is not None
        assert features.shape == (3, 2)  # 3 samples, 2 features

    def test_compute_all_features_disabled(self) -> None:
        """Returns None when disabled."""
        config = _build_minimal_config(enabled=False)
        engineer = FeatureEngineer(config)

        result = engineer.compute_all_features(
            snapshot_depth_data=[],
            mid_prices=np.array([]),
            anchor_indices=[],
            cadence_seconds=10,
        )

        assert result is None


class TestFeatureEngineerProperties:
    """Property-based tests for FeatureEngineer."""

    @given(
        best_bid=st.floats(min_value=10.0, max_value=1000.0),
        best_ask=st.floats(min_value=10.0, max_value=1000.0),
    )
    def test_spread_nonnegative(self, best_bid: float, best_ask: float) -> None:
        """Spread is always non-negative when bid < ask."""
        if best_bid >= best_ask:
            return  # Skip invalid order book

        config = _build_minimal_config(order_book_features=["bid_ask_spread"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid=best_bid, best_ask=best_ask)
        features = engineer.compute_order_book_features(depth)

        assert features["bid_ask_spread"] >= 0

    @given(
        bid_qty=st.floats(min_value=0.1, max_value=1000.0),
        ask_qty=st.floats(min_value=0.1, max_value=1000.0),
    )
    def test_imbalance_in_range(self, bid_qty: float, ask_qty: float) -> None:
        """Volume imbalance is always in [-1, 1]."""
        config = _build_minimal_config(order_book_features=["volume_imbalance"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=bid_qty, best_ask_qty=ask_qty)
        features = engineer.compute_order_book_features(depth)

        assert -1.0 <= features["volume_imbalance"] <= 1.0

    @given(
        effective_window=st.integers(min_value=1, max_value=100),
        full_window=st.integers(min_value=1, max_value=100),
    )
    def test_decay_weight_in_range(self, effective_window: int, full_window: int) -> None:
        """Decay weight is always in (0, 1] for valid inputs."""
        if effective_window > full_window:
            effective_window = full_window

        config = _build_minimal_config(edge_decay_method="linear")
        engineer = FeatureEngineer(config)

        weight = engineer._compute_decay_weight(effective_window, full_window)

        assert 0.0 < weight <= 1.0

    @given(
        best_bid=st.floats(min_value=50.0, max_value=500.0),
        best_ask=st.floats(min_value=50.0, max_value=500.0),
        bid_qty=st.floats(min_value=0.1, max_value=100.0),
        ask_qty=st.floats(min_value=0.1, max_value=100.0),
    )
    def test_weighted_mid_between_bid_and_ask(
        self, best_bid: float, best_ask: float, bid_qty: float, ask_qty: float
    ) -> None:
        """Weighted mid-price is always between best bid and best ask."""
        if best_bid >= best_ask:
            return  # Skip invalid order book

        config = _build_minimal_config(order_book_features=["weighted_mid_price"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(
            best_bid=best_bid, best_ask=best_ask,
            best_bid_qty=bid_qty, best_ask_qty=ask_qty,
        )
        features = engineer.compute_order_book_features(depth)

        weighted_mid = features["weighted_mid_price"]
        assert best_bid <= weighted_mid <= best_ask, (
            f"Weighted mid {weighted_mid} not in [{best_bid}, {best_ask}]"
        )

    @given(
        bid_qty=st.floats(min_value=0.1, max_value=1000.0),
        ask_qty=st.floats(min_value=0.1, max_value=1000.0),
    )
    def test_volume_proxy_positive(self, bid_qty: float, ask_qty: float) -> None:
        """Volume proxy is always positive for valid depth."""
        config = _build_minimal_config(volume_proxy_method="top_of_book")
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=bid_qty, best_ask_qty=ask_qty)
        proxy = engineer.compute_volume_proxy(depth)

        assert proxy > 0, f"Volume proxy should be positive, got {proxy}"

    @given(
        effective_window=st.integers(min_value=1, max_value=100),
        full_window=st.integers(min_value=1, max_value=100),
    )
    def test_exponential_decay_in_range(self, effective_window: int, full_window: int) -> None:
        """Exponential decay weight is always in (0, 1] for valid inputs."""
        if effective_window > full_window:
            effective_window = full_window

        config = _build_minimal_config(edge_decay_method="exponential")
        engineer = FeatureEngineer(config)

        weight = engineer._compute_decay_weight(effective_window, full_window)

        assert 0.0 < weight <= 1.0, f"Exponential decay weight {weight} not in (0, 1]"

    @given(
        n_samples=st.integers(min_value=1, max_value=20),
        n_snapshots=st.integers(min_value=5, max_value=30),
    )
    def test_compute_all_features_shape_invariant(
        self, n_samples: int, n_snapshots: int
    ) -> None:
        """Output shape is always (len(anchor_indices), num_features)."""
        config = _build_minimal_config(
            order_book_features=["bid_ask_spread", "volume_imbalance"],
            derived_features=["price_momentum"],
        )
        engineer = FeatureEngineer(config)

        snapshot_depth_data = [_build_snapshot_depth() for _ in range(n_snapshots)]
        mid_prices = np.linspace(100.0, 110.0, n_snapshots)

        # Generate valid anchor indices (must be < n_snapshots)
        max_anchor = n_snapshots - 1
        anchor_indices = [
            min(i * 2, max_anchor) for i in range(min(n_samples, (n_snapshots + 1) // 2))
        ]

        if not anchor_indices:
            return  # Skip if no valid anchors

        features = engineer.compute_all_features(
            snapshot_depth_data=snapshot_depth_data,
            mid_prices=mid_prices,
            anchor_indices=anchor_indices,
            cadence_seconds=10,
        )

        assert features is not None
        assert features.shape[0] == len(anchor_indices)
        assert features.shape[1] == engineer.num_total_features

    @given(
        price_change=st.floats(min_value=-0.5, max_value=0.5),
    )
    def test_momentum_magnitude_symmetry(self, price_change: float) -> None:
        """Momentum magnitude should be symmetric for +X% and -X% changes."""
        config = _build_minimal_config(
            derived_features=["price_momentum"],
            edge_decay_enabled=False,
        )
        engineer = FeatureEngineer(config)

        # Create price series with known change
        base_price = 100.0
        end_price_up = base_price * (1 + price_change)
        end_price_down = base_price * (1 - price_change)

        mid_prices_up = np.array([base_price, end_price_up])
        mid_prices_down = np.array([base_price, end_price_down])
        volumes = np.array([10.0, 10.0])

        features_up = engineer.compute_momentum_features(
            mid_prices_up, volumes, anchor_idx=1, momentum_window_steps=1
        )
        features_down = engineer.compute_momentum_features(
            mid_prices_down, volumes, anchor_idx=1, momentum_window_steps=1
        )

        # Magnitudes should be approximately equal (not exactly due to asymmetry in ratios)
        # For small changes, they should be very close
        if abs(price_change) < 0.1:
            assert abs(abs(features_up["price_momentum"]) - abs(features_down["price_momentum"])) < 0.01

    @given(
        bid_qty=st.floats(min_value=0.1, max_value=1000.0),
        ask_qty=st.floats(min_value=0.1, max_value=1000.0),
    )
    def test_total_depth_proxy_geq_top_of_book(self, bid_qty: float, ask_qty: float) -> None:
        """Total depth proxy should be >= top-of-book proxy."""
        config_tob = _build_minimal_config(volume_proxy_method="top_of_book")
        config_total = _build_minimal_config(volume_proxy_method="total_depth")

        engineer_tob = FeatureEngineer(config_tob)
        engineer_total = FeatureEngineer(config_total)

        depth = _build_snapshot_depth(best_bid_qty=bid_qty, best_ask_qty=ask_qty)

        proxy_tob = engineer_tob.compute_volume_proxy(depth)
        proxy_total = engineer_total.compute_volume_proxy(depth)

        assert proxy_total >= proxy_tob, (
            f"Total depth proxy {proxy_total} should be >= top-of-book {proxy_tob}"
        )

    @given(
        bid_qty=st.floats(min_value=0.1, max_value=1000.0),
        ask_qty=st.floats(min_value=0.1, max_value=1000.0),
    )
    def test_imbalance_sign_matches_dominant_side(self, bid_qty: float, ask_qty: float) -> None:
        """Volume imbalance sign should match the dominant side."""
        config = _build_minimal_config(order_book_features=["volume_imbalance"])
        engineer = FeatureEngineer(config)

        depth = _build_snapshot_depth(best_bid_qty=bid_qty, best_ask_qty=ask_qty)
        features = engineer.compute_order_book_features(depth)

        imbalance = features["volume_imbalance"]

        # With default setup, imbalance should reflect top-of-book dominance
        # Note: full depth includes more levels, so this is approximate
        if bid_qty > ask_qty * 1.5:  # Strong bid dominance
            assert imbalance > 0, f"Expected positive imbalance with bid_qty={bid_qty}, ask_qty={ask_qty}"
        elif ask_qty > bid_qty * 1.5:  # Strong ask dominance
            assert imbalance < 0, f"Expected negative imbalance with bid_qty={bid_qty}, ask_qty={ask_qty}"
