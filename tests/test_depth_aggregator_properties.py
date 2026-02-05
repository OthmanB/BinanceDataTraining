"""Property-based tests for the depth_aggregator module using hypothesis.

These tests verify invariants that should hold for all valid configurations
and input data.
"""

import unittest
import os

import numpy as np

# Reduce hypothesis examples in CI for faster test runs
_CI_MODE = os.environ.get("CI", "").lower() in ("true", "1", "yes")
_MAX_EXAMPLES = 10 if _CI_MODE else 100

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from preprocessing.depth_aggregator import (
    validate_hybrid_config,
    get_hybrid_output_shape,
    compute_bin_boundaries,
    aggregate_depth_levels,
    aggregate_snapshot_to_hybrid,
)


def _make_hybrid_config(
    depth_levels: int,
    raw_levels: int,
    aggregated_bins: int,
    bin_strategy: str,
) -> dict:
    """Create a valid hybrid configuration dictionary."""
    return {
        "data": {
            "order_book": {
                "depth_levels": depth_levels,
                "representation": "hybrid",
                "hybrid": {
                    "raw_levels": raw_levels,
                    "aggregated_bins": aggregated_bins,
                    "bin_strategy": bin_strategy,
                },
            }
        }
    }


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestDepthAggregatorProperties(unittest.TestCase):
    """Property-based tests for depth_aggregator functions."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=2000),
        raw_levels=st.integers(min_value=1, max_value=99),
        aggregated_bins=st.integers(min_value=1, max_value=100),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_output_shape_invariant(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: get_hybrid_output_shape == raw_levels + aggregated_bins."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        config = _make_hybrid_config(depth_levels, raw_levels, aggregated_bins, bin_strategy)
        shape = get_hybrid_output_shape(config)
        self.assertEqual(shape, raw_levels + aggregated_bins)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=2000),
        raw_levels=st.integers(min_value=1, max_value=99),
        aggregated_bins=st.integers(min_value=1, max_value=100),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_bin_boundaries_monotonic(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: bin boundaries are monotonically increasing."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        boundaries = compute_bin_boundaries(
            depth_levels, raw_levels, aggregated_bins, bin_strategy
        )

        for i in range(len(boundaries) - 1):
            self.assertLessEqual(boundaries[i], boundaries[i + 1])

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=2000),
        raw_levels=st.integers(min_value=1, max_value=99),
        aggregated_bins=st.integers(min_value=1, max_value=100),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_bin_boundaries_coverage(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: bin boundaries start at raw_levels and end at depth_levels."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        boundaries = compute_bin_boundaries(
            depth_levels, raw_levels, aggregated_bins, bin_strategy
        )

        self.assertEqual(int(boundaries[0]), raw_levels)
        self.assertEqual(int(boundaries[-1]), depth_levels)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=500),
        raw_levels=st.integers(min_value=10, max_value=50),
        aggregated_bins=st.integers(min_value=10, max_value=50),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_aggregate_depth_levels_output_shape(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: aggregate_depth_levels produces correct output shape."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        config = _make_hybrid_config(depth_levels, raw_levels, aggregated_bins, bin_strategy)

        prices = np.linspace(100, 99, depth_levels)
        quantities = np.abs(np.random.randn(depth_levels)) * 10

        hybrid_p, hybrid_q = aggregate_depth_levels(prices, quantities, config)

        expected_levels = raw_levels + aggregated_bins
        self.assertEqual(hybrid_p.shape, (expected_levels,))
        self.assertEqual(hybrid_q.shape, (expected_levels,))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=500),
        raw_levels=st.integers(min_value=10, max_value=50),
        aggregated_bins=st.integers(min_value=10, max_value=50),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_aggregate_snapshot_output_shape(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: aggregate_snapshot_to_hybrid produces (L, 4) shape."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        config = _make_hybrid_config(depth_levels, raw_levels, aggregated_bins, bin_strategy)

        bid_prices = np.linspace(100, 99, depth_levels)
        bid_qtys = np.abs(np.random.randn(depth_levels)) * 10
        ask_prices = np.linspace(100.01, 101, depth_levels)
        ask_qtys = np.abs(np.random.randn(depth_levels)) * 10

        result = aggregate_snapshot_to_hybrid(
            bid_prices, bid_qtys, ask_prices, ask_qtys, config
        )

        expected_levels = raw_levels + aggregated_bins
        self.assertEqual(result.shape, (expected_levels, 4))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=500),
        raw_levels=st.integers(min_value=10, max_value=50),
        aggregated_bins=st.integers(min_value=10, max_value=50),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_raw_levels_preserved(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: top raw_levels are preserved exactly."""
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        config = _make_hybrid_config(depth_levels, raw_levels, aggregated_bins, bin_strategy)

        prices = np.linspace(100, 99, depth_levels)
        quantities = np.abs(np.random.randn(depth_levels)) * 10

        hybrid_p, hybrid_q = aggregate_depth_levels(prices, quantities, config)

        np.testing.assert_allclose(hybrid_p[:raw_levels], prices[:raw_levels])
        np.testing.assert_allclose(hybrid_q[:raw_levels], quantities[:raw_levels])

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        depth_levels=st.integers(min_value=100, max_value=500),
        raw_levels=st.integers(min_value=10, max_value=50),
        aggregated_bins=st.integers(min_value=10, max_value=50),
        bin_strategy=st.sampled_from(["equal_width", "log_spaced"]),
    )
    def test_validation_passes_for_valid_config(
        self,
        depth_levels: int,
        raw_levels: int,
        aggregated_bins: int,
        bin_strategy: str,
    ) -> None:
        """Property: validate_hybrid_config succeeds for valid configs."""
        assume(raw_levels > 0)
        assume(aggregated_bins > 0)
        assume(raw_levels < depth_levels)
        assume(raw_levels + aggregated_bins <= depth_levels)

        config = _make_hybrid_config(depth_levels, raw_levels, aggregated_bins, bin_strategy)

        # Should not raise
        validate_hybrid_config(config)

    def test_validation_fails_for_invalid_strategy(self) -> None:
        """Validation should fail for unsupported bin_strategy."""
        config = _make_hybrid_config(1000, 50, 50, "invalid_strategy")
        with self.assertRaises(ValueError):
            validate_hybrid_config(config)

    def test_validation_fails_when_sum_exceeds_depth(self) -> None:
        """Validation should fail when raw_levels + aggregated_bins > depth_levels."""
        config = _make_hybrid_config(100, 60, 50, "equal_width")
        with self.assertRaises(ValueError):
            validate_hybrid_config(config)

    def test_validation_fails_when_raw_levels_exceeds_depth(self) -> None:
        """Validation should fail when raw_levels >= depth_levels."""
        config = _make_hybrid_config(100, 100, 10, "equal_width")
        with self.assertRaises(ValueError):
            validate_hybrid_config(config)

    def test_validation_fails_when_raw_levels_zero(self) -> None:
        """Validation should fail when raw_levels <= 0."""
        config = _make_hybrid_config(1000, 0, 50, "equal_width")
        with self.assertRaises(ValueError):
            validate_hybrid_config(config)

    def test_validation_fails_when_aggregated_bins_zero(self) -> None:
        """Validation should fail when aggregated_bins <= 0."""
        config = _make_hybrid_config(1000, 50, 0, "equal_width")
        with self.assertRaises(ValueError):
            validate_hybrid_config(config)


if __name__ == "__main__":
    unittest.main()
