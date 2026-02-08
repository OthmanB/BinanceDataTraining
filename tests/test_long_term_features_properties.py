"""Property-based tests for long-term feature computation (TD-019).

These tests verify invariants that should hold for all valid inputs,
catching edge cases that unit tests might miss.
"""

import os
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

# Reduce hypothesis examples in CI for faster test runs
_CI_MODE = os.environ.get("CI", "").lower() in ("true", "1", "yes")
_MAX_EXAMPLES = 10 if _CI_MODE else 50

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from preprocessing.long_term_features import (
    SUPPORTED_LT_FEATURES,
    SUPPORTED_SUMMARY_METHODS,
    LongTermConfig,
    LongTermFeatureError,
    build_long_term_features_multiscale,
    compute_daily_aggregates,
    compute_long_term_features,
)


def _make_config(
    enabled: bool = True,
    windows_days: List[int] | None = None,
    features: List[str] | None = None,
    summary_method: str = "mean",
) -> Dict[str, Any]:
    """Create a config dict for testing."""
    return {
        "model": {
            "long_term": {
                "enabled": enabled,
                "windows_days": windows_days or [7, 30, 90],
                "features": features or ["mean_return", "volatility"],
                "summary_method": summary_method,
                "resolution_days": 1,
                "ewma_halflife_days": 7.0,
                "input_dim": None,
                "architecture": {"conv1d": {"activation": "relu", "layers": []}, "dense": {"layers": [{"units": 32, "dropout": 0.2}]}},
            }
        }
    }


def _ts_from_date(year: int, month: int, day: int) -> float:
    """Create Unix timestamp from date components."""
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return dt.timestamp()


# Strategies for generating test data
if HYPOTHESIS_AVAILABLE:
    # Strategy for valid window sizes
    valid_windows = st.lists(
        st.integers(min_value=1, max_value=365),
        min_size=1,
        max_size=5,
        unique=True,
    )

    # Strategy for valid feature names
    valid_features = st.lists(
        st.sampled_from(sorted(SUPPORTED_LT_FEATURES)),
        min_size=1,
        max_size=len(SUPPORTED_LT_FEATURES),
        unique=True,
    )

    # Strategy for valid summary methods
    valid_summary_method = st.sampled_from(sorted(SUPPORTED_SUMMARY_METHODS))

    # Strategy for generating time series data
    def price_series(n_samples: int):
        """Generate a synthetic price series."""
        return arrays(
            dtype=np.float64,
            shape=(n_samples,),
            elements=st.floats(
                min_value=100.0,
                max_value=100000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestLongTermConfigProperties(unittest.TestCase):
    """Property-based tests for LongTermConfig."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        windows=valid_windows,
        features=valid_features,
    )
    def test_input_dim_calculation(
        self,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: input_dim = len(windows) * len(features)."""
        config = LongTermConfig(
            enabled=True,
            windows_days=windows,
            features=features,
        )
        expected_dim = len(windows) * len(features)
        self.assertEqual(config.input_dim, expected_dim)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        windows=valid_windows,
        features=valid_features,
        summary_method=valid_summary_method,
    )
    def test_valid_config_does_not_raise(
        self,
        windows: List[int],
        features: List[str],
        summary_method: str,
    ) -> None:
        """Property: valid configuration never raises."""
        ewma_halflife = 7.0 if summary_method == "ewma" else 1.0
        try:
            config = LongTermConfig(
                enabled=True,
                windows_days=windows,
                features=features,
                summary_method=summary_method,
                ewma_halflife_days=ewma_halflife,
            )
            # Should succeed
            self.assertTrue(config.enabled)
        except LongTermFeatureError:
            self.fail("Valid config raised LongTermFeatureError")

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        windows=valid_windows,
        features=valid_features,
    )
    def test_disabled_input_dim_is_zero(
        self,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: disabled config always has input_dim = 0."""
        config = LongTermConfig(
            enabled=False,
            windows_days=windows,
            features=features,
        )
        self.assertEqual(config.input_dim, 0)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_windows=st.integers(min_value=1, max_value=10))
    def test_from_config_parses_correctly(self, n_windows: int) -> None:
        """Property: from_config correctly parses window count."""
        windows = list(range(7, 7 * n_windows + 1, 7))
        config_dict = _make_config(enabled=True, windows_days=windows)
        lt_config = LongTermConfig.from_config(config_dict)
        self.assertEqual(len(lt_config.windows_days), n_windows)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestComputeDailyAggregatesProperties(unittest.TestCase):
    """Property-based tests for compute_daily_aggregates."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_days=st.integers(min_value=1, max_value=100),
        snapshots_per_day=st.integers(min_value=1, max_value=100),
    )
    def test_output_shape_consistency(
        self,
        n_days: int,
        snapshots_per_day: int,
    ) -> None:
        """Property: all outputs have the same length."""
        # Generate synthetic data
        base_ts = _ts_from_date(2024, 1, 1)
        cadence = 86400 // snapshots_per_day

        n_total = n_days * snapshots_per_day
        timestamps = np.array([base_ts + i * cadence for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)
        volumes = 100.0 + np.random.rand(n_total) * 50

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=volumes,
            cadence_seconds=cadence,
        )

        # All outputs should have same length
        self.assertEqual(len(daily_ts), len(daily_ret))
        self.assertEqual(len(daily_ts), len(daily_vol))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_days=st.integers(min_value=1, max_value=100))
    def test_first_return_is_zero(self, n_days: int) -> None:
        """Property: first daily return is always 0."""
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_days) * 100)

        _, daily_ret, _ = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=None,
            cadence_seconds=86400,
        )

        if len(daily_ret) > 0:
            self.assertEqual(daily_ret[0], 0.0)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_days=st.integers(min_value=2, max_value=50))
    def test_output_is_finite(self, n_days: int) -> None:
        """Property: all outputs are finite (no NaN/Inf)."""
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_days) * 100)
        volumes = 100.0 + np.random.rand(n_days) * 50

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=volumes,
            cadence_seconds=86400,
        )

        self.assertTrue(np.all(np.isfinite(daily_ts)))
        self.assertTrue(np.all(np.isfinite(daily_ret)))
        self.assertTrue(np.all(np.isfinite(daily_vol)))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_days=st.integers(min_value=1, max_value=100))
    def test_output_length_bounded_by_unique_days(self, n_days: int) -> None:
        """Property: output length <= number of unique days in input."""
        base_ts = _ts_from_date(2024, 1, 1)
        # Multiple snapshots per day
        snapshots_per_day = 10
        cadence = 86400 // snapshots_per_day
        n_total = n_days * snapshots_per_day

        timestamps = np.array([base_ts + i * cadence for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)

        daily_ts, _, _ = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=None,
            cadence_seconds=cadence,
        )

        self.assertLessEqual(len(daily_ts), n_days)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestBuildLongTermFeaturesProperties(unittest.TestCase):
    """Property-based tests for build_long_term_features_multiscale."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_days=st.integers(min_value=10, max_value=200),
        windows=valid_windows,
        features=valid_features,
    )
    def test_output_shape_matches_config(
        self,
        n_days: int,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: output shape = n_windows * n_features."""
        config = LongTermConfig(
            enabled=True,
            windows_days=windows,
            features=features,
        )

        # Generate daily data
        base_ts = _ts_from_date(2024, 1, 1)
        daily_timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        daily_returns = np.random.randn(n_days) * 0.01
        daily_volumes = 100.0 + np.random.rand(n_days) * 50

        # Anchor at the end
        anchor_ts = daily_timestamps[-1] + 43200

        result = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        expected_len = len(windows) * len(features)
        self.assertEqual(len(result), expected_len)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_days=st.integers(min_value=10, max_value=100),
        windows=valid_windows,
        features=valid_features,
    )
    def test_output_is_finite(
        self,
        n_days: int,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: output contains no NaN or Inf values."""
        config = LongTermConfig(
            enabled=True,
            windows_days=windows,
            features=features,
        )

        base_ts = _ts_from_date(2024, 1, 1)
        daily_timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        daily_returns = np.random.randn(n_days) * 0.01
        daily_volumes = 100.0 + np.random.rand(n_days) * 50

        anchor_ts = daily_timestamps[-1] + 43200

        result = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        self.assertTrue(np.all(np.isfinite(result)))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_days=st.integers(min_value=10, max_value=100),
        windows=valid_windows,
        features=valid_features,
    )
    def test_output_is_deterministic(
        self,
        n_days: int,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: same inputs produce same outputs."""
        config = LongTermConfig(
            enabled=True,
            windows_days=windows,
            features=features,
        )

        base_ts = _ts_from_date(2024, 1, 1)
        daily_timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        daily_returns = np.random.randn(n_days) * 0.01
        daily_volumes = 100.0 + np.random.rand(n_days) * 50
        anchor_ts = daily_timestamps[-1] + 43200

        result1 = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        result2 = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        np.testing.assert_array_equal(result1, result2)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestComputeLongTermFeaturesProperties(unittest.TestCase):
    """Property-based tests for compute_long_term_features entry point."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=1, max_value=50),
        windows=valid_windows,
        features=valid_features,
    )
    def test_output_shape_matches_samples_and_config(
        self,
        n_samples: int,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: output shape = (n_samples, n_windows * n_features)."""
        config = _make_config(enabled=True, windows_days=windows, features=features)

        # Generate intraday data
        base_ts = _ts_from_date(2024, 1, 1)
        n_total = 1000
        timestamps = np.array([base_ts + i * 3600 for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)

        # Sample anchor timestamps
        anchor_indices = np.linspace(100, n_total - 1, n_samples, dtype=int)
        anchor_timestamps = timestamps[anchor_indices]

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=3600,
        )

        expected_dim = len(windows) * len(features)
        self.assertEqual(result.shape, (n_samples, expected_dim))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_samples=st.integers(min_value=1, max_value=50))
    def test_output_dtype_is_float32(self, n_samples: int) -> None:
        """Property: output dtype is always float32."""
        config = _make_config(enabled=True)

        base_ts = _ts_from_date(2024, 1, 1)
        n_total = 1000
        timestamps = np.array([base_ts + i * 3600 for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)

        anchor_indices = np.linspace(100, n_total - 1, n_samples, dtype=int)
        anchor_timestamps = timestamps[anchor_indices]

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=3600,
        )

        self.assertEqual(result.dtype, np.float32)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(n_samples=st.integers(min_value=1, max_value=50))
    def test_disabled_returns_zero_dim(self, n_samples: int) -> None:
        """Property: disabled config returns shape (n_samples, 0)."""
        config = _make_config(enabled=False)

        base_ts = _ts_from_date(2024, 1, 1)
        n_total = 100
        timestamps = np.array([base_ts + i * 3600 for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)
        anchor_timestamps = timestamps[:n_samples]

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=3600,
        )

        self.assertEqual(result.shape, (n_samples, 0))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=1, max_value=30),
        windows=valid_windows,
        features=valid_features,
    )
    def test_output_is_finite(
        self,
        n_samples: int,
        windows: List[int],
        features: List[str],
    ) -> None:
        """Property: output contains no NaN or Inf values."""
        config = _make_config(enabled=True, windows_days=windows, features=features)

        base_ts = _ts_from_date(2024, 1, 1)
        n_total = 2000
        timestamps = np.array([base_ts + i * 3600 for i in range(n_total)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)
        # Ensure positive prices
        mid_prices = np.abs(mid_prices) + 1.0

        anchor_indices = np.linspace(100, n_total - 1, n_samples, dtype=int)
        anchor_timestamps = timestamps[anchor_indices]

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=3600,
        )

        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
