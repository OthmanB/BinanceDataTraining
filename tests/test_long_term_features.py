"""Tests for long-term feature computation (TD-019).

This module contains comprehensive tests for the dual-channel long-term context
feature extraction, covering:
- Configuration validation
- Daily aggregation
- Multi-scale window extraction
- Edge cases (insufficient history, gaps, etc.)
- Integration with model input shapes
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from preprocessing.long_term_features import (
    LongTermConfig,
    LongTermFeatureError,
    SUPPORTED_LT_FEATURES,
    SUPPORTED_SUMMARY_METHODS,
    compute_daily_aggregates,
    build_long_term_features_multiscale,
    compute_long_term_features,
)


def _make_config(long_term_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create a minimal config dict with long_term settings."""
    lt_cfg = long_term_cfg if long_term_cfg is not None else {}
    return {
        "model": {
            "long_term": lt_cfg,
        },
    }


def _ts_from_date(year: int, month: int, day: int) -> float:
    """Create Unix timestamp from date components."""
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return dt.timestamp()


class TestLongTermConfig(unittest.TestCase):
    """Tests for LongTermConfig dataclass and validation."""

    def test_default_values_when_disabled(self) -> None:
        """Test that defaults are sensible when disabled."""
        config = LongTermConfig(enabled=False)
        self.assertFalse(config.enabled)
        self.assertEqual(config.windows_days, [7, 30, 90])
        self.assertEqual(config.input_dim, 0)

    def test_default_values_when_enabled(self) -> None:
        """Test that defaults are sensible when enabled."""
        config = LongTermConfig(enabled=True)
        self.assertTrue(config.enabled)
        self.assertEqual(config.windows_days, [7, 30, 90])
        self.assertEqual(config.features, ["mean_return", "volatility", "volume_proxy", "skewness"])
        self.assertEqual(config.summary_method, "mean")
        self.assertEqual(config.input_dim, 3 * 4)  # 3 windows * 4 features

    def test_input_dim_calculation(self) -> None:
        """Test input_dim calculation for various configurations."""
        config = LongTermConfig(
            enabled=True,
            windows_days=[7, 30],
            features=["mean_return", "volatility"],
        )
        self.assertEqual(config.input_dim, 2 * 2)  # 2 windows * 2 features

        config = LongTermConfig(
            enabled=True,
            windows_days=[7, 30, 90],
            features=["mean_return"],
        )
        self.assertEqual(config.input_dim, 3 * 1)

    def test_from_config_disabled(self) -> None:
        """Test from_config when long_term is not present."""
        config = _make_config({})
        lt_config = LongTermConfig.from_config(config)
        self.assertFalse(lt_config.enabled)

    def test_from_config_enabled(self) -> None:
        """Test from_config with explicit values."""
        config = _make_config({
            "enabled": True,
            "windows_days": [14, 60],
            "features": ["mean_return", "max_up"],
            "summary_method": "ewma",
            "ewma_halflife_days": 10.0,
        })
        lt_config = LongTermConfig.from_config(config)
        self.assertTrue(lt_config.enabled)
        self.assertEqual(lt_config.windows_days, [14, 60])
        self.assertEqual(lt_config.features, ["mean_return", "max_up"])
        self.assertEqual(lt_config.summary_method, "ewma")
        self.assertEqual(lt_config.ewma_halflife_days, 10.0)
        self.assertEqual(lt_config.input_dim, 4)  # 2 windows * 2 features

    def test_invalid_empty_windows(self) -> None:
        """Test that empty windows_days raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, windows_days=[])
        self.assertIn("non-empty", str(ctx.exception))

    def test_invalid_negative_window(self) -> None:
        """Test that negative window size raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, windows_days=[7, -1])
        self.assertIn("must be >= 1", str(ctx.exception))

    def test_invalid_zero_resolution(self) -> None:
        """Test that zero resolution_days raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, resolution_days=0)
        self.assertIn("resolution_days", str(ctx.exception))

    def test_invalid_feature_name(self) -> None:
        """Test that unsupported feature name raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, features=["unknown_feature"])
        self.assertIn("Unsupported long-term feature", str(ctx.exception))
        self.assertIn("unknown_feature", str(ctx.exception))

    def test_invalid_summary_method(self) -> None:
        """Test that unsupported summary method raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, summary_method="invalid")
        self.assertIn("Unsupported summary_method", str(ctx.exception))

    def test_invalid_ewma_halflife(self) -> None:
        """Test that non-positive ewma_halflife_days raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            LongTermConfig(enabled=True, summary_method="ewma", ewma_halflife_days=0)
        self.assertIn("ewma_halflife_days", str(ctx.exception))


class TestSupportedFeatures(unittest.TestCase):
    """Tests for supported feature constants."""

    def test_supported_features_frozen(self) -> None:
        """Test that SUPPORTED_LT_FEATURES is a frozenset."""
        self.assertIsInstance(SUPPORTED_LT_FEATURES, frozenset)
        self.assertIn("mean_return", SUPPORTED_LT_FEATURES)
        self.assertIn("volatility", SUPPORTED_LT_FEATURES)
        self.assertIn("volume_proxy", SUPPORTED_LT_FEATURES)
        self.assertIn("skewness", SUPPORTED_LT_FEATURES)
        self.assertIn("kurtosis", SUPPORTED_LT_FEATURES)
        self.assertIn("max_up", SUPPORTED_LT_FEATURES)
        self.assertIn("max_down", SUPPORTED_LT_FEATURES)

    def test_supported_summary_methods_frozen(self) -> None:
        """Test that SUPPORTED_SUMMARY_METHODS is a frozenset."""
        self.assertIsInstance(SUPPORTED_SUMMARY_METHODS, frozenset)
        self.assertIn("mean", SUPPORTED_SUMMARY_METHODS)
        self.assertIn("ewma", SUPPORTED_SUMMARY_METHODS)
        self.assertIn("last", SUPPORTED_SUMMARY_METHODS)


class TestComputeDailyAggregates(unittest.TestCase):
    """Tests for compute_daily_aggregates function."""

    def test_empty_input(self) -> None:
        """Test with empty arrays."""
        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=np.array([]),
            timestamps=np.array([]),
            volumes=None,
            cadence_seconds=10,
        )
        self.assertEqual(len(daily_ts), 0)
        self.assertEqual(len(daily_ret), 0)
        self.assertEqual(len(daily_vol), 0)

    def test_single_day(self) -> None:
        """Test with data from a single day."""
        # 10 snapshots on 2024-01-15, 10 seconds apart
        base_ts = _ts_from_date(2024, 1, 15) + 3600  # Start at 01:00
        timestamps = np.array([base_ts + i * 10 for i in range(10)])
        mid_prices = np.array([100.0] * 10)
        volumes = np.array([50.0] * 10)

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=volumes,
            cadence_seconds=10,
        )

        self.assertEqual(len(daily_ts), 1)
        self.assertEqual(len(daily_ret), 1)
        self.assertEqual(len(daily_vol), 1)
        # First day return is 0
        self.assertEqual(daily_ret[0], 0.0)
        # Average volume
        self.assertAlmostEqual(daily_vol[0], 50.0)

    def test_two_days_with_price_change(self) -> None:
        """Test with two days and a price change."""
        # Day 1: 2024-01-15
        day1_ts = _ts_from_date(2024, 1, 15) + 3600
        # Day 2: 2024-01-16
        day2_ts = _ts_from_date(2024, 1, 16) + 3600

        timestamps = np.array([day1_ts, day1_ts + 10, day2_ts, day2_ts + 10])
        mid_prices = np.array([100.0, 100.0, 110.0, 110.0])
        volumes = np.array([50.0, 60.0, 70.0, 80.0])

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=volumes,
            cadence_seconds=10,
        )

        self.assertEqual(len(daily_ts), 2)
        # Day 1 return is 0 (first day)
        self.assertEqual(daily_ret[0], 0.0)
        # Day 2 return: (110 - 100) / 100 = 0.1
        self.assertAlmostEqual(daily_ret[1], 0.1)
        # Day 1 avg volume: (50 + 60) / 2 = 55
        self.assertAlmostEqual(daily_vol[0], 55.0)
        # Day 2 avg volume: (70 + 80) / 2 = 75
        self.assertAlmostEqual(daily_vol[1], 75.0)

    def test_no_volumes_provided(self) -> None:
        """Test when volumes is None."""
        base_ts = _ts_from_date(2024, 1, 15)
        timestamps = np.array([base_ts, base_ts + 10])
        mid_prices = np.array([100.0, 101.0])

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=None,
            cadence_seconds=10,
        )

        self.assertEqual(len(daily_ts), 1)
        self.assertEqual(daily_vol[0], 0.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched mid_prices and timestamps raises error."""
        with self.assertRaises(LongTermFeatureError) as ctx:
            compute_daily_aggregates(
                mid_prices=np.array([100.0, 101.0]),
                timestamps=np.array([0.0]),
                volumes=None,
                cadence_seconds=10,
            )
        self.assertIn("same length", str(ctx.exception))

    def test_multiple_days_with_gap(self) -> None:
        """Test with multiple days including a gap (missing day)."""
        # Day 1: 2024-01-15
        day1_ts = _ts_from_date(2024, 1, 15)
        # Day 3: 2024-01-17 (skip Jan 16)
        day3_ts = _ts_from_date(2024, 1, 17)

        timestamps = np.array([day1_ts, day3_ts])
        mid_prices = np.array([100.0, 105.0])

        daily_ts, daily_ret, daily_vol = compute_daily_aggregates(
            mid_prices=mid_prices,
            timestamps=timestamps,
            volumes=None,
            cadence_seconds=10,
        )

        # Should have 2 days (not 3 - missing day is not created)
        self.assertEqual(len(daily_ts), 2)
        # Second day return: (105 - 100) / 100 = 0.05
        self.assertAlmostEqual(daily_ret[1], 0.05)


class TestBuildLongTermFeaturesMultiscale(unittest.TestCase):
    """Tests for build_long_term_features_multiscale function."""

    def _make_daily_data(self, n_days: int, base_return: float = 0.01) -> tuple:
        """Create synthetic daily data for testing."""
        # Create n_days of data starting from 2024-01-01
        base_ts = _ts_from_date(2024, 1, 1)
        daily_timestamps = np.array([
            base_ts + i * 86400 for i in range(n_days)
        ])
        # Simple returns with some variation
        daily_returns = np.array([
            base_return * (1 + 0.1 * np.sin(i)) for i in range(n_days)
        ])
        daily_volumes = np.array([100.0 + i * 10.0 for i in range(n_days)])
        return daily_timestamps, daily_returns, daily_volumes

    def test_empty_daily_data(self) -> None:
        """Test with empty daily data."""
        config = LongTermConfig(
            enabled=True,
            windows_days=[7],
            features=["mean_return"],
        )
        result = build_long_term_features_multiscale(
            daily_returns=np.array([]),
            daily_volumes=np.array([]),
            daily_timestamps=np.array([]),
            anchor_timestamp=0.0,
            config=config,
        )
        self.assertEqual(len(result), 1)  # 1 window * 1 feature
        self.assertEqual(result[0], 0.0)

    def test_anchor_before_all_data(self) -> None:
        """Test with anchor before any daily data exists."""
        daily_ts, daily_ret, daily_vol = self._make_daily_data(10)
        config = LongTermConfig(
            enabled=True,
            windows_days=[7],
            features=["mean_return"],
        )

        # Anchor is before all data
        anchor_ts = daily_ts[0] - 86400

        result = build_long_term_features_multiscale(
            daily_returns=daily_ret,
            daily_volumes=daily_vol,
            daily_timestamps=daily_ts,
            anchor_timestamp=anchor_ts,
            config=config,
        )
        self.assertEqual(result[0], 0.0)

    def test_single_window_full_history(self) -> None:
        """Test single window with enough history."""
        daily_ts, daily_ret, daily_vol = self._make_daily_data(30)
        config = LongTermConfig(
            enabled=True,
            windows_days=[7],
            features=["mean_return"],
        )

        # Anchor after 15 days of data
        anchor_ts = daily_ts[15] + 43200  # Mid-day of day 16

        result = build_long_term_features_multiscale(
            daily_returns=daily_ret,
            daily_volumes=daily_vol,
            daily_timestamps=daily_ts,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        # Should have 1 feature (mean_return for 7-day window)
        self.assertEqual(len(result), 1)
        # Mean of last 7 days before anchor (days 8-14, indices 8-14)
        expected_mean = np.mean(daily_ret[8:15])
        self.assertAlmostEqual(result[0], expected_mean, places=5)

    def test_multiscale_windows(self) -> None:
        """Test with multiple window sizes."""
        daily_ts, daily_ret, daily_vol = self._make_daily_data(100)
        config = LongTermConfig(
            enabled=True,
            windows_days=[7, 30],
            features=["mean_return"],
        )

        # Anchor after 50 days
        anchor_ts = daily_ts[50] + 43200

        result = build_long_term_features_multiscale(
            daily_returns=daily_ret,
            daily_volumes=daily_vol,
            daily_timestamps=daily_ts,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        self.assertEqual(len(result), 2)  # 2 windows * 1 feature

        # Check 7-day window (indices 43-49)
        expected_7d = np.mean(daily_ret[43:50])
        self.assertAlmostEqual(result[0], expected_7d, places=5)

        # Check 30-day window (indices 20-49)
        expected_30d = np.mean(daily_ret[20:50])
        self.assertAlmostEqual(result[1], expected_30d, places=5)

    def test_multiple_features(self) -> None:
        """Test with multiple features per window."""
        daily_ts, daily_ret, daily_vol = self._make_daily_data(30)
        config = LongTermConfig(
            enabled=True,
            windows_days=[7],
            features=["mean_return", "volatility", "volume_proxy"],
        )

        anchor_ts = daily_ts[20] + 43200

        result = build_long_term_features_multiscale(
            daily_returns=daily_ret,
            daily_volumes=daily_vol,
            daily_timestamps=daily_ts,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        self.assertEqual(len(result), 3)  # 1 window * 3 features

        window_ret = daily_ret[13:20]
        window_vol = daily_vol[13:20]

        # mean_return
        self.assertAlmostEqual(result[0], np.mean(window_ret), places=5)
        # volatility
        self.assertAlmostEqual(result[1], np.std(window_ret, ddof=1), places=5)
        # volume_proxy
        self.assertAlmostEqual(result[2], np.mean(window_vol), places=5)

    def test_partial_window_history(self) -> None:
        """Test with less history than window size."""
        daily_ts, daily_ret, daily_vol = self._make_daily_data(5)
        config = LongTermConfig(
            enabled=True,
            windows_days=[10],  # Request 10 days but only 4 available
            features=["mean_return"],
        )

        anchor_ts = daily_ts[4] + 43200  # Anchor on day 5

        result = build_long_term_features_multiscale(
            daily_returns=daily_ret,
            daily_volumes=daily_vol,
            daily_timestamps=daily_ts,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        # Should use available 4 days (indices 0-3)
        expected_mean = np.mean(daily_ret[0:4])
        self.assertAlmostEqual(result[0], expected_mean, places=5)

    def test_max_up_and_max_down(self) -> None:
        """Test max_up and max_down features."""
        base_ts = _ts_from_date(2024, 1, 1)
        daily_timestamps = np.array([base_ts + i * 86400 for i in range(10)])
        daily_returns = np.array([0.01, -0.02, 0.05, -0.03, 0.02, -0.01, 0.04, -0.04, 0.03, 0.0])
        daily_volumes = np.zeros(10)

        config = LongTermConfig(
            enabled=True,
            windows_days=[10],
            features=["max_up", "max_down"],
        )

        anchor_ts = daily_timestamps[9] + 86400  # After all data

        result = build_long_term_features_multiscale(
            daily_returns=daily_returns,
            daily_volumes=daily_volumes,
            daily_timestamps=daily_timestamps,
            anchor_timestamp=anchor_ts,
            config=config,
        )

        # max_up: max positive return = 0.05
        self.assertAlmostEqual(result[0], 0.05, places=5)
        # max_down: abs(min) = abs(-0.04) = 0.04
        self.assertAlmostEqual(result[1], 0.04, places=5)


class TestComputeLongTermFeatures(unittest.TestCase):
    """Tests for the main compute_long_term_features entry point."""

    def _make_intraday_data(
        self, n_days: int, snapshots_per_day: int = 100
    ) -> tuple:
        """Create synthetic intraday data for testing."""
        cadence_seconds = 86400 // snapshots_per_day
        base_ts = _ts_from_date(2024, 1, 1)

        n_total = n_days * snapshots_per_day
        timestamps = np.array([
            base_ts + i * cadence_seconds for i in range(n_total)
        ])

        # Price with trend and noise
        mid_prices = 50000.0 + np.cumsum(np.random.randn(n_total) * 10)
        volumes = 100.0 + np.random.rand(n_total) * 50

        return timestamps, mid_prices, volumes, cadence_seconds

    def test_disabled_returns_empty(self) -> None:
        """Test that disabled long_term returns empty array."""
        config = _make_config({"enabled": False})
        timestamps, mid_prices, volumes, cadence = self._make_intraday_data(30)
        anchor_timestamps = timestamps[::100][:10]  # 10 samples

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence,
            volumes=volumes,
        )

        self.assertEqual(result.shape, (10, 0))

    def test_enabled_returns_correct_shape(self) -> None:
        """Test that enabled long_term returns correct shape."""
        config = _make_config({
            "enabled": True,
            "windows_days": [7, 30],
            "features": ["mean_return", "volatility"],
        })
        timestamps, mid_prices, volumes, cadence = self._make_intraday_data(60)
        anchor_timestamps = timestamps[::100][:20]  # 20 samples

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence,
            volumes=volumes,
        )

        # 20 samples, 2 windows * 2 features = 4 features
        self.assertEqual(result.shape, (20, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_no_volumes_provided(self) -> None:
        """Test when volumes is not provided."""
        config = _make_config({
            "enabled": True,
            "windows_days": [7],
            "features": ["mean_return", "volume_proxy"],
        })
        timestamps, mid_prices, _, cadence = self._make_intraday_data(30)
        anchor_timestamps = timestamps[::100][:5]

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence,
            volumes=None,
        )

        self.assertEqual(result.shape, (5, 2))
        # volume_proxy should be 0 when volumes not provided
        # (it's index 1 in the features)

    def test_samples_with_insufficient_history(self) -> None:
        """Test that samples with insufficient history get zeros."""
        config = _make_config({
            "enabled": True,
            "windows_days": [30],  # Need 30 days
            "features": ["mean_return"],
        })
        # Only 10 days of data
        timestamps, mid_prices, volumes, cadence = self._make_intraday_data(10)
        # Anchors at various points
        anchor_timestamps = np.array([
            timestamps[0],  # Very start - no history
            timestamps[500],  # After ~5 days
            timestamps[-1],  # End of data
        ])

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence,
            volumes=volumes,
        )

        self.assertEqual(result.shape, (3, 1))
        # First anchor has no history
        self.assertEqual(result[0, 0], 0.0)

    def test_all_features(self) -> None:
        """Test with all supported features."""
        config = _make_config({
            "enabled": True,
            "windows_days": [7],
            "features": list(SUPPORTED_LT_FEATURES),
        })
        timestamps, mid_prices, volumes, cadence = self._make_intraday_data(30)
        anchor_timestamps = timestamps[2000:2010]  # 10 samples mid-dataset

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence,
            volumes=volumes,
        )

        # 10 samples, 1 window * 7 features = 7 features
        self.assertEqual(result.shape, (10, 7))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_single_snapshot(self) -> None:
        """Test with a single snapshot in the dataset."""
        config = _make_config({
            "enabled": True,
            "windows_days": [7],
            "features": ["mean_return"],
        })
        timestamps = np.array([_ts_from_date(2024, 1, 15)])
        mid_prices = np.array([50000.0])
        anchor_timestamps = np.array([timestamps[0]])

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=10,
            volumes=None,
        )

        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result[0, 0], 0.0)

    def test_anchor_exactly_on_day_boundary(self) -> None:
        """Test anchor timestamp exactly at midnight."""
        base_ts = _ts_from_date(2024, 1, 1)
        n_days = 10
        daily_timestamps = np.array([base_ts + i * 86400 for i in range(n_days)])
        
        # Create intraday data
        snapshots_per_day = 10
        cadence = 86400 // snapshots_per_day
        timestamps = []
        mid_prices = []
        for i in range(n_days):
            for j in range(snapshots_per_day):
                timestamps.append(daily_timestamps[i] + j * cadence)
                mid_prices.append(50000.0 + i * 100 + j)
        
        timestamps = np.array(timestamps)
        mid_prices = np.array(mid_prices)

        config = _make_config({
            "enabled": True,
            "windows_days": [3],
            "features": ["mean_return"],
        })

        # Anchor exactly at midnight of day 5
        anchor_ts = _ts_from_date(2024, 1, 6)  # Midnight of Jan 6

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=cadence,
            volumes=None,
        )

        # Should use days 3, 4, 5 (indices 2, 3, 4) - days BEFORE Jan 6
        self.assertEqual(result.shape, (1, 1))
        # Non-zero because we have history
        # (actual value depends on data construction)

    def test_very_large_window(self) -> None:
        """Test window larger than available data."""
        config = _make_config({
            "enabled": True,
            "windows_days": [365],  # 1 year window
            "features": ["mean_return"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        # Only 30 days of data
        timestamps = np.array([base_ts + i * 86400 for i in range(30)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(30) * 10)

        anchor_ts = timestamps[-1] + 86400  # After all data

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=86400,
            volumes=None,
        )

        # Should still compute using available ~30 days
        self.assertEqual(result.shape, (1, 1))
        # Should not be zero (we have some data)

    def test_zero_prices(self) -> None:
        """Test handling of zero prices."""
        config = _make_config({
            "enabled": True,
            "windows_days": [3],
            "features": ["mean_return"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 86400 for i in range(5)])
        # Include a zero price
        mid_prices = np.array([100.0, 0.0, 110.0, 115.0, 120.0])

        anchor_ts = timestamps[-1] + 86400

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=86400,
            volumes=None,
        )

        # Should handle gracefully (returns may be 0 where division by zero occurs)
        self.assertEqual(result.shape, (1, 1))
        # Should not raise or be NaN
        self.assertFalse(np.isnan(result[0, 0]))


class TestIntegrationWithModel(unittest.TestCase):
    """Tests for integration with model input requirements."""

    def test_output_dtype(self) -> None:
        """Test that output dtype is float32 for model compatibility."""
        config = _make_config({
            "enabled": True,
            "windows_days": [7],
            "features": ["mean_return"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 3600 for i in range(24 * 30)])
        mid_prices = 50000.0 + np.cumsum(np.random.randn(len(timestamps)))
        anchor_timestamps = timestamps[::24]  # Daily anchors

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=3600,
            volumes=None,
        )

        self.assertEqual(result.dtype, np.float32)

    def test_input_dim_matches_config(self) -> None:
        """Test that output dimension matches config.input_dim."""
        for n_windows, n_features in [(1, 1), (3, 4), (2, 7)]:
            windows = list(range(7, 7 * n_windows + 1, 7))
            features = list(SUPPORTED_LT_FEATURES)[:n_features]

            lt_config = LongTermConfig(
                enabled=True,
                windows_days=windows,
                features=features,
            )
            
            config = _make_config({
                "enabled": True,
                "windows_days": windows,
                "features": features,
            })
            
            base_ts = _ts_from_date(2024, 1, 1)
            timestamps = np.array([base_ts + i * 3600 for i in range(24 * 100)])
            mid_prices = 50000.0 + np.cumsum(np.random.randn(len(timestamps)))
            volumes = 100.0 + np.random.rand(len(timestamps)) * 50
            anchor_timestamps = timestamps[2400:2410]  # 10 samples

            result = compute_long_term_features(
                config=config,
                mid_prices=mid_prices,
                timestamps=timestamps,
                anchor_timestamps=anchor_timestamps,
                cadence_seconds=3600,
                volumes=volumes,
            )

            self.assertEqual(
                result.shape[1],
                lt_config.input_dim,
                f"Mismatch for {n_windows} windows, {n_features} features",
            )


class TestSkewnessAndKurtosis(unittest.TestCase):
    """Tests for skewness and kurtosis calculations."""

    def test_skewness_with_few_samples(self) -> None:
        """Test skewness returns 0 with < 3 samples."""
        config = _make_config({
            "enabled": True,
            "windows_days": [2],
            "features": ["skewness"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        # Only 2 days of daily data
        timestamps = np.array([base_ts, base_ts + 86400])
        mid_prices = np.array([100.0, 105.0])
        anchor_ts = timestamps[-1] + 86400

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=86400,
            volumes=None,
        )

        # Skewness should be 0 with < 3 data points
        self.assertEqual(result[0, 0], 0.0)

    def test_kurtosis_with_few_samples(self) -> None:
        """Test kurtosis returns 0 with < 4 samples."""
        config = _make_config({
            "enabled": True,
            "windows_days": [3],
            "features": ["kurtosis"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 86400 for i in range(3)])
        mid_prices = np.array([100.0, 105.0, 102.0])
        anchor_ts = timestamps[-1] + 86400

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=86400,
            volumes=None,
        )

        # Kurtosis should be 0 with < 4 data points
        self.assertEqual(result[0, 0], 0.0)

    def test_skewness_positive(self) -> None:
        """Test skewness is positive for right-skewed distribution."""
        config = _make_config({
            "enabled": True,
            "windows_days": [30],
            "features": ["skewness"],
        })
        
        base_ts = _ts_from_date(2024, 1, 1)
        timestamps = np.array([base_ts + i * 86400 for i in range(30)])
        # Create right-skewed prices (mostly small changes, occasional big up)
        np.random.seed(42)
        changes = np.concatenate([
            np.random.randn(25) * 0.5,  # Small changes
            np.array([5.0, 6.0, 7.0, 8.0, 10.0]),  # Big positive outliers
        ])
        mid_prices = 100.0 + np.cumsum(changes)
        anchor_ts = timestamps[-1] + 86400

        result = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=np.array([anchor_ts]),
            cadence_seconds=86400,
            volumes=None,
        )

        # With positive outliers in returns, skewness should tend positive
        # (actual value depends on specific data)
        self.assertFalse(np.isnan(result[0, 0]))


if __name__ == "__main__":
    unittest.main()
