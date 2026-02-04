"""Tests for the backtesting module.

This test suite covers:
- BacktestConfig validation
- Signal generation strategies (net_intensity and threshold)
- Trade simulation
- Equity curve building
- Risk metrics computation
- Run backtest orchestration
- MLflow logging (mocked)
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

# Reduce hypothesis examples in CI for faster test runs
_MAX_EXAMPLES = 10 if os.environ.get("CI") else 50

try:
    from hypothesis import given, settings, HealthCheck, assume
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    HealthCheck = None  # type: ignore[assignment,misc]
    assume = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]
    arrays = None  # type: ignore[assignment]

from evaluation.backtesting import (
    SIGNAL_STRATEGIES,
    POSITION_SIZING_METHODS,
    BacktestConfig,
    BacktestError,
    BacktestResult,
    Trade,
    build_equity_curve,
    compute_risk_metrics,
    generate_signals,
    generate_signals_net_intensity,
    generate_signals_threshold,
    log_backtest_to_mlflow,
    run_backtest,
    simulate_trades,
)


class TestBacktestConfig(unittest.TestCase):
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = BacktestConfig()
        self.assertEqual(config.initial_capital, 10000.0)
        self.assertEqual(config.transaction_cost_pct, 0.001)
        self.assertEqual(config.signal_strategy, "net_intensity")
        self.assertEqual(config.signal_threshold, 0.6)
        self.assertEqual(config.intensity_threshold, 1)
        self.assertEqual(config.position_sizing, "equal")
        self.assertEqual(config.max_position_pct, 1.0)

    def test_custom_values(self) -> None:
        """Test that custom values are accepted."""
        config = BacktestConfig(
            initial_capital=50000.0,
            transaction_cost_pct=0.002,
            signal_strategy="threshold",
            signal_threshold=0.7,
            intensity_threshold=2,
            position_sizing="confidence",
            max_position_pct=0.5,
        )
        self.assertEqual(config.initial_capital, 50000.0)
        self.assertEqual(config.signal_strategy, "threshold")
        self.assertEqual(config.position_sizing, "confidence")

    def test_invalid_initial_capital(self) -> None:
        """Test that non-positive initial capital raises error."""
        with self.assertRaises(BacktestError) as ctx:
            BacktestConfig(initial_capital=0)
        self.assertIn("initial_capital must be positive", str(ctx.exception))

        with self.assertRaises(BacktestError):
            BacktestConfig(initial_capital=-100)

    def test_invalid_signal_strategy(self) -> None:
        """Test that invalid signal strategy raises error."""
        with self.assertRaises(BacktestError) as ctx:
            BacktestConfig(signal_strategy="invalid")
        self.assertIn("signal_strategy must be one of", str(ctx.exception))

    def test_invalid_signal_threshold(self) -> None:
        """Test that invalid signal threshold raises error."""
        with self.assertRaises(BacktestError):
            BacktestConfig(signal_threshold=0)
        with self.assertRaises(BacktestError):
            BacktestConfig(signal_threshold=1.5)

    def test_invalid_intensity_threshold(self) -> None:
        """Test that invalid intensity threshold raises error."""
        with self.assertRaises(BacktestError):
            BacktestConfig(intensity_threshold=-1)
        with self.assertRaises(BacktestError):
            BacktestConfig(intensity_threshold=4)

    def test_invalid_position_sizing(self) -> None:
        """Test that invalid position sizing raises error."""
        with self.assertRaises(BacktestError) as ctx:
            BacktestConfig(position_sizing="invalid")
        self.assertIn("position_sizing must be one of", str(ctx.exception))


class TestTrade(unittest.TestCase):
    """Tests for Trade dataclass."""

    def test_trade_creation(self) -> None:
        """Test basic trade creation."""
        trade = Trade(
            entry_idx=0,
            exit_idx=10,
            direction="long",
            entry_price=100.0,
            exit_price=110.0,
            pnl_pct=9.8,
            signal_confidence=0.7,
        )
        self.assertEqual(trade.entry_idx, 0)
        self.assertEqual(trade.exit_idx, 10)
        self.assertEqual(trade.direction, "long")
        self.assertAlmostEqual(trade.pnl_pct, 9.8)

    def test_trade_to_dict(self) -> None:
        """Test trade serialization."""
        trade = Trade(
            entry_idx=5,
            exit_idx=15,
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 12, 30, 0),
            direction="short",
            entry_price=50000.0,
            exit_price=49000.0,
            pnl_pct=1.8,
            signal_confidence=0.65,
        )
        d = trade.to_dict()
        self.assertEqual(d["entry_idx"], 5)
        self.assertEqual(d["direction"], "short")
        self.assertIn("2024-01-01", d["entry_time"])


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestSignalProperties(unittest.TestCase):
    """Property-based tests for signal generation invariants."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_net_intensity_signals_within_bounds(self, data: Any) -> None:
        n_samples = data.draw(st.integers(min_value=1, max_value=50))
        n_classes = data.draw(st.integers(min_value=2, max_value=6))
        intensity_threshold = data.draw(st.integers(min_value=0, max_value=n_classes - 1))

        raw_up = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_classes),
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        raw_down = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_classes),
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )

        up_sums = raw_up.sum(axis=1)
        down_sums = raw_down.sum(axis=1)
        assume(np.all(up_sums > 0.0))
        assume(np.all(down_sums > 0.0))

        y_prob_up = raw_up / up_sums[:, None]
        y_prob_down = raw_down / down_sums[:, None]

        signals, confidences = generate_signals_net_intensity(
            y_prob_up,
            y_prob_down,
            intensity_threshold=intensity_threshold,
        )

        self.assertEqual(signals.shape[0], n_samples)
        self.assertTrue(np.all(np.isin(signals, [-1, 0, 1])))
        self.assertTrue(np.all(confidences >= 0.0))
        self.assertTrue(np.all(confidences <= 1.0 + 1e-6))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_threshold_signals_within_bounds(self, data: Any) -> None:
        n_samples = data.draw(st.integers(min_value=1, max_value=50))
        n_classes = data.draw(st.integers(min_value=2, max_value=6))
        intensity_threshold = data.draw(st.integers(min_value=0, max_value=n_classes - 1))
        threshold = data.draw(st.floats(min_value=0.01, max_value=0.99))

        raw_up = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_classes),
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        raw_down = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_classes),
                elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )

        up_sums = raw_up.sum(axis=1)
        down_sums = raw_down.sum(axis=1)
        assume(np.all(up_sums > 0.0))
        assume(np.all(down_sums > 0.0))

        y_prob_up = raw_up / up_sums[:, None]
        y_prob_down = raw_down / down_sums[:, None]

        signals, confidences = generate_signals_threshold(
            y_prob_up,
            y_prob_down,
            threshold=threshold,
            intensity_threshold=intensity_threshold,
        )

        self.assertEqual(signals.shape[0], n_samples)
        self.assertTrue(np.all(np.isin(signals, [-1, 0, 1])))
        self.assertTrue(np.all(confidences >= 0.0))
        self.assertTrue(np.all(confidences <= 1.0 + 1e-6))


class TestBacktestResult(unittest.TestCase):
    """Tests for BacktestResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty result creation."""
        result = BacktestResult()
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(len(result.equity_curve), 0)

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        config = BacktestConfig()
        result = BacktestResult(
            trades=[Trade(entry_idx=0, exit_idx=10, pnl_pct=5.0)],
            equity_curve=np.array([10000, 10500]),
            metrics={"total_return_pct": 5.0},
            config=config,
        )
        d = result.to_dict()
        self.assertEqual(d["num_trades"], 1)
        self.assertEqual(d["metrics"]["total_return_pct"], 5.0)
        self.assertEqual(d["config"]["initial_capital"], 10000.0)


class TestGenerateSignalsNetIntensity(unittest.TestCase):
    """Tests for generate_signals_net_intensity function."""

    def test_empty_input(self) -> None:
        """Test with empty arrays."""
        y_up = np.array([]).reshape(0, 4)
        y_down = np.array([]).reshape(0, 4)
        signals, confidences = generate_signals_net_intensity(y_up, y_down)
        self.assertEqual(len(signals), 0)
        self.assertEqual(len(confidences), 0)

    def test_strong_up_signal(self) -> None:
        """Test that strong up probability generates long signal."""
        # Class 0=low, 1-3=significant. P(up significant) = 0.8, P(down significant) = 0.2
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        y_down = np.array([[0.8, 0.1, 0.05, 0.05]])  # sum(1:) = 0.2
        signals, confidences = generate_signals_net_intensity(y_up, y_down, intensity_threshold=1)
        self.assertEqual(signals[0], +1)  # Long
        self.assertAlmostEqual(confidences[0], 0.6, places=5)

    def test_strong_down_signal(self) -> None:
        """Test that strong down probability generates short signal."""
        y_up = np.array([[0.8, 0.1, 0.05, 0.05]])  # sum(1:) = 0.2
        y_down = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        signals, confidences = generate_signals_net_intensity(y_up, y_down, intensity_threshold=1)
        self.assertEqual(signals[0], -1)  # Short
        self.assertAlmostEqual(confidences[0], 0.6, places=5)

    def test_equal_probabilities_hold(self) -> None:
        """Test that equal probabilities generate hold signal."""
        y_up = np.array([[0.5, 0.2, 0.2, 0.1]])  # sum(1:) = 0.5
        y_down = np.array([[0.5, 0.2, 0.2, 0.1]])  # sum(1:) = 0.5
        signals, confidences = generate_signals_net_intensity(y_up, y_down, intensity_threshold=1)
        self.assertEqual(signals[0], 0)  # Hold
        self.assertAlmostEqual(confidences[0], 0.0, places=5)

    def test_multiple_samples(self) -> None:
        """Test with multiple samples."""
        y_up = np.array([
            [0.2, 0.3, 0.3, 0.2],  # Up dominant
            [0.8, 0.1, 0.05, 0.05],  # Down dominant
            [0.5, 0.25, 0.15, 0.1],  # Equal
        ])
        y_down = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.2, 0.3, 0.3, 0.2],
            [0.5, 0.25, 0.15, 0.1],
        ])
        signals, _ = generate_signals_net_intensity(y_up, y_down, intensity_threshold=1)
        self.assertEqual(signals[0], +1)  # Long
        self.assertEqual(signals[1], -1)  # Short
        self.assertEqual(signals[2], 0)   # Hold

    def test_different_intensity_thresholds(self) -> None:
        """Test with different intensity thresholds."""
        # All prob in class 1, none in 2+
        y_up = np.array([[0.3, 0.7, 0.0, 0.0]])  # sum(1:) = 0.7, sum(2:) = 0.0
        y_down = np.array([[0.7, 0.3, 0.0, 0.0]])  # sum(1:) = 0.3, sum(2:) = 0.0

        # With threshold 1, up wins
        signals1, _ = generate_signals_net_intensity(y_up, y_down, intensity_threshold=1)
        self.assertEqual(signals1[0], +1)

        # With threshold 2, equal (both 0)
        signals2, _ = generate_signals_net_intensity(y_up, y_down, intensity_threshold=2)
        self.assertEqual(signals2[0], 0)

    def test_invalid_input_shapes(self) -> None:
        """Test that invalid shapes raise errors."""
        with self.assertRaises(BacktestError):
            generate_signals_net_intensity(np.array([0.5]), np.array([0.5]))

        with self.assertRaises(BacktestError):
            generate_signals_net_intensity(
                np.array([[0.5, 0.5]]),
                np.array([[0.5, 0.5, 0.0]])
            )

    def test_intensity_threshold_too_high(self) -> None:
        """Test that intensity threshold >= num_classes raises error."""
        y_up = np.array([[0.25, 0.25, 0.25, 0.25]])
        y_down = np.array([[0.25, 0.25, 0.25, 0.25]])
        with self.assertRaises(BacktestError):
            generate_signals_net_intensity(y_up, y_down, intensity_threshold=4)


class TestGenerateSignalsThreshold(unittest.TestCase):
    """Tests for generate_signals_threshold function."""

    def test_empty_input(self) -> None:
        """Test with empty arrays."""
        y_up = np.array([]).reshape(0, 4)
        y_down = np.array([]).reshape(0, 4)
        signals, confidences = generate_signals_threshold(y_up, y_down)
        self.assertEqual(len(signals), 0)

    def test_up_exceeds_threshold(self) -> None:
        """Test that up probability exceeding threshold generates long."""
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8 > 0.6
        y_down = np.array([[0.9, 0.05, 0.03, 0.02]])  # sum(1:) = 0.1 < 0.6
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.6)
        self.assertEqual(signals[0], +1)

    def test_down_exceeds_threshold(self) -> None:
        """Test that down probability exceeding threshold generates short."""
        y_up = np.array([[0.9, 0.05, 0.03, 0.02]])  # sum(1:) = 0.1
        y_down = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.6)
        self.assertEqual(signals[0], -1)

    def test_neither_exceeds_threshold(self) -> None:
        """Test that no signal when neither exceeds threshold."""
        y_up = np.array([[0.6, 0.2, 0.1, 0.1]])  # sum(1:) = 0.4 < 0.6
        y_down = np.array([[0.6, 0.2, 0.1, 0.1]])  # sum(1:) = 0.4 < 0.6
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.6)
        self.assertEqual(signals[0], 0)

    def test_both_exceed_up_wins(self) -> None:
        """Test that when both exceed, stronger wins (up)."""
        y_up = np.array([[0.1, 0.3, 0.3, 0.3]])  # sum(1:) = 0.9
        y_down = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.6)
        self.assertEqual(signals[0], +1)  # Up wins

    def test_both_exceed_down_wins(self) -> None:
        """Test that when both exceed, stronger wins (down)."""
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        y_down = np.array([[0.1, 0.3, 0.3, 0.3]])  # sum(1:) = 0.9
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.6)
        self.assertEqual(signals[0], -1)  # Down wins

    def test_confidence_values(self) -> None:
        """Test that confidence is max of up and down probabilities."""
        y_up = np.array([[0.2, 0.4, 0.3, 0.1]])  # sum(1:) = 0.8
        y_down = np.array([[0.5, 0.3, 0.1, 0.1]])  # sum(1:) = 0.5
        _, confidences = generate_signals_threshold(y_up, y_down, threshold=0.5)
        self.assertAlmostEqual(confidences[0], 0.8, places=5)  # max(0.8, 0.5)

    def test_high_threshold(self) -> None:
        """Test with high threshold that filters most signals."""
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        y_down = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8
        signals, _ = generate_signals_threshold(y_up, y_down, threshold=0.9)
        self.assertEqual(signals[0], 0)  # Neither exceeds 0.9


class TestGenerateSignals(unittest.TestCase):
    """Tests for generate_signals dispatch function."""

    def test_dispatch_net_intensity(self) -> None:
        """Test that net_intensity strategy is dispatched correctly."""
        config = BacktestConfig(signal_strategy="net_intensity", intensity_threshold=1)
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])
        y_down = np.array([[0.8, 0.1, 0.05, 0.05]])
        signals, _ = generate_signals(y_up, y_down, config)
        self.assertEqual(signals[0], +1)

    def test_dispatch_threshold(self) -> None:
        """Test that threshold strategy is dispatched correctly."""
        config = BacktestConfig(signal_strategy="threshold", signal_threshold=0.5)
        y_up = np.array([[0.2, 0.3, 0.3, 0.2]])  # sum(1:) = 0.8 > 0.5
        y_down = np.array([[0.9, 0.05, 0.03, 0.02]])  # sum(1:) = 0.1 < 0.5
        signals, _ = generate_signals(y_up, y_down, config)
        self.assertEqual(signals[0], +1)


class TestSimulateTrades(unittest.TestCase):
    """Tests for simulate_trades function."""

    def test_no_signals(self) -> None:
        """Test that no signals produces no trades."""
        signals = np.array([0, 0, 0, 0, 0])
        confidences = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        config = BacktestConfig()
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)
        self.assertEqual(len(trades), 0)

    def test_single_long_trade(self) -> None:
        """Test a single long trade."""
        signals = np.array([1, 0, 0, 0, 0])
        confidences = np.array([0.8, 0.0, 0.0, 0.0, 0.0])
        prices = np.array([100.0, 101.0, 105.0, 103.0, 104.0])
        config = BacktestConfig(transaction_cost_pct=0.001)
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.entry_idx, 0)
        self.assertEqual(trade.exit_idx, 2)
        self.assertEqual(trade.direction, "long")
        self.assertEqual(trade.entry_price, 100.0)
        self.assertEqual(trade.exit_price, 105.0)
        # Raw PnL: 5%, minus 0.2% costs = 4.8%
        self.assertAlmostEqual(trade.pnl_pct, 4.8, places=1)

    def test_single_short_trade(self) -> None:
        """Test a single short trade."""
        signals = np.array([-1, 0, 0, 0, 0])
        confidences = np.array([0.7, 0.0, 0.0, 0.0, 0.0])
        prices = np.array([100.0, 99.0, 95.0, 97.0, 98.0])
        config = BacktestConfig(transaction_cost_pct=0.001)
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.direction, "short")
        # Short: entry 100, exit 95. Raw PnL: (100-95)/100 = 5%, minus 0.2% = 4.8%
        self.assertAlmostEqual(trade.pnl_pct, 4.8, places=1)

    def test_losing_trade(self) -> None:
        """Test a losing long trade."""
        signals = np.array([1, 0, 0, 0, 0])
        confidences = np.array([0.6, 0.0, 0.0, 0.0, 0.0])
        prices = np.array([100.0, 98.0, 95.0, 97.0, 99.0])
        config = BacktestConfig(transaction_cost_pct=0.001)
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)

        trade = trades[0]
        # Long: entry 100, exit 95. Raw PnL: -5%, minus 0.2% = -5.2%
        self.assertAlmostEqual(trade.pnl_pct, -5.2, places=1)

    def test_multiple_trades(self) -> None:
        """Test multiple trades in sequence."""
        signals = np.array([1, 0, 0, -1, 0, 0, 1, 0, 0, 0])
        confidences = np.full(10, 0.7)
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
        config = BacktestConfig(transaction_cost_pct=0.0)  # No costs for simplicity
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)

        self.assertEqual(len(trades), 3)
        # Trade 1: Long at 100, exit at 102 -> +2%
        self.assertAlmostEqual(trades[0].pnl_pct, 2.0, places=1)
        # Trade 2: Short at 103, exit at 105 -> (103-105)/103*100 = -1.94%
        self.assertAlmostEqual(trades[1].pnl_pct, -1.94, places=1)
        # Trade 3: Long at 106, exit at 108 -> +1.89%
        self.assertAlmostEqual(trades[2].pnl_pct, 1.89, places=1)

    def test_trade_near_end_excluded(self) -> None:
        """Test that trades that can't complete are excluded."""
        signals = np.array([1, 0, 0, 0, 1])  # Signal at index 4 can't complete with horizon=2
        confidences = np.full(5, 0.7)
        prices = np.array([100, 101, 102, 103, 104], dtype=float)
        config = BacktestConfig(transaction_cost_pct=0.0)
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)

        self.assertEqual(len(trades), 1)  # Only first trade completes
        self.assertEqual(trades[0].entry_idx, 0)

    def test_zero_price_excluded(self) -> None:
        """Test that trades with zero prices are excluded."""
        signals = np.array([1, 0, 0, 0, 0])
        confidences = np.full(5, 0.7)
        prices = np.array([0.0, 101, 102, 103, 104])  # Zero entry price
        config = BacktestConfig()
        trades = simulate_trades(signals, confidences, prices, horizon_steps=2, config=config)
        self.assertEqual(len(trades), 0)

    def test_mismatched_lengths_error(self) -> None:
        """Test that mismatched array lengths raise error."""
        with self.assertRaises(BacktestError):
            simulate_trades(
                signals=np.array([1, 0, 0]),
                confidences=np.array([0.5, 0.5]),  # Wrong length
                prices=np.array([100, 101, 102]),
                horizon_steps=1,
                config=BacktestConfig(),
            )

    def test_invalid_horizon_steps(self) -> None:
        """Test that invalid horizon steps raise error."""
        with self.assertRaises(BacktestError):
            simulate_trades(
                signals=np.array([1, 0, 0]),
                confidences=np.array([0.5, 0.5, 0.5]),
                prices=np.array([100, 101, 102]),
                horizon_steps=0,
                config=BacktestConfig(),
            )


class TestBuildEquityCurve(unittest.TestCase):
    """Tests for build_equity_curve function."""

    def test_no_trades(self) -> None:
        """Test that no trades produces flat equity curve."""
        equity = build_equity_curve([], initial_capital=10000, n_samples=5)
        self.assertEqual(len(equity), 5)
        np.testing.assert_array_equal(equity, [10000] * 5)

    def test_single_winning_trade(self) -> None:
        """Test equity curve with single winning trade."""
        trades = [Trade(entry_idx=0, exit_idx=2, pnl_pct=10.0)]
        equity = build_equity_curve(trades, initial_capital=10000, n_samples=5)
        # Before exit: 10000, after exit: 11000
        self.assertEqual(equity[0], 10000)
        self.assertEqual(equity[1], 10000)
        self.assertEqual(equity[2], 11000)
        self.assertEqual(equity[4], 11000)

    def test_single_losing_trade(self) -> None:
        """Test equity curve with single losing trade."""
        trades = [Trade(entry_idx=1, exit_idx=3, pnl_pct=-5.0)]
        equity = build_equity_curve(trades, initial_capital=10000, n_samples=5)
        self.assertEqual(equity[0], 10000)
        self.assertEqual(equity[2], 10000)
        self.assertEqual(equity[3], 9500)
        self.assertEqual(equity[4], 9500)

    def test_multiple_trades(self) -> None:
        """Test equity curve with multiple trades."""
        trades = [
            Trade(entry_idx=0, exit_idx=2, pnl_pct=10.0),  # 10000 -> 11000
            Trade(entry_idx=3, exit_idx=5, pnl_pct=-5.0),  # 11000 -> 10450
        ]
        equity = build_equity_curve(trades, initial_capital=10000, n_samples=7)
        self.assertEqual(equity[0], 10000)
        self.assertEqual(equity[2], 11000)
        self.assertEqual(equity[5], 10450)

    def test_confidence_position_sizing(self) -> None:
        """Test equity curve with confidence-based position sizing."""
        trades = [Trade(entry_idx=0, exit_idx=2, pnl_pct=10.0, signal_confidence=0.5)]
        equity = build_equity_curve(
            trades,
            initial_capital=10000,
            n_samples=5,
            position_sizing="confidence",
            max_position_pct=1.0,
        )
        # With 0.5 confidence, position is 5000. 10% of 5000 = 500 gain.
        self.assertEqual(equity[2], 10500)


class TestComputeRiskMetrics(unittest.TestCase):
    """Tests for compute_risk_metrics function."""

    def test_no_trades(self) -> None:
        """Test metrics with no trades."""
        metrics = compute_risk_metrics([], np.array([10000, 10000, 10000]))
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["total_return_pct"], 0.0)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)

    def test_all_winning_trades(self) -> None:
        """Test metrics with all winning trades."""
        trades = [
            Trade(entry_idx=0, exit_idx=2, pnl_pct=5.0),
            Trade(entry_idx=3, exit_idx=5, pnl_pct=3.0),
            Trade(entry_idx=6, exit_idx=8, pnl_pct=2.0),
        ]
        equity = np.array([10000, 10000, 10500, 10500, 10500, 10815, 10815, 10815, 11031.3])
        metrics = compute_risk_metrics(trades, equity)

        self.assertEqual(metrics["num_trades"], 3)
        self.assertEqual(metrics["win_rate"], 1.0)
        self.assertGreater(metrics["profit_factor"], 100)  # Capped at 999.99
        self.assertGreater(metrics["sharpe_ratio"], 0)

    def test_all_losing_trades(self) -> None:
        """Test metrics with all losing trades."""
        trades = [
            Trade(entry_idx=0, exit_idx=2, pnl_pct=-5.0),
            Trade(entry_idx=3, exit_idx=5, pnl_pct=-3.0),
        ]
        equity = np.array([10000, 10000, 9500, 9500, 9500, 9215])
        metrics = compute_risk_metrics(trades, equity)

        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)
        self.assertLess(metrics["sharpe_ratio"], 0)

    def test_mixed_trades(self) -> None:
        """Test metrics with mixed winning/losing trades."""
        trades = [
            Trade(entry_idx=0, exit_idx=2, pnl_pct=10.0),  # Win
            Trade(entry_idx=3, exit_idx=5, pnl_pct=-5.0),  # Loss
            Trade(entry_idx=6, exit_idx=8, pnl_pct=8.0),   # Win
            Trade(entry_idx=9, exit_idx=11, pnl_pct=-3.0),  # Loss
        ]
        equity = np.linspace(10000, 11000, 12)
        metrics = compute_risk_metrics(trades, equity)

        self.assertEqual(metrics["num_trades"], 4)
        self.assertEqual(metrics["win_rate"], 0.5)
        # Profit factor: (10+8) / (5+3) = 18/8 = 2.25
        self.assertAlmostEqual(metrics["profit_factor"], 2.25, places=2)
        # Avg PnL: (10-5+8-3)/4 = 2.5
        self.assertAlmostEqual(metrics["avg_trade_pnl_pct"], 2.5, places=2)

    def test_max_drawdown(self) -> None:
        """Test max drawdown calculation."""
        trades = [Trade(entry_idx=0, exit_idx=5, pnl_pct=-20.0)]
        # Equity peaks at 10000, drops to 8000, then recovers
        equity = np.array([10000, 10000, 9000, 8000, 8500, 8000, 9000, 10000])
        metrics = compute_risk_metrics(trades, equity)
        # Max drawdown: (10000-8000)/10000 = 20%
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 20.0, places=1)

    def test_total_return(self) -> None:
        """Test total return calculation."""
        trades = [Trade(entry_idx=0, exit_idx=5, pnl_pct=25.0)]
        equity = np.array([10000, 10000, 10000, 10000, 10000, 12500])
        metrics = compute_risk_metrics(trades, equity)
        self.assertAlmostEqual(metrics["total_return_pct"], 25.0, places=1)

    def test_avg_trade_duration(self) -> None:
        """Test average trade duration calculation."""
        trades = [
            Trade(entry_idx=0, exit_idx=10, pnl_pct=5.0),  # 10 steps
            Trade(entry_idx=15, exit_idx=20, pnl_pct=3.0),  # 5 steps
            Trade(entry_idx=25, exit_idx=40, pnl_pct=2.0),  # 15 steps
        ]
        equity = np.linspace(10000, 11000, 50)
        metrics = compute_risk_metrics(trades, equity)
        # Avg duration: (10+5+15)/3 = 10
        self.assertAlmostEqual(metrics["avg_trade_duration_steps"], 10.0, places=1)


class TestRunBacktest(unittest.TestCase):
    """Tests for run_backtest orchestration function."""

    def _make_config(self, **overrides: Any) -> Dict[str, Any]:
        """Create a test configuration."""
        bt_cfg = {
            "enabled": True,
            "horizon_steps": 5,
            "initial_capital": 10000,
            "transaction_cost": 0.001,
            "signal_strategy": "net_intensity",
            "signal_threshold": 0.6,
            "intensity_threshold": 1,
            "position_sizing": "equal",
            "max_position_pct": 1.0,
        }
        bt_cfg.update(overrides)
        return {"evaluation": {"backtesting": bt_cfg}}

    def test_basic_backtest(self) -> None:
        """Test basic backtest execution."""
        config = self._make_config()

        # Create test data: strong up signal at index 0
        n_samples = 20
        y_prob_up = np.zeros((n_samples, 4))
        y_prob_up[:, 0] = 0.2
        y_prob_up[:, 1] = 0.3
        y_prob_up[:, 2] = 0.3
        y_prob_up[:, 3] = 0.2  # sum(1:) = 0.8

        y_prob_down = np.zeros((n_samples, 4))
        y_prob_down[:, 0] = 0.8
        y_prob_down[:, 1] = 0.1
        y_prob_down[:, 2] = 0.05
        y_prob_down[:, 3] = 0.05  # sum(1:) = 0.2

        prices = np.linspace(100, 110, n_samples)  # Rising prices

        result = run_backtest(
            config=config,
            y_prob_up=y_prob_up,
            y_prob_down=y_prob_down,
            prices=prices,
            horizon_steps=5,
        )

        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.trades), 0)
        self.assertEqual(len(result.equity_curve), n_samples)
        self.assertIn("total_return_pct", result.metrics)

    def test_backtest_with_threshold_strategy(self) -> None:
        """Test backtest with threshold strategy."""
        config = self._make_config(signal_strategy="threshold", signal_threshold=0.7)

        n_samples = 10
        # Most samples have weak signals (sum(1:) = 0.3 < 0.7)
        y_prob_up = np.zeros((n_samples, 4))
        y_prob_up[:, 0] = 0.7  # Class 0 dominant, sum(1:) = 0.3
        y_prob_up[:, 1] = 0.1
        y_prob_up[:, 2] = 0.1
        y_prob_up[:, 3] = 0.1

        # Only indices 0 and 5 have strong up signals
        y_prob_up[0, :] = [0.1, 0.3, 0.3, 0.3]  # sum(1:) = 0.9 > 0.7
        y_prob_up[5, :] = [0.1, 0.3, 0.3, 0.3]  # sum(1:) = 0.9 > 0.7

        # Down signals are weak for all
        y_prob_down = np.zeros((n_samples, 4))
        y_prob_down[:, 0] = 0.7
        y_prob_down[:, 1] = 0.1
        y_prob_down[:, 2] = 0.1
        y_prob_down[:, 3] = 0.1

        prices = np.linspace(100, 105, n_samples)

        result = run_backtest(
            config=config,
            y_prob_up=y_prob_up,
            y_prob_down=y_prob_down,
            prices=prices,
            horizon_steps=2,
        )

        # Should have exactly 2 trades (at indices 0 and 5)
        # Index 5 with horizon 2 completes at index 7, which is within bounds
        self.assertEqual(len(result.trades), 2)

    def test_backtest_empty_data(self) -> None:
        """Test backtest with empty data."""
        config = self._make_config()

        y_prob_up = np.array([]).reshape(0, 4)
        y_prob_down = np.array([]).reshape(0, 4)
        prices = np.array([])

        result = run_backtest(
            config=config,
            y_prob_up=y_prob_up,
            y_prob_down=y_prob_down,
            prices=prices,
            horizon_steps=5,
        )

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.metrics["num_trades"], 0)


class TestLogBacktestToMLflow(unittest.TestCase):
    """Tests for log_backtest_to_mlflow function."""

    def test_log_metrics(self) -> None:
        """Test that metrics are logged to MLflow."""
        result = BacktestResult(
            trades=[Trade(entry_idx=0, exit_idx=5, pnl_pct=5.0)],
            equity_curve=np.array([10000, 10500]),
            metrics={"total_return_pct": 5.0, "win_rate": 1.0},
            config=BacktestConfig(),
        )

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            # Re-import to pick up the mock
            import importlib
            import evaluation.backtesting as bt_module
            try:
                importlib.reload(bt_module)
                bt_module.log_backtest_to_mlflow(result)
                # Check that log_metric was called for each metric
                calls = mock_mlflow.log_metric.call_args_list
                metric_names = [call[0][0] for call in calls]
                self.assertIn("backtest_total_return_pct", metric_names)
                self.assertIn("backtest_win_rate", metric_names)
            finally:
                importlib.reload(bt_module)

    def test_log_artifacts(self) -> None:
        """Test that artifacts are logged to MLflow."""
        result = BacktestResult(
            trades=[Trade(entry_idx=0, exit_idx=5, pnl_pct=5.0)],
            equity_curve=np.array([10000, 10250, 10500]),
            metrics={"total_return_pct": 5.0},
            config=BacktestConfig(),
        )

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            import importlib
            import evaluation.backtesting as bt_module
            try:
                importlib.reload(bt_module)
                bt_module.log_backtest_to_mlflow(result)
                # Check that log_artifact was called
                self.assertTrue(mock_mlflow.log_artifact.called)
                # Should have 3 artifacts: equity_curve, trades, summary
                self.assertEqual(mock_mlflow.log_artifact.call_count, 3)
            finally:
                importlib.reload(bt_module)

    def test_log_empty_result(self) -> None:
        """Test logging empty result doesn't crash."""
        result = BacktestResult(
            trades=[],
            equity_curve=np.array([]),
            metrics={},
            config=BacktestConfig(),
        )

        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            import importlib
            import evaluation.backtesting as bt_module
            try:
                importlib.reload(bt_module)
                bt_module.log_backtest_to_mlflow(result)
                # Should still log summary artifact (at minimum)
                self.assertTrue(mock_mlflow.log_artifact.called)
            finally:
                importlib.reload(bt_module)

    def test_mlflow_import_error(self) -> None:
        """Test graceful handling when MLflow import fails."""
        result = BacktestResult(metrics={"test": 1.0}, config=BacktestConfig())

        # Make mlflow import raise ImportError
        def raise_import_error(*args: Any, **kwargs: Any) -> None:
            raise ImportError("No module named 'mlflow'")

        with patch.dict("sys.modules", {"mlflow": None}):
            import importlib
            import evaluation.backtesting as bt_module
            try:
                importlib.reload(bt_module)
                # Should not raise, just log warning
                bt_module.log_backtest_to_mlflow(result)
            finally:
                importlib.reload(bt_module)


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_signal_strategies(self) -> None:
        """Test SIGNAL_STRATEGIES constant."""
        self.assertIn("net_intensity", SIGNAL_STRATEGIES)
        self.assertIn("threshold", SIGNAL_STRATEGIES)
        self.assertEqual(len(SIGNAL_STRATEGIES), 2)

    def test_position_sizing_methods(self) -> None:
        """Test POSITION_SIZING_METHODS constant."""
        self.assertIn("equal", POSITION_SIZING_METHODS)
        self.assertIn("confidence", POSITION_SIZING_METHODS)
        self.assertEqual(len(POSITION_SIZING_METHODS), 2)


if __name__ == "__main__":
    unittest.main()
