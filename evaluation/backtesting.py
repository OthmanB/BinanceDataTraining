"""Backtesting module for simulated trading performance evaluation.

This module provides utilities to:
- Convert two-head intensity model predictions into trading signals
- Simulate trade execution with transaction costs
- Compute risk-adjusted performance metrics (Sharpe, drawdown, win rate)
- Log results to MLflow

The backtesting framework supports two signal generation strategies:
1. net_intensity: Compare aggregated up vs down probabilities
2. threshold: Only trade when one direction exceeds a confidence threshold
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Valid signal generation strategies
SIGNAL_STRATEGIES = frozenset({"net_intensity", "threshold"})

# Valid position sizing methods
POSITION_SIZING_METHODS = frozenset({"equal", "confidence"})


class BacktestError(Exception):
    """Exception raised for backtesting-related errors."""

    pass


@dataclass
class BacktestConfig:
    """Configuration for backtesting simulation.

    Attributes:
        initial_capital: Starting capital for the backtest
        transaction_cost_pct: Per-trade transaction cost as a fraction (e.g., 0.001 = 0.1%)
        signal_strategy: Strategy for generating signals ("net_intensity" or "threshold")
        signal_threshold: Probability threshold for threshold strategy
        intensity_threshold: Minimum intensity class for action (0=any, 1=1%+, 2=2%+, 3=5%+)
        position_sizing: How to size positions ("equal" or "confidence")
        max_position_pct: Maximum capital per position as a fraction (1.0 = all-in)
    """

    initial_capital: float = 10000.0
    transaction_cost_pct: float = 0.001
    signal_strategy: str = "net_intensity"
    signal_threshold: float = 0.6
    intensity_threshold: int = 1
    position_sizing: str = "equal"
    max_position_pct: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.initial_capital <= 0:
            raise BacktestError("initial_capital must be positive")
        if self.transaction_cost_pct < 0:
            raise BacktestError("transaction_cost_pct cannot be negative")
        if self.signal_strategy not in SIGNAL_STRATEGIES:
            raise BacktestError(
                f"signal_strategy must be one of {SIGNAL_STRATEGIES}, got {self.signal_strategy!r}"
            )
        if not 0 < self.signal_threshold <= 1.0:
            raise BacktestError("signal_threshold must be in (0, 1]")
        if not 0 <= self.intensity_threshold <= 3:
            raise BacktestError("intensity_threshold must be in [0, 3]")
        if self.position_sizing not in POSITION_SIZING_METHODS:
            raise BacktestError(
                f"position_sizing must be one of {POSITION_SIZING_METHODS}, got {self.position_sizing!r}"
            )
        if not 0 < self.max_position_pct <= 1.0:
            raise BacktestError("max_position_pct must be in (0, 1]")


@dataclass
class Trade:
    """Represents a single trade in the backtest.

    Attributes:
        entry_idx: Sample index at trade entry
        exit_idx: Sample index at trade exit
        entry_time: Timestamp at trade entry (if available)
        exit_time: Timestamp at trade exit (if available)
        direction: Trade direction ("long" or "short")
        entry_price: Price at entry
        exit_price: Price at exit
        pnl_pct: Profit/loss as a percentage
        signal_confidence: Confidence level of the signal that triggered the trade
    """

    entry_idx: int
    exit_idx: int
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    direction: str = "long"
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    signal_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_pct": self.pnl_pct,
            "signal_confidence": self.signal_confidence,
        }


@dataclass
class BacktestResult:
    """Results from a backtesting simulation.

    Attributes:
        trades: List of executed trades
        equity_curve: Array of equity values over time
        timestamps: Array of timestamps for equity curve (if available)
        metrics: Dictionary of computed risk/return metrics
        config: Configuration used for the backtest
    """

    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[BacktestConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "num_trades": len(self.trades),
            "trades": [t.to_dict() for t in self.trades],
            "metrics": self.metrics,
            "config": {
                "initial_capital": self.config.initial_capital if self.config else None,
                "transaction_cost_pct": self.config.transaction_cost_pct if self.config else None,
                "signal_strategy": self.config.signal_strategy if self.config else None,
                "signal_threshold": self.config.signal_threshold if self.config else None,
                "intensity_threshold": self.config.intensity_threshold if self.config else None,
                "position_sizing": self.config.position_sizing if self.config else None,
                "max_position_pct": self.config.max_position_pct if self.config else None,
            },
        }


def generate_signals_net_intensity(
    y_prob_up: np.ndarray,
    y_prob_down: np.ndarray,
    intensity_threshold: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate trading signals using net intensity comparison.

    Compares aggregated probability of significant up moves vs down moves.
    - Long (+1) if P(up > threshold) > P(down > threshold)
    - Short (-1) if P(down > threshold) > P(up > threshold)
    - Hold (0) if equal

    Args:
        y_prob_up: Up-intensity probabilities, shape (N, num_classes)
        y_prob_down: Down-intensity probabilities, shape (N, num_classes)
        intensity_threshold: Minimum class for action (classes >= this are summed)

    Returns:
        Tuple of (signals, confidences):
        - signals: Array of {-1, 0, +1} for short/hold/long
        - confidences: Array of confidence values (probability gap)
    """
    if y_prob_up.ndim != 2 or y_prob_down.ndim != 2:
        raise BacktestError("Probability arrays must be 2D with shape (N, num_classes)")
    if y_prob_up.shape != y_prob_down.shape:
        raise BacktestError("Up and down probability arrays must have the same shape")
    if y_prob_up.shape[0] == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=np.float64)

    num_classes = y_prob_up.shape[1]
    if intensity_threshold >= num_classes:
        raise BacktestError(
            f"intensity_threshold ({intensity_threshold}) must be < num_classes ({num_classes})"
        )

    # Aggregate probability for significant moves (classes >= intensity_threshold)
    up_prob = y_prob_up[:, intensity_threshold:].sum(axis=1)
    down_prob = y_prob_down[:, intensity_threshold:].sum(axis=1)

    signals = np.zeros(len(y_prob_up), dtype=np.int8)
    signals[up_prob > down_prob] = +1  # Long
    signals[down_prob > up_prob] = -1  # Short
    # If equal (rare), hold (0)

    confidences = np.abs(up_prob - down_prob)
    return signals, confidences


def generate_signals_threshold(
    y_prob_up: np.ndarray,
    y_prob_down: np.ndarray,
    threshold: float = 0.6,
    intensity_threshold: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate trading signals using threshold-based strategy.

    Only generates signals when one direction exceeds a confidence threshold.
    - Long (+1) if P(up > intensity) > threshold
    - Short (-1) if P(down > intensity) > threshold
    - Hold (0) otherwise

    Args:
        y_prob_up: Up-intensity probabilities, shape (N, num_classes)
        y_prob_down: Down-intensity probabilities, shape (N, num_classes)
        threshold: Probability threshold for generating signals
        intensity_threshold: Minimum class for action

    Returns:
        Tuple of (signals, confidences):
        - signals: Array of {-1, 0, +1}
        - confidences: Array of confidence values
    """
    if y_prob_up.ndim != 2 or y_prob_down.ndim != 2:
        raise BacktestError("Probability arrays must be 2D with shape (N, num_classes)")
    if y_prob_up.shape != y_prob_down.shape:
        raise BacktestError("Up and down probability arrays must have the same shape")
    if y_prob_up.shape[0] == 0:
        return np.array([], dtype=np.int8), np.array([], dtype=np.float64)

    num_classes = y_prob_up.shape[1]
    if intensity_threshold >= num_classes:
        raise BacktestError(
            f"intensity_threshold ({intensity_threshold}) must be < num_classes ({num_classes})"
        )

    # Aggregate probability for significant moves
    up_prob = y_prob_up[:, intensity_threshold:].sum(axis=1)
    down_prob = y_prob_down[:, intensity_threshold:].sum(axis=1)

    signals = np.zeros(len(y_prob_up), dtype=np.int8)
    up_exceeds = up_prob > threshold
    down_exceeds = down_prob > threshold

    signals[up_exceeds] = +1
    signals[down_exceeds] = -1

    # Conflict resolution: strongest wins
    conflict = up_exceeds & down_exceeds
    signals[conflict] = np.where(up_prob[conflict] > down_prob[conflict], +1, -1)

    confidences = np.maximum(up_prob, down_prob)
    return signals, confidences


def generate_signals(
    y_prob_up: np.ndarray,
    y_prob_down: np.ndarray,
    config: BacktestConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate trading signals based on configuration.

    Dispatches to the appropriate strategy based on config.signal_strategy.

    Args:
        y_prob_up: Up-intensity probabilities, shape (N, num_classes)
        y_prob_down: Down-intensity probabilities, shape (N, num_classes)
        config: Backtest configuration

    Returns:
        Tuple of (signals, confidences)
    """
    if config.signal_strategy == "net_intensity":
        return generate_signals_net_intensity(
            y_prob_up, y_prob_down, config.intensity_threshold
        )
    elif config.signal_strategy == "threshold":
        return generate_signals_threshold(
            y_prob_up, y_prob_down, config.signal_threshold, config.intensity_threshold
        )
    else:
        raise BacktestError(f"Unknown signal strategy: {config.signal_strategy}")


def simulate_trades(
    signals: np.ndarray,
    confidences: np.ndarray,
    prices: np.ndarray,
    horizon_steps: int,
    config: BacktestConfig,
    timestamps: Optional[np.ndarray] = None,
) -> List[Trade]:
    """Simulate trade execution based on signals.

    For each non-zero signal:
    - Entry at current price with transaction cost
    - Exit at price after horizon_steps (hold until horizon)
    - Record PnL after transaction costs

    Args:
        signals: Array of trading signals {-1, 0, +1}
        confidences: Array of signal confidence values
        prices: Array of prices at each sample point
        horizon_steps: Number of steps to hold position
        config: Backtest configuration
        timestamps: Optional array of timestamps

    Returns:
        List of Trade objects
    """
    if len(signals) != len(prices):
        raise BacktestError("signals and prices must have the same length")
    if len(confidences) != len(signals):
        raise BacktestError("confidences and signals must have the same length")
    if horizon_steps <= 0:
        raise BacktestError("horizon_steps must be positive")

    trades: List[Trade] = []
    n_samples = len(signals)

    for i in range(n_samples):
        if signals[i] == 0:
            continue

        # Check if we can exit within the data
        exit_idx = i + horizon_steps
        if exit_idx >= n_samples:
            # Not enough data to complete this trade
            continue

        entry_price = float(prices[i])
        exit_price = float(prices[exit_idx])

        if entry_price <= 0 or exit_price <= 0:
            # Skip invalid prices
            continue

        direction = "long" if signals[i] == +1 else "short"

        # Calculate PnL
        if direction == "long":
            raw_pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:  # short
            raw_pnl_pct = (entry_price - exit_price) / entry_price * 100.0

        # Apply transaction costs (entry + exit)
        total_cost_pct = config.transaction_cost_pct * 2 * 100.0
        pnl_pct = raw_pnl_pct - total_cost_pct

        # Get timestamps if available
        entry_time = None
        exit_time = None
        if timestamps is not None and len(timestamps) > exit_idx:
            try:
                entry_ts = timestamps[i]
                exit_ts = timestamps[exit_idx]
                if hasattr(entry_ts, "astype"):
                    entry_time = datetime.utcfromtimestamp(
                        entry_ts.astype("datetime64[s]").astype("int64")
                    )
                    exit_time = datetime.utcfromtimestamp(
                        exit_ts.astype("datetime64[s]").astype("int64")
                    )
                elif isinstance(entry_ts, (int, float)):
                    entry_time = datetime.utcfromtimestamp(entry_ts)
                    exit_time = datetime.utcfromtimestamp(exit_ts)
            except Exception:  # noqa: BLE001
                pass  # Timestamps are optional

        trade = Trade(
            entry_idx=i,
            exit_idx=exit_idx,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            signal_confidence=float(confidences[i]),
        )
        trades.append(trade)

    return trades


def build_equity_curve(
    trades: List[Trade],
    initial_capital: float,
    n_samples: int,
    position_sizing: str = "equal",
    max_position_pct: float = 1.0,
) -> np.ndarray:
    """Build equity curve from list of trades.

    Args:
        trades: List of Trade objects
        initial_capital: Starting capital
        n_samples: Total number of sample points
        position_sizing: How to size positions ("equal" or "confidence")
        max_position_pct: Maximum position size as fraction of equity

    Returns:
        Array of equity values at each sample point
    """
    if n_samples <= 0:
        return np.array([initial_capital])

    equity = np.full(n_samples, initial_capital, dtype=np.float64)

    if not trades:
        return equity

    # Sort trades by entry index
    sorted_trades = sorted(trades, key=lambda t: t.entry_idx)

    current_equity = initial_capital
    for trade in sorted_trades:
        if trade.entry_idx >= n_samples or trade.exit_idx >= n_samples:
            continue

        # Determine position size
        if position_sizing == "confidence":
            position_pct = min(trade.signal_confidence, max_position_pct)
        else:  # equal
            position_pct = max_position_pct

        # Calculate position value
        position_value = current_equity * position_pct

        # Apply PnL
        pnl = position_value * (trade.pnl_pct / 100.0)
        current_equity += pnl

        # Update equity from exit point onwards
        equity[trade.exit_idx:] = current_equity

    return equity


def compute_risk_metrics(
    trades: List[Trade],
    equity_curve: np.ndarray,
    annualization_factor: float = 252.0,
) -> Dict[str, float]:
    """Compute risk and return metrics from backtest results.

    Args:
        trades: List of Trade objects
        equity_curve: Array of equity values
        annualization_factor: Factor for annualizing returns (252 for daily)

    Returns:
        Dictionary of metrics
    """
    metrics: Dict[str, float] = {}

    # Basic trade statistics
    num_trades = len(trades)
    metrics["num_trades"] = float(num_trades)

    if num_trades == 0:
        metrics["total_return_pct"] = 0.0
        metrics["sharpe_ratio"] = 0.0
        metrics["max_drawdown_pct"] = 0.0
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_trade_pnl_pct"] = 0.0
        metrics["avg_trade_duration_steps"] = 0.0
        return metrics

    # Total return
    if len(equity_curve) > 1:
        initial = equity_curve[0]
        final = equity_curve[-1]
        total_return_pct = (final - initial) / initial * 100.0 if initial > 0 else 0.0
    else:
        total_return_pct = 0.0
    metrics["total_return_pct"] = float(total_return_pct)

    # Win rate
    winning_trades = [t for t in trades if t.pnl_pct > 0]
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
    metrics["win_rate"] = float(win_rate)

    # Average trade PnL
    avg_pnl = np.mean([t.pnl_pct for t in trades])
    metrics["avg_trade_pnl_pct"] = float(avg_pnl)

    # Average trade duration
    avg_duration = np.mean([t.exit_idx - t.entry_idx for t in trades])
    metrics["avg_trade_duration_steps"] = float(avg_duration)

    # Profit factor
    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    if profit_factor == float("inf"):
        profit_factor = 999.99  # Cap for display
    metrics["profit_factor"] = float(profit_factor)

    # Sharpe ratio (using trade returns)
    trade_returns = np.array([t.pnl_pct / 100.0 for t in trades])
    if len(trade_returns) > 1 and np.std(trade_returns) > 0:
        sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = float(sharpe)

    # Maximum drawdown
    if len(equity_curve) > 1:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100.0
        max_drawdown = float(np.max(drawdown))
    else:
        max_drawdown = 0.0
    metrics["max_drawdown_pct"] = max_drawdown

    return metrics


def run_backtest(
    config: Dict[str, Any],
    y_prob_up: np.ndarray,
    y_prob_down: np.ndarray,
    prices: np.ndarray,
    horizon_steps: int,
    timestamps: Optional[np.ndarray] = None,
) -> BacktestResult:
    """Run a complete backtest simulation.

    Args:
        config: Full configuration dictionary
        y_prob_up: Up-intensity probabilities, shape (N, num_classes)
        y_prob_down: Down-intensity probabilities, shape (N, num_classes)
        prices: Array of prices at each sample point
        horizon_steps: Number of steps in prediction horizon
        timestamps: Optional array of timestamps

    Returns:
        BacktestResult with trades, equity curve, and metrics
    """
    # Extract backtest configuration
    eval_cfg = config.get("evaluation", {})
    backtest_cfg_dict = eval_cfg.get("backtesting", {})

    bt_config = BacktestConfig(
        initial_capital=float(backtest_cfg_dict.get("initial_capital", 10000.0)),
        transaction_cost_pct=float(backtest_cfg_dict.get("transaction_cost", 0.001)),
        signal_strategy=str(backtest_cfg_dict.get("signal_strategy", "net_intensity")),
        signal_threshold=float(backtest_cfg_dict.get("signal_threshold", 0.6)),
        intensity_threshold=int(backtest_cfg_dict.get("intensity_threshold", 1)),
        position_sizing=str(backtest_cfg_dict.get("position_sizing", "equal")),
        max_position_pct=float(backtest_cfg_dict.get("max_position_pct", 1.0)),
    )

    # Generate signals
    signals, confidences = generate_signals(y_prob_up, y_prob_down, bt_config)

    # Simulate trades
    trades = simulate_trades(
        signals=signals,
        confidences=confidences,
        prices=prices,
        horizon_steps=horizon_steps,
        config=bt_config,
        timestamps=timestamps,
    )

    # Build equity curve
    equity_curve = build_equity_curve(
        trades=trades,
        initial_capital=bt_config.initial_capital,
        n_samples=len(prices),
        position_sizing=bt_config.position_sizing,
        max_position_pct=bt_config.max_position_pct,
    )

    # Compute metrics
    metrics = compute_risk_metrics(trades, equity_curve)

    logger.info(
        "Backtest complete: num_trades=%d, total_return=%.2f%%, sharpe=%.2f, max_drawdown=%.2f%%, win_rate=%.2f%%",
        len(trades),
        metrics.get("total_return_pct", 0),
        metrics.get("sharpe_ratio", 0),
        metrics.get("max_drawdown_pct", 0),
        metrics.get("win_rate", 0) * 100,
    )

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        timestamps=timestamps,
        metrics=metrics,
        config=bt_config,
    )


def log_backtest_to_mlflow(result: BacktestResult) -> None:
    """Log backtest results to MLflow.

    Logs metrics as scalars and saves equity curve and trades as artifacts.

    Args:
        result: BacktestResult to log
    """
    try:
        import mlflow  # type: ignore[import]
    except ImportError:
        logger.warning("MLflow not available; skipping backtest logging")
        return

    # Log metrics
    for name, value in result.metrics.items():
        try:
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                mlflow.log_metric(f"backtest_{name}", float(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log backtest metric %s: %s", name, exc)

    # Log artifacts
    try:
        tmp_dir = Path(tempfile.mkdtemp())

        # Equity curve CSV
        if len(result.equity_curve) > 0:
            equity_path = tmp_dir / "equity_curve.csv"
            # Compute drawdown for export
            peak = np.maximum.accumulate(result.equity_curve)
            drawdown_pct = (peak - result.equity_curve) / peak * 100.0
            data = np.column_stack([
                np.arange(len(result.equity_curve)),
                result.equity_curve,
                drawdown_pct,
            ])
            header = "sample_idx,equity,drawdown_pct"
            np.savetxt(
                equity_path,
                data,
                fmt=["%d", "%.2f", "%.4f"],
                delimiter=",",
                header=header,
                comments="",
            )
            mlflow.log_artifact(str(equity_path), artifact_path="backtest")

        # Trades CSV
        if result.trades:
            trades_path = tmp_dir / "trades.csv"
            with open(trades_path, "w") as f:
                f.write("entry_idx,exit_idx,direction,entry_price,exit_price,pnl_pct,confidence\n")
                for t in result.trades:
                    f.write(
                        f"{t.entry_idx},{t.exit_idx},{t.direction},"
                        f"{t.entry_price:.6f},{t.exit_price:.6f},{t.pnl_pct:.4f},{t.signal_confidence:.4f}\n"
                    )
            mlflow.log_artifact(str(trades_path), artifact_path="backtest")

        # Summary JSON
        summary_path = tmp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        mlflow.log_artifact(str(summary_path), artifact_path="backtest")

        logger.info("Logged backtest artifacts to MLflow: equity_curve, trades, summary")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log backtest artifacts to MLflow: %s", exc)


__all__ = [
    "SIGNAL_STRATEGIES",
    "POSITION_SIZING_METHODS",
    "BacktestError",
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "generate_signals_net_intensity",
    "generate_signals_threshold",
    "generate_signals",
    "simulate_trades",
    "build_equity_curve",
    "compute_risk_metrics",
    "run_backtest",
    "log_backtest_to_mlflow",
]
