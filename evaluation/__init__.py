"""Evaluation utilities for Binance ML Training Platform.

This module provides:
- evaluate_model: Evaluate a trained model on test data
- evaluate_snapshot_model: Evaluate using snapshot datasets
- calibration: Post-hoc calibration utilities (temperature scaling)
- temporal_degradation: Temporal degradation analysis utilities
- backtesting: Simulated trading performance evaluation
"""

from .evaluator import evaluate_model, evaluate_snapshot_model
from .calibration import (
    compute_calibration_metrics,
    TemperatureScaler,
    apply_temperature_scaling,
    fit_temperature,
)
from .temporal_degradation import (
    WindowMetrics,
    TemporalDegradationResult,
    compute_window_metrics,
    compute_temporal_degradation,
    evaluate_temporal_degradation_from_generator,
)
from .backtesting import (
    BacktestError,
    BacktestConfig,
    Trade,
    BacktestResult,
    run_backtest,
    log_backtest_to_mlflow,
)

__all__ = [
    "evaluate_model",
    "evaluate_snapshot_model",
    "compute_calibration_metrics",
    "TemperatureScaler",
    "apply_temperature_scaling",
    "fit_temperature",
    "WindowMetrics",
    "TemporalDegradationResult",
    "compute_window_metrics",
    "compute_temporal_degradation",
    "evaluate_temporal_degradation_from_generator",
    "BacktestError",
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "run_backtest",
    "log_backtest_to_mlflow",
]
