"""Temporal degradation analysis for model performance over time.

This module provides tools to evaluate how model performance degrades
as predictions move further from the training data boundary, helping
validate non-stationarity assumptions and inform retraining schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class WindowMetrics:
    """Metrics for a single evaluation window.

    Attributes
    ----------
    window_index : int
        Zero-based index of this window.
    start_index : int
        Start sample index (inclusive).
    end_index : int
        End sample index (exclusive).
    accuracy : float
        Classification accuracy in this window.
    precision_macro : float
        Macro-averaged precision.
    recall_macro : float
        Macro-averaged recall.
    f1_macro : float
        Macro-averaged F1 score.
    num_samples : int
        Number of samples in this window.
    per_class_accuracy : List[float]
        Per-class accuracy values.
    """

    window_index: int
    start_index: int
    end_index: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    num_samples: int
    per_class_accuracy: List[float] = field(default_factory=list)


@dataclass
class TemporalDegradationResult:
    """Results from temporal degradation analysis.

    Attributes
    ----------
    window_metrics : List[WindowMetrics]
        Metrics for each evaluation window.
    overall_trend : float
        Linear trend coefficient for accuracy over windows.
        Negative means degradation, positive means improvement.
    degradation_rate : float
        Percentage accuracy drop per window (average).
    first_window_accuracy : float
        Accuracy in the first (earliest) window.
    last_window_accuracy : float
        Accuracy in the last (most recent) window.
    total_degradation : float
        Absolute accuracy drop from first to last window.
    """

    window_metrics: List[WindowMetrics]
    overall_trend: float
    degradation_rate: float
    first_window_accuracy: float
    last_window_accuracy: float
    total_degradation: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "window_metrics": [
                {
                    "window_index": w.window_index,
                    "start_index": w.start_index,
                    "end_index": w.end_index,
                    "accuracy": w.accuracy,
                    "precision_macro": w.precision_macro,
                    "recall_macro": w.recall_macro,
                    "f1_macro": w.f1_macro,
                    "num_samples": w.num_samples,
                    "per_class_accuracy": w.per_class_accuracy,
                }
                for w in self.window_metrics
            ],
            "overall_trend": self.overall_trend,
            "degradation_rate": self.degradation_rate,
            "first_window_accuracy": self.first_window_accuracy,
            "last_window_accuracy": self.last_window_accuracy,
            "total_degradation": self.total_degradation,
        }


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_index: int,
    start_index: int,
    end_index: int,
    num_classes: int,
) -> WindowMetrics:
    """Compute metrics for a single evaluation window.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels of shape (n_samples,).
    window_index : int
        Index of this window.
    start_index : int
        Start sample index.
    end_index : int
        End sample index.
    num_classes : int
        Number of classes.

    Returns
    -------
    WindowMetrics
        Computed metrics for this window.
    """
    n_samples = len(y_true)

    if n_samples == 0:
        return WindowMetrics(
            window_index=window_index,
            start_index=start_index,
            end_index=end_index,
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            num_samples=0,
            per_class_accuracy=[],
        )

    # Overall accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[int(t), int(p)] += 1

    # Per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_accuracy = []

    for cls in range(num_classes):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)
        class_total = float(confusion[cls, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        class_acc = tp / class_total if class_total > 0 else 0.0

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        per_class_accuracy.append(class_acc)

    precision_macro = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    recall_macro = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    f1_macro = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    return WindowMetrics(
        window_index=window_index,
        start_index=start_index,
        end_index=end_index,
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        num_samples=n_samples,
        per_class_accuracy=per_class_accuracy,
    )


def compute_temporal_degradation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    *,
    num_windows: int = 5,
    overlap_fraction: float = 0.0,
) -> TemporalDegradationResult:
    """Analyze model performance degradation over time using rolling windows.

    Divides the test set into chronological windows and computes metrics
    for each window, then analyzes the trend to quantify degradation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n_samples,), ordered chronologically.
    y_pred : np.ndarray
        Predicted labels of shape (n_samples,).
    num_classes : int
        Number of classes.
    num_windows : int
        Number of evaluation windows to create.
    overlap_fraction : float
        Fraction of overlap between consecutive windows (0.0 to 0.5).

    Returns
    -------
    TemporalDegradationResult
        Analysis results including per-window metrics and degradation stats.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    y_true_arr = np.asarray(y_true, dtype="int64")
    y_pred_arr = np.asarray(y_pred, dtype="int64")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )

    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D arrays, got {y_true_arr.ndim}D")

    n_samples = len(y_true_arr)

    if n_samples == 0:
        raise ValueError("Cannot compute temporal degradation on empty dataset")

    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    if not 0.0 <= overlap_fraction < 0.5:
        raise ValueError(f"overlap_fraction must be in [0.0, 0.5), got {overlap_fraction}")

    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    # Compute window boundaries
    # With overlap, windows share some samples
    # Window size = n_samples / (num_windows - (num_windows - 1) * overlap_fraction)
    if num_windows == 1:
        window_size = n_samples
        step_size = n_samples
    else:
        # Effective number of non-overlapping units
        effective_units = 1 + (num_windows - 1) * (1 - overlap_fraction)
        window_size = int(np.ceil(n_samples / effective_units))
        step_size = int(window_size * (1 - overlap_fraction))
        step_size = max(1, step_size)

    window_metrics: List[WindowMetrics] = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, n_samples)

        if start_idx >= n_samples:
            break

        y_true_window = y_true_arr[start_idx:end_idx]
        y_pred_window = y_pred_arr[start_idx:end_idx]

        metrics = compute_window_metrics(
            y_true_window,
            y_pred_window,
            window_index=i,
            start_index=start_idx,
            end_index=end_idx,
            num_classes=num_classes,
        )
        window_metrics.append(metrics)

    if len(window_metrics) == 0:
        return TemporalDegradationResult(
            window_metrics=[],
            overall_trend=0.0,
            degradation_rate=0.0,
            first_window_accuracy=0.0,
            last_window_accuracy=0.0,
            total_degradation=0.0,
        )

    accuracies = [w.accuracy for w in window_metrics]
    first_acc = accuracies[0]
    last_acc = accuracies[-1]
    total_degradation = first_acc - last_acc

    # Compute linear trend using simple linear regression
    if len(accuracies) > 1:
        x = np.arange(len(accuracies), dtype="float64")
        y = np.array(accuracies, dtype="float64")

        # y = mx + b, we want m (slope)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator > 0:
            slope = float(numerator / denominator)
        else:
            slope = 0.0

        # Degradation rate as average drop per window
        degradation_rate = -slope if len(accuracies) > 1 else 0.0
    else:
        slope = 0.0
        degradation_rate = 0.0

    return TemporalDegradationResult(
        window_metrics=window_metrics,
        overall_trend=slope,
        degradation_rate=degradation_rate,
        first_window_accuracy=first_acc,
        last_window_accuracy=last_acc,
        total_degradation=total_degradation,
    )


def evaluate_temporal_degradation_from_generator(
    model: Any,
    data_generator: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    head: str = "up",
    *,
    num_windows: int = 5,
    overlap_fraction: float = 0.0,
) -> TemporalDegradationResult:
    """Evaluate temporal degradation by streaming predictions from a generator.

    This is memory-efficient for large test sets, as it processes data in chunks
    rather than loading everything at once.

    Parameters
    ----------
    model : Any
        Keras model with predict() method returning [up_probs, down_probs].
    data_generator : Callable
        Generator that yields (x_batch, y_up_batch, y_down_batch) tuples.
    num_classes : int
        Number of classes.
    head : str
        Which head to evaluate: "up" or "down".
    num_windows : int
        Number of evaluation windows.
    overlap_fraction : float
        Fraction of overlap between windows.

    Returns
    -------
    TemporalDegradationResult
        Temporal degradation analysis results.
    """
    if head not in ("up", "down"):
        raise ValueError(f"head must be 'up' or 'down', got {head!r}")

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    for x_batch, y_up_batch, y_down_batch in data_generator():
        y_pred_list = model.predict(x_batch, verbose=0)

        if not isinstance(y_pred_list, (list, tuple)) or len(y_pred_list) != 2:
            raise ValueError("Model must return two outputs for two_head_intensity")

        y_prob_up, y_prob_down = y_pred_list
        y_prob = y_prob_up if head == "up" else y_prob_down
        y_true = y_up_batch if head == "up" else y_down_batch

        y_pred_labels = np.argmax(y_prob, axis=1)
        all_y_true.append(np.asarray(y_true, dtype="int64"))
        all_y_pred.append(y_pred_labels)

    if not all_y_true:
        raise ValueError("Generator yielded no data")

    y_true_concat = np.concatenate(all_y_true)
    y_pred_concat = np.concatenate(all_y_pred)

    return compute_temporal_degradation(
        y_true_concat,
        y_pred_concat,
        num_classes,
        num_windows=num_windows,
        overlap_fraction=overlap_fraction,
    )


__all__ = [
    "WindowMetrics",
    "TemporalDegradationResult",
    "compute_window_metrics",
    "compute_temporal_degradation",
    "evaluate_temporal_degradation_from_generator",
]
