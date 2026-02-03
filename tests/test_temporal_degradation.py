"""Tests for evaluation.temporal_degradation module.

This module tests:
- compute_window_metrics: Per-window metric computation
- compute_temporal_degradation: Rolling window degradation analysis
- WindowMetrics and TemporalDegradationResult dataclasses
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from evaluation.temporal_degradation import (
    WindowMetrics,
    TemporalDegradationResult,
    compute_window_metrics,
    compute_temporal_degradation,
)


class TestComputeWindowMetricsBasic(unittest.TestCase):
    """Basic tests for compute_window_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have accuracy 1.0."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        result = compute_window_metrics(
            y_true, y_pred,
            window_index=0, start_index=0, end_index=6, num_classes=3
        )

        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.precision_macro, 1.0)
        self.assertEqual(result.recall_macro, 1.0)
        self.assertEqual(result.f1_macro, 1.0)
        self.assertEqual(result.num_samples, 6)

    def test_random_predictions(self) -> None:
        """Random predictions should have lower accuracy."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 2, 0, 2, 0, 1])  # All wrong

        result = compute_window_metrics(
            y_true, y_pred,
            window_index=0, start_index=0, end_index=6, num_classes=3
        )

        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.num_samples, 6)

    def test_empty_window(self) -> None:
        """Empty window should return zero metrics."""
        y_true = np.array([], dtype="int64")
        y_pred = np.array([], dtype="int64")

        result = compute_window_metrics(
            y_true, y_pred,
            window_index=0, start_index=0, end_index=0, num_classes=3
        )

        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.num_samples, 0)

    def test_per_class_accuracy(self) -> None:
        """Per-class accuracy should be computed correctly."""
        # Class 0: 2/2 correct, Class 1: 1/2 correct, Class 2: 0/2 correct
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 0, 1])

        result = compute_window_metrics(
            y_true, y_pred,
            window_index=0, start_index=0, end_index=6, num_classes=3
        )

        self.assertEqual(len(result.per_class_accuracy), 3)
        assert_allclose(result.per_class_accuracy[0], 1.0)  # Class 0: perfect
        assert_allclose(result.per_class_accuracy[1], 0.5)  # Class 1: 50%
        assert_allclose(result.per_class_accuracy[2], 0.0)  # Class 2: none correct


class TestComputeWindowMetricsValidation(unittest.TestCase):
    """Validation tests for compute_window_metrics."""

    def test_window_index_preserved(self) -> None:
        """Window index should be preserved in result."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        result = compute_window_metrics(
            y_true, y_pred,
            window_index=5, start_index=100, end_index=102, num_classes=2
        )

        self.assertEqual(result.window_index, 5)
        self.assertEqual(result.start_index, 100)
        self.assertEqual(result.end_index, 102)


class TestComputeTemporalDegradationBasic(unittest.TestCase):
    """Basic tests for compute_temporal_degradation function."""

    def test_uniform_performance_no_degradation(self) -> None:
        """Constant accuracy across windows should show no degradation."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        # Perfect predictions throughout
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = y_true.copy()

        result = compute_temporal_degradation(
            y_true, y_pred, n_classes, num_windows=5
        )

        self.assertEqual(len(result.window_metrics), 5)
        self.assertEqual(result.first_window_accuracy, 1.0)
        self.assertEqual(result.last_window_accuracy, 1.0)
        self.assertEqual(result.total_degradation, 0.0)
        self.assertAlmostEqual(result.overall_trend, 0.0, places=5)

    def test_degrading_performance(self) -> None:
        """Performance that degrades over time should be detected."""
        n_samples = 100
        n_classes = 2

        # Create degrading performance: first half perfect, second half random
        y_true = np.zeros(n_samples, dtype="int64")
        y_pred = np.zeros(n_samples, dtype="int64")
        y_pred[50:] = 1  # Wrong predictions in second half

        result = compute_temporal_degradation(
            y_true, y_pred, n_classes, num_windows=4
        )

        # First window should be better than last
        self.assertGreater(result.first_window_accuracy, result.last_window_accuracy)
        # Total degradation should be positive
        self.assertGreater(result.total_degradation, 0)
        # Trend should be negative (degradation)
        self.assertLess(result.overall_trend, 0)

    def test_improving_performance(self) -> None:
        """Performance that improves over time should be detected."""
        n_samples = 100
        n_classes = 2

        # Create improving performance: first half wrong, second half correct
        y_true = np.zeros(n_samples, dtype="int64")
        y_pred = np.ones(n_samples, dtype="int64")
        y_pred[50:] = 0  # Correct predictions in second half

        result = compute_temporal_degradation(
            y_true, y_pred, n_classes, num_windows=4
        )

        # First window should be worse than last
        self.assertLess(result.first_window_accuracy, result.last_window_accuracy)
        # Total degradation should be negative (improvement)
        self.assertLess(result.total_degradation, 0)
        # Trend should be positive (improvement)
        self.assertGreater(result.overall_trend, 0)

    def test_single_window(self) -> None:
        """Single window should work and show no trend."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])  # 75% accuracy

        result = compute_temporal_degradation(
            y_true, y_pred, num_classes=2, num_windows=1
        )

        self.assertEqual(len(result.window_metrics), 1)
        self.assertEqual(result.first_window_accuracy, 0.75)
        self.assertEqual(result.last_window_accuracy, 0.75)
        self.assertEqual(result.total_degradation, 0.0)


class TestComputeTemporalDegradationValidation(unittest.TestCase):
    """Validation tests for compute_temporal_degradation."""

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched y_true and y_pred shapes should raise."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(y_true, y_pred, num_classes=3)

        self.assertIn("shape", str(ctx.exception).lower())

    def test_2d_array_raises(self) -> None:
        """2D arrays should raise ValueError."""
        y_true = np.array([[0, 1], [2, 0]])
        y_pred = np.array([[0, 1], [2, 0]])

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(y_true, y_pred, num_classes=3)

        self.assertIn("1D", str(ctx.exception))

    def test_empty_array_raises(self) -> None:
        """Empty arrays should raise ValueError."""
        y_true = np.array([], dtype="int64")
        y_pred = np.array([], dtype="int64")

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(y_true, y_pred, num_classes=3)

        self.assertIn("empty", str(ctx.exception).lower())

    def test_zero_windows_raises(self) -> None:
        """num_windows=0 should raise ValueError."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(y_true, y_pred, num_classes=3, num_windows=0)

        self.assertIn("num_windows", str(ctx.exception))

    def test_invalid_overlap_raises(self) -> None:
        """Overlap >= 0.5 should raise ValueError."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(
                y_true, y_pred, num_classes=3, overlap_fraction=0.5
            )

        self.assertIn("overlap", str(ctx.exception).lower())

    def test_single_class_raises(self) -> None:
        """num_classes < 2 should raise ValueError."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        with self.assertRaises(ValueError) as ctx:
            compute_temporal_degradation(y_true, y_pred, num_classes=1)

        self.assertIn("num_classes", str(ctx.exception))


class TestComputeTemporalDegradationOverlap(unittest.TestCase):
    """Tests for overlapping windows."""

    def test_overlap_increases_coverage(self) -> None:
        """Overlapping windows should cover more samples per window."""
        n_samples = 100
        y_true = np.zeros(n_samples, dtype="int64")
        y_pred = np.zeros(n_samples, dtype="int64")

        result_no_overlap = compute_temporal_degradation(
            y_true, y_pred, num_classes=2, num_windows=4, overlap_fraction=0.0
        )
        result_with_overlap = compute_temporal_degradation(
            y_true, y_pred, num_classes=2, num_windows=4, overlap_fraction=0.25
        )

        # With overlap, windows should have more samples
        total_no_overlap = sum(w.num_samples for w in result_no_overlap.window_metrics)
        total_with_overlap = sum(w.num_samples for w in result_with_overlap.window_metrics)
        self.assertGreaterEqual(total_with_overlap, total_no_overlap)


class TestTemporalDegradationResultSerialization(unittest.TestCase):
    """Tests for TemporalDegradationResult serialization."""

    def test_to_dict_structure(self) -> None:
        """to_dict should produce correct structure."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

        result = compute_temporal_degradation(
            y_true, y_pred, num_classes=2, num_windows=2
        )

        data = result.to_dict()

        self.assertIn("window_metrics", data)
        self.assertIn("overall_trend", data)
        self.assertIn("degradation_rate", data)
        self.assertIn("first_window_accuracy", data)
        self.assertIn("last_window_accuracy", data)
        self.assertIn("total_degradation", data)

        self.assertEqual(len(data["window_metrics"]), 2)
        for w in data["window_metrics"]:
            self.assertIn("window_index", w)
            self.assertIn("accuracy", w)
            self.assertIn("f1_macro", w)


class TestWindowMetricsDataclass(unittest.TestCase):
    """Tests for WindowMetrics dataclass."""

    def test_dataclass_fields(self) -> None:
        """WindowMetrics should have all expected fields."""
        metrics = WindowMetrics(
            window_index=0,
            start_index=0,
            end_index=10,
            accuracy=0.8,
            precision_macro=0.75,
            recall_macro=0.7,
            f1_macro=0.72,
            num_samples=10,
            per_class_accuracy=[0.9, 0.7],
        )

        self.assertEqual(metrics.window_index, 0)
        self.assertEqual(metrics.accuracy, 0.8)
        self.assertEqual(metrics.num_samples, 10)
        self.assertEqual(len(metrics.per_class_accuracy), 2)


if __name__ == "__main__":
    unittest.main()
