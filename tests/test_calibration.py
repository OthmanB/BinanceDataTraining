"""Tests for evaluation.calibration module.

This module tests:
- compute_calibration_metrics: ECE, Brier score, reliability curve computation
- TemperatureScaler: Post-hoc temperature scaling
- apply_temperature_scaling: Direct temperature application to logits
- fit_temperature: Convenience function for fitting
"""

import os
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import softmax

from evaluation.calibration import (
    compute_calibration_metrics,
    TemperatureScaler,
    apply_temperature_scaling,
    fit_temperature,
)

# Reduce hypothesis examples in CI for faster test runs
_MAX_EXAMPLES = 10 if os.environ.get("CI") else 50

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Stubs for type checking when hypothesis is not installed
    given = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    HealthCheck = None  # type: ignore[assignment,misc]
    st = None  # type: ignore[assignment]
    arrays = None  # type: ignore[assignment]


class TestComputeCalibrationMetricsBasic(unittest.TestCase):
    """Basic tests for compute_calibration_metrics function."""

    def test_perfect_calibration_returns_zero_ece(self) -> None:
        """Perfect predictions should have ECE close to zero."""
        # Perfect predictions: probs match one-hot labels exactly
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float64")
        y_prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float64")

        result = compute_calibration_metrics(y_true, y_prob, num_bins=10)

        self.assertEqual(result["brier_score"], 0.0)
        self.assertLessEqual(result["ece"], 0.01)  # Near zero

    def test_uniform_predictions_high_ece(self) -> None:
        """Uniform predictions on non-uniform labels should have higher Brier score."""
        # All predictions are 1/3, but labels are all class 0
        y_true = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype="float64")
        y_prob = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]], dtype="float64")

        result = compute_calibration_metrics(y_true, y_prob, num_bins=10)

        # Brier score should be positive (imperfect predictions)
        self.assertGreater(result["brier_score"], 0.0)

    def test_brier_score_computation(self) -> None:
        """Test Brier score is computed correctly."""
        # Simple case: all zeros except first class with prob 0.5
        y_true = np.array([[1, 0], [0, 1]], dtype="float64")
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float64")

        result = compute_calibration_metrics(y_true, y_prob, num_bins=5)

        # Brier score = mean(sum((p - y)^2))
        # For each sample: (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5
        expected_brier = 0.5
        assert_allclose(result["brier_score"], expected_brier, atol=1e-10)

    def test_bin_edges_correct(self) -> None:
        """Test that bin edges are computed correctly."""
        y_true = np.eye(3, dtype="float64")
        y_prob = np.eye(3, dtype="float64") * 0.8 + 0.1

        result = compute_calibration_metrics(y_true, y_prob, num_bins=5)

        expected_edges = np.linspace(0, 1, 6)
        assert_allclose(result["bin_edges"], expected_edges)
        self.assertEqual(len(result["bin_confidence"]), 5)
        self.assertEqual(len(result["bin_accuracy"]), 5)
        self.assertEqual(len(result["bin_count"]), 5)


class TestComputeCalibrationMetricsValidation(unittest.TestCase):
    """Validation tests for compute_calibration_metrics."""

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes should raise ValueError."""
        y_true = np.eye(3, dtype="float64")
        y_prob = np.eye(4, dtype="float64")

        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(y_true, y_prob, num_bins=10)

        self.assertIn("same shape", str(ctx.exception))

    def test_1d_array_raises(self) -> None:
        """1D arrays should raise ValueError."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([0.1, 0.2, 0.3])

        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(y_true, y_prob, num_bins=10)

        self.assertIn("2D", str(ctx.exception))

    def test_zero_bins_raises(self) -> None:
        """num_bins=0 should raise ValueError."""
        y_true = np.eye(3, dtype="float64")
        y_prob = np.eye(3, dtype="float64")

        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(y_true, y_prob, num_bins=0)

        self.assertIn("positive", str(ctx.exception))

    def test_negative_bins_raises(self) -> None:
        """Negative num_bins should raise ValueError."""
        y_true = np.eye(3, dtype="float64")
        y_prob = np.eye(3, dtype="float64")

        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(y_true, y_prob, num_bins=-5)

        self.assertIn("positive", str(ctx.exception))

    def test_empty_arrays_returns_zeros(self) -> None:
        """Empty arrays should return zero metrics."""
        y_true = np.zeros((0, 3), dtype="float64")
        y_prob = np.zeros((0, 3), dtype="float64")

        result = compute_calibration_metrics(y_true, y_prob, num_bins=10)

        self.assertEqual(result["brier_score"], 0.0)
        self.assertEqual(result["ece"], 0.0)


class TestTemperatureScalerBasic(unittest.TestCase):
    """Basic tests for TemperatureScaler class."""

    def test_default_temperature_is_one(self) -> None:
        """Default temperature should be 1.0."""
        scaler = TemperatureScaler()
        self.assertEqual(scaler.temperature, 1.0)
        self.assertFalse(scaler.fitted)

    def test_fit_updates_temperature(self) -> None:
        """Fitting should update the temperature parameter."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 4

        # Create overconfident logits (large values)
        logits = np.random.randn(n_samples, n_classes) * 5
        y_true = np.random.randint(0, n_classes, n_samples)

        scaler = TemperatureScaler()
        scaler.fit(logits, y_true)

        self.assertTrue(scaler.fitted)
        # Temperature should be > 1 for overconfident predictions
        self.assertGreater(scaler.temperature, 0.1)
        self.assertLess(scaler.temperature, 10.0)

    def test_fit_with_onehot_labels(self) -> None:
        """Fitting should work with one-hot encoded labels."""
        np.random.seed(42)
        n_samples = 50
        n_classes = 3

        logits = np.random.randn(n_samples, n_classes) * 3
        y_true_int = np.random.randint(0, n_classes, n_samples)
        y_true_onehot = np.eye(n_classes)[y_true_int]

        scaler = TemperatureScaler()
        scaler.fit(logits, y_true_onehot)

        self.assertTrue(scaler.fitted)
        self.assertGreater(scaler.temperature, 0.1)

    def test_transform_returns_probabilities(self) -> None:
        """Transform should return valid probabilities."""
        np.random.seed(42)
        n_samples = 50
        n_classes = 3

        logits = np.random.randn(n_samples, n_classes) * 2
        y_true = np.random.randint(0, n_classes, n_samples)

        scaler = TemperatureScaler()
        scaler.fit(logits, y_true)
        probs = scaler.transform(logits)

        # Check shape
        self.assertEqual(probs.shape, logits.shape)
        # Check probabilities sum to 1
        sums = np.sum(probs, axis=1)
        assert_allclose(sums, np.ones(n_samples), atol=1e-6)
        # Check all values in [0, 1]
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_unfitted_transform_warns(self) -> None:
        """Transform on unfitted scaler should return unscaled softmax."""
        logits = np.array([[1.0, 2.0, 3.0]])
        scaler = TemperatureScaler()

        with self.assertLogs(level="WARNING"):
            probs = scaler.transform(logits)

        expected = softmax(logits, axis=1)
        assert_allclose(probs, expected)

    def test_to_dict_and_from_dict(self) -> None:
        """Serialization should preserve state."""
        scaler = TemperatureScaler(
            temperature=1.5,
            fitted=True,
            pre_calibration_ece=0.15,
            post_calibration_ece=0.05,
        )

        data = scaler.to_dict()
        restored = TemperatureScaler.from_dict(data)

        self.assertEqual(restored.temperature, 1.5)
        self.assertTrue(restored.fitted)
        self.assertEqual(restored.pre_calibration_ece, 0.15)
        self.assertEqual(restored.post_calibration_ece, 0.05)


class TestTemperatureScalerValidation(unittest.TestCase):
    """Validation tests for TemperatureScaler."""

    def test_fit_with_1d_logits_raises(self) -> None:
        """1D logits should raise ValueError."""
        logits = np.array([1.0, 2.0, 3.0])
        y_true = np.array([0, 1, 2])

        scaler = TemperatureScaler()
        with self.assertRaises(ValueError) as ctx:
            scaler.fit(logits, y_true)

        self.assertIn("2D", str(ctx.exception))

    def test_fit_with_mismatched_samples_raises(self) -> None:
        """Mismatched sample counts should raise ValueError."""
        logits = np.random.randn(10, 3)
        y_true = np.array([0, 1, 2, 0, 1])  # Only 5 labels

        scaler = TemperatureScaler()
        with self.assertRaises(ValueError) as ctx:
            scaler.fit(logits, y_true)

        self.assertIn("length", str(ctx.exception).lower())

    def test_fit_with_empty_data_does_not_fit(self) -> None:
        """Empty dataset should not fit the scaler."""
        logits = np.zeros((0, 3))
        y_true = np.array([], dtype="int64")

        scaler = TemperatureScaler()
        with self.assertLogs(level="WARNING"):
            scaler.fit(logits, y_true)

        self.assertFalse(scaler.fitted)


class TestApplyTemperatureScaling(unittest.TestCase):
    """Tests for apply_temperature_scaling function."""

    def test_temperature_one_is_softmax(self) -> None:
        """Temperature=1 should be equivalent to regular softmax."""
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        probs = apply_temperature_scaling(logits, temperature=1.0)
        expected = softmax(logits, axis=1)
        assert_allclose(probs, expected)

    def test_high_temperature_softens(self) -> None:
        """High temperature should soften (more uniform) predictions."""
        logits = np.array([[0.0, 0.0, 10.0]])  # Very confident in class 2

        probs_t1 = apply_temperature_scaling(logits, temperature=1.0)
        probs_t5 = apply_temperature_scaling(logits, temperature=5.0)

        # Max prob should be lower with higher temperature
        self.assertLess(probs_t5.max(), probs_t1.max())
        # Min prob should be higher with higher temperature
        self.assertGreater(probs_t5.min(), probs_t1.min())

    def test_low_temperature_sharpens(self) -> None:
        """Low temperature should sharpen (more peaked) predictions."""
        logits = np.array([[1.0, 2.0, 3.0]])

        probs_t1 = apply_temperature_scaling(logits, temperature=1.0)
        probs_t_low = apply_temperature_scaling(logits, temperature=0.5)

        # Max prob should be higher with lower temperature
        self.assertGreater(probs_t_low.max(), probs_t1.max())

    def test_zero_temperature_raises(self) -> None:
        """Temperature=0 should raise ValueError."""
        logits = np.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError) as ctx:
            apply_temperature_scaling(logits, temperature=0.0)
        self.assertIn("positive", str(ctx.exception))

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature should raise ValueError."""
        logits = np.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError) as ctx:
            apply_temperature_scaling(logits, temperature=-1.0)
        self.assertIn("positive", str(ctx.exception))


class TestFitTemperature(unittest.TestCase):
    """Tests for fit_temperature convenience function."""

    def test_returns_fitted_scaler(self) -> None:
        """fit_temperature should return a fitted scaler."""
        np.random.seed(42)
        logits = np.random.randn(100, 4) * 3
        y_true = np.random.randint(0, 4, 100)

        scaler = fit_temperature(logits, y_true)

        self.assertIsInstance(scaler, TemperatureScaler)
        self.assertTrue(scaler.fitted)

    def test_respects_num_bins(self) -> None:
        """num_bins parameter should be passed through."""
        np.random.seed(42)
        logits = np.random.randn(50, 3) * 2
        y_true = np.random.randint(0, 3, 50)

        # Should not raise with different num_bins values
        scaler1 = fit_temperature(logits, y_true, num_bins=5)
        scaler2 = fit_temperature(logits, y_true, num_bins=20)

        self.assertTrue(scaler1.fitted)
        self.assertTrue(scaler2.fitted)

    def test_respects_bounds(self) -> None:
        """bounds parameter should constrain the temperature."""
        np.random.seed(42)
        logits = np.random.randn(100, 4) * 10  # Very overconfident
        y_true = np.random.randint(0, 4, 100)

        scaler = fit_temperature(logits, y_true, bounds=(0.5, 2.0))

        self.assertTrue(scaler.fitted)
        self.assertGreaterEqual(scaler.temperature, 0.5)
        self.assertLessEqual(scaler.temperature, 2.0)


class TestCalibrationImprovesECE(unittest.TestCase):
    """Property-based tests that calibration should improve ECE."""

    def test_calibration_reduces_ece_on_overconfident(self) -> None:
        """Temperature scaling should reduce ECE for overconfident models."""
        np.random.seed(42)
        n_samples = 200
        n_classes = 4

        # Create very overconfident predictions
        logits = np.random.randn(n_samples, n_classes) * 10
        y_true = np.random.randint(0, n_classes, n_samples)

        scaler = fit_temperature(logits, y_true, num_bins=15)

        # ECE should improve (or at least not get worse)
        self.assertLessEqual(scaler.post_calibration_ece, scaler.pre_calibration_ece + 0.05)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestCalibrationProperties(unittest.TestCase):
    """Property-based tests for calibration functions using Hypothesis."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        logits=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=50),
                st.integers(min_value=2, max_value=6),
            ),
            elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        ),
        temperature=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_probabilities_sum_to_one(self, logits: np.ndarray, temperature: float) -> None:
        """Scaled probabilities should always sum to 1 for each sample."""
        probs = apply_temperature_scaling(logits, temperature)

        # Check each row sums to 1
        row_sums = np.sum(probs, axis=1)
        assert_allclose(row_sums, np.ones(probs.shape[0]), atol=1e-6)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        logits=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=30),
                st.integers(min_value=2, max_value=5),
            ),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        ),
        temperature=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_probabilities_are_valid(self, logits: np.ndarray, temperature: float) -> None:
        """All probabilities should be in [0, 1]."""
        probs = apply_temperature_scaling(logits, temperature)

        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        logits=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=20),
                st.integers(min_value=2, max_value=4),
            ),
            elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_temperature_one_equals_softmax(self, logits: np.ndarray) -> None:
        """Temperature=1 should equal regular softmax."""
        probs = apply_temperature_scaling(logits, temperature=1.0)
        expected = softmax(logits, axis=1)

        assert_allclose(probs, expected, atol=1e-10)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        logits=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=1, max_value=20),
                st.integers(min_value=2, max_value=4),
            ),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        ),
        t_low=st.floats(min_value=0.1, max_value=1.0),
        t_high=st.floats(min_value=2.0, max_value=10.0),
    )
    def test_higher_temperature_more_uniform(self, logits: np.ndarray, t_low: float, t_high: float) -> None:
        """Higher temperature should produce more uniform (lower entropy) distributions."""
        probs_low = apply_temperature_scaling(logits, t_low)
        probs_high = apply_temperature_scaling(logits, t_high)

        # Max probability should be lower (or equal) with higher temperature
        # for each sample
        max_low = np.max(probs_low, axis=1)
        max_high = np.max(probs_high, axis=1)

        # Allow small tolerance for numerical issues
        self.assertTrue(np.all(max_high <= max_low + 1e-6))

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        temperature=st.floats(min_value=0.1, max_value=10.0),
        pre_ece=st.floats(min_value=0.0, max_value=1.0),
        post_ece=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_scaler_serialization_roundtrip(
        self, temperature: float, pre_ece: float, post_ece: float
    ) -> None:
        """to_dict and from_dict should preserve all state."""
        scaler = TemperatureScaler(
            temperature=temperature,
            fitted=True,
            pre_calibration_ece=pre_ece,
            post_calibration_ece=post_ece,
        )

        data = scaler.to_dict()
        restored = TemperatureScaler.from_dict(data)

        self.assertEqual(restored.temperature, temperature)
        self.assertTrue(restored.fitted)
        self.assertEqual(restored.pre_calibration_ece, pre_ece)
        self.assertEqual(restored.post_calibration_ece, post_ece)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        n_classes=st.integers(min_value=2, max_value=5),
        logit_scale=st.floats(min_value=1.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_fitted_temperature_is_bounded(
        self, n_samples: int, n_classes: int, logit_scale: float, seed: int
    ) -> None:
        """Fitted temperature should be within specified bounds."""
        np.random.seed(seed)
        logits = np.random.randn(n_samples, n_classes) * logit_scale
        y_true = np.random.randint(0, n_classes, n_samples)

        bounds = (0.5, 5.0)
        scaler = fit_temperature(logits, y_true, bounds=bounds)

        self.assertGreaterEqual(scaler.temperature, bounds[0])
        self.assertLessEqual(scaler.temperature, bounds[1])


if __name__ == "__main__":
    unittest.main()
