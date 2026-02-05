"""Property-based tests for the Normalizer class using hypothesis.

These tests verify invariants that should hold for all valid inputs,
catching edge cases that unit tests might miss.
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
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from preprocessing.normalizer import Normalizer, create_normalizer_from_config


def _make_config(method: str) -> dict:
    """Create a minimal valid configuration dictionary for testing."""
    return {
        "preprocessing": {
            "normalization": {
                "method": method,
                "per_asset": False,
                "fit_on_train_only": True,
            }
        }
    }


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestNormalizerProperties(unittest.TestCase):
    """Property-based tests for Normalizer."""

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_shape_preservation_min_max(self, data: np.ndarray) -> None:
        """Property: output shape always equals input shape for min_max."""
        normalizer = Normalizer("min_max")
        output = normalizer.fit_transform(data)
        self.assertEqual(output.shape, data.shape)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_shape_preservation_standard(self, data: np.ndarray) -> None:
        """Property: output shape always equals input shape for standard."""
        normalizer = Normalizer("standard")
        output = normalizer.fit_transform(data)
        self.assertEqual(output.shape, data.shape)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_shape_preservation_robust(self, data: np.ndarray) -> None:
        """Property: output shape always equals input shape for robust."""
        normalizer = Normalizer("robust")
        output = normalizer.fit_transform(data)
        self.assertEqual(output.shape, data.shape)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_min_max_output_bounds(self, data: np.ndarray) -> None:
        """Property: min_max normalization produces output in [0, 1] range."""
        normalizer = Normalizer("min_max")
        output = normalizer.fit_transform(data)

        # Allow small tolerance for floating-point precision
        self.assertGreaterEqual(output.min(), -1e-6)
        self.assertLessEqual(output.max(), 1.0 + 1e-6)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=100),
                st.integers(min_value=1, max_value=5),
            ),
            elements=st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    def test_standard_zero_mean(self, data: np.ndarray) -> None:
        """Property: standard normalization produces output with mean ≈ 0."""
        # Skip if all values are identical (constant features)
        assume(data.std() > 1e-10)

        normalizer = Normalizer("standard")
        output = normalizer.fit_transform(data)

        # Mean should be close to zero (allow tolerance for floating-point)
        output_flat = output.reshape(output.shape[0], -1)
        for col in range(output_flat.shape[1]):
            col_std = data.reshape(data.shape[0], -1)[:, col].std()
            if col_std > 1e-10:
                self.assertAlmostEqual(output_flat[:, col].mean(), 0.0, places=4)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_fit_transform_equivalence(self, data: np.ndarray) -> None:
        """Property: fit_transform(x) == fit(x).transform(x)."""
        normalizer1 = Normalizer("min_max")
        output1 = normalizer1.fit_transform(data)

        normalizer2 = Normalizer("min_max")
        normalizer2.fit(data)
        output2 = normalizer2.transform(data)

        np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-5)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(min_value=2, max_value=50),
                st.integers(min_value=1, max_value=10),
            ),
            elements=st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    def test_transform_idempotent_on_same_data(self, data: np.ndarray) -> None:
        """Property: transform(x) on same data after fit gives consistent results."""
        normalizer = Normalizer("min_max")
        normalizer.fit(data)

        output1 = normalizer.transform(data)
        output2 = normalizer.transform(data)

        np.testing.assert_array_equal(output1, output2)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(method=st.sampled_from(["min_max", "standard", "robust"]))
    def test_is_fitted_property(self, method: str) -> None:
        """Property: is_fitted is False before fit, True after."""
        normalizer = Normalizer(method)
        self.assertFalse(normalizer.is_fitted)

        data = np.random.randn(10, 5).astype("float32")
        normalizer.fit(data)
        self.assertTrue(normalizer.is_fitted)

    @settings(max_examples=_MAX_EXAMPLES, suppress_health_check=[HealthCheck.too_slow])
    @given(method=st.sampled_from(["min_max", "standard", "robust"]))
    def test_create_from_config(self, method: str) -> None:
        """Property: create_normalizer_from_config returns correct method."""
        config = _make_config(method)
        normalizer = create_normalizer_from_config(config)
        self.assertEqual(normalizer.method, method)
        self.assertFalse(normalizer.is_fitted)


if __name__ == "__main__":
    unittest.main()
