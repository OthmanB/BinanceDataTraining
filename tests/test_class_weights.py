"""Unit tests for class weight computation.

Tests cover:
- Balanced distributions
- Imbalanced distributions
- Edge case: class with zero samples
- Input validation
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from training.class_weights import compute_class_weights


class TestComputeClassWeightsBasic:
    """Basic unit tests for compute_class_weights."""

    def test_balanced_distribution(self) -> None:
        """Weights should be approximately 1.0 for balanced classes."""
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        weights = compute_class_weights(labels, num_classes=4)

        assert len(weights) == 4
        for c in range(4):
            assert c in weights
            # With 8 samples and 4 classes, each class has 2 samples
            # weight = 8 / (4 * 2) = 1.0
            assert weights[c] == pytest.approx(1.0)

    def test_imbalanced_distribution(self) -> None:
        """Rare classes should have higher weights."""
        # Class 0: 10 samples, Class 1: 2 samples
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        weights = compute_class_weights(labels, num_classes=2)

        assert len(weights) == 2
        # n_samples = 12
        # weight[0] = 12 / (2 * 10) = 0.6
        # weight[1] = 12 / (2 * 2) = 3.0
        assert weights[0] == pytest.approx(0.6)
        assert weights[1] == pytest.approx(3.0)
        # Rare class should have higher weight
        assert weights[1] > weights[0]

    def test_zero_sample_class(self) -> None:
        """Class with zero samples should receive max weight from other classes."""
        # Class 0: 4 samples, Class 1: 2 samples, Class 2: 0 samples
        labels = np.array([0, 0, 0, 0, 1, 1])
        weights = compute_class_weights(labels, num_classes=3)

        assert len(weights) == 3
        # n_samples = 6
        # weight[0] = 6 / (3 * 4) = 0.5
        # weight[1] = 6 / (3 * 2) = 1.0
        # weight[2] = max(0.5, 1.0) = 1.0 (max weight)
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(1.0)
        assert weights[2] == pytest.approx(1.0)  # max weight from other classes

    def test_single_class_only(self) -> None:
        """When only one class is present, others get max weight."""
        labels = np.array([0, 0, 0, 0])
        weights = compute_class_weights(labels, num_classes=3)

        assert len(weights) == 3
        # n_samples = 4
        # weight[0] = 4 / (3 * 4) = 0.333...
        # weight[1] = weight[2] = 0.333... (max weight from non-empty)
        expected_w0 = 4.0 / (3.0 * 4.0)
        assert weights[0] == pytest.approx(expected_w0)
        assert weights[1] == pytest.approx(expected_w0)
        assert weights[2] == pytest.approx(expected_w0)

    def test_single_sample(self) -> None:
        """Single sample should still produce valid weights."""
        labels = np.array([1])
        weights = compute_class_weights(labels, num_classes=3)

        assert len(weights) == 3
        # n_samples = 1, class 1 has count=1
        # weight[1] = 1 / (3 * 1) = 0.333...
        expected = 1.0 / 3.0
        assert weights[1] == pytest.approx(expected)
        # Other classes get max weight
        assert weights[0] == pytest.approx(expected)
        assert weights[2] == pytest.approx(expected)


class TestComputeClassWeightsValidation:
    """Input validation tests for compute_class_weights."""

    def test_empty_labels_raises(self) -> None:
        """Empty labels array should raise ValueError."""
        labels = np.array([], dtype=np.int64)
        with pytest.raises(ValueError, match="labels array is empty"):
            compute_class_weights(labels, num_classes=3)

    def test_non_1d_labels_raises(self) -> None:
        """Non-1D labels array should raise ValueError."""
        labels = np.array([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="must be a 1D array"):
            compute_class_weights(labels, num_classes=4)

    def test_negative_labels_raises(self) -> None:
        """Labels with negative values should raise ValueError."""
        labels = np.array([0, 1, -1, 2])
        with pytest.raises(ValueError, match="values outside"):
            compute_class_weights(labels, num_classes=3)

    def test_labels_exceed_num_classes_raises(self) -> None:
        """Labels exceeding num_classes should raise ValueError."""
        labels = np.array([0, 1, 5])
        with pytest.raises(ValueError, match="values outside"):
            compute_class_weights(labels, num_classes=3)


class TestComputeClassWeightsProperties:
    """Property-based tests for compute_class_weights."""

    @given(
        st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=1,
            max_size=100,
        ),
    )
    def test_weights_are_positive(self, label_list: list) -> None:
        """All weights should be positive."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 5
        weights = compute_class_weights(labels, num_classes)

        for c in range(num_classes):
            assert weights[c] > 0, f"Weight for class {c} should be positive"

    @given(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=1,
            max_size=100,
        ),
    )
    def test_all_classes_have_weights(self, label_list: list) -> None:
        """All classes should have a weight entry."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 4
        weights = compute_class_weights(labels, num_classes)

        assert len(weights) == num_classes
        for c in range(num_classes):
            assert c in weights

    @given(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=10,
            max_size=100,
        ),
    )
    def test_rare_class_higher_weight(self, label_list: list) -> None:
        """Classes with fewer samples should have higher or equal weights."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 3
        weights = compute_class_weights(labels, num_classes)
        counts = np.bincount(labels, minlength=num_classes)

        for c1 in range(num_classes):
            for c2 in range(num_classes):
                if counts[c1] > 0 and counts[c2] > 0:
                    if counts[c1] < counts[c2]:
                        assert weights[c1] >= weights[c2], (
                            f"Class {c1} (count={counts[c1]}) should have "
                            f">= weight than class {c2} (count={counts[c2]})"
                        )

    @given(
        st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=1,
            max_size=200,
        ),
    )
    def test_weighted_sum_equals_n_samples(self, label_list: list) -> None:
        """Sum of (weight[c] * count[c]) should equal n_samples for non-empty classes."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 5
        weights = compute_class_weights(labels, num_classes)
        counts = np.bincount(labels, minlength=num_classes)

        # For non-empty classes: weight[c] * count[c] = n_samples / num_classes
        # Sum over non-empty: sum(n_samples / num_classes) = n_samples * (k / num_classes)
        # where k = number of non-empty classes
        n_samples = len(label_list)
        weighted_sum = sum(weights[c] * counts[c] for c in range(num_classes) if counts[c] > 0)

        # Each non-empty class contributes n_samples / num_classes
        non_empty_classes = sum(1 for c in counts if c > 0)
        expected = n_samples * non_empty_classes / num_classes

        assert weighted_sum == pytest.approx(expected, rel=1e-9)

    @given(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=5,
            max_size=50,
        ),
        st.random_module(),
    )
    def test_idempotency(self, label_list: list, _random: object) -> None:
        """Same input always produces same weights."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 4

        weights1 = compute_class_weights(labels, num_classes)
        weights2 = compute_class_weights(labels, num_classes)

        for c in range(num_classes):
            assert weights1[c] == weights2[c], f"Weights differ for class {c}"

    @given(
        st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=1,
            max_size=100,
        ),
    )
    def test_weight_formula_correctness(self, label_list: list) -> None:
        """Verify weight formula: weight[c] = n_samples / (num_classes * count[c])."""
        labels = np.array(label_list, dtype=np.int64)
        num_classes = 6
        weights = compute_class_weights(labels, num_classes)
        counts = np.bincount(labels, minlength=num_classes)
        n_samples = len(label_list)

        for c in range(num_classes):
            if counts[c] > 0:
                expected = n_samples / (num_classes * counts[c])
                assert weights[c] == pytest.approx(expected, rel=1e-9), (
                    f"Weight for class {c} incorrect: expected {expected}, got {weights[c]}"
                )
