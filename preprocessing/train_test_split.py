"""Chronological train/validation/test split utilities.

These functions operate on sample counts and ratios provided by the YAML
configuration and return index lists for each split.
"""

from typing import List, Tuple
import logging


logger = logging.getLogger(__name__)


def chronological_split_indices(
    n_samples: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Compute chronological train/validation/test index splits.

    The ratios must sum to 1.0 (within a small numerical tolerance).
    """

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")

    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * validation_ratio)
    if val_end > n_samples:
        val_end = n_samples
    test_end = n_samples

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, test_end))

    logger.info(
        "Chronological split computed: n_samples=%d, train=%d, val=%d, test=%d",
        n_samples,
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    return train_indices, val_indices, test_indices


__all__ = ["chronological_split_indices"]
