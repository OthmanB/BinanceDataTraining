"""Class weight computation for handling imbalanced labels.

This module computes inverse-frequency class weights from training labels,
following the formula: weight[c] = n_samples / (num_classes * count[c]).

All parameters must come from configuration; no implicit defaults are used.
"""

from __future__ import annotations

from typing import Dict
import logging

import numpy as np


logger = logging.getLogger(__name__)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute inverse-frequency class weights from label array.

    Parameters
    ----------
    labels:
        1D array of integer class labels in range [0, num_classes-1].
    num_classes:
        Total number of classes (must match model output configuration).

    Returns
    -------
    Dict[int, float]:
        Mapping from class index to weight. Format compatible with Keras
        class_weight parameter.

    Raises
    ------
    ValueError:
        If labels is not 1D or contains values outside [0, num_classes-1].

    Notes
    -----
    - Weight formula: weight[c] = n_samples / (num_classes * count[c])
    - For classes with zero samples, the maximum weight from other classes
      is assigned to avoid division by zero and ensure rare classes are
      not ignored.
    """
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1D array; got ndim={labels.ndim}, shape={labels.shape!r}",
        )

    if labels.size == 0:
        raise ValueError("labels array is empty; cannot compute class weights")

    label_min = int(labels.min())
    label_max = int(labels.max())

    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            f"labels contain values outside [0, num_classes-1]; "
            f"got min={label_min}, max={label_max}, num_classes={num_classes}",
        )

    n_samples = labels.shape[0]
    counts = np.bincount(labels, minlength=num_classes)

    weights: Dict[int, float] = {}
    non_zero_weights: list[float] = []

    for c in range(num_classes):
        count_c = int(counts[c])
        if count_c > 0:
            weight_c = float(n_samples) / (float(num_classes) * float(count_c))
            weights[c] = weight_c
            non_zero_weights.append(weight_c)

    # Handle classes with zero samples: assign max weight from other classes
    if non_zero_weights:
        max_weight = max(non_zero_weights)
    else:
        # Edge case: all classes have zero samples (impossible with non-empty labels)
        max_weight = 1.0

    zero_sample_classes = []
    for c in range(num_classes):
        if c not in weights:
            weights[c] = max_weight
            zero_sample_classes.append(c)

    if zero_sample_classes:
        logger.warning(
            "Classes with zero samples assigned max weight: classes=%s, max_weight=%s",
            zero_sample_classes,
            max_weight,
        )

    logger.info(
        "Computed class weights: num_classes=%s, n_samples=%s, weights=%s",
        num_classes,
        n_samples,
        {c: round(w, 4) for c, w in sorted(weights.items())},
    )

    return weights


__all__ = ["compute_class_weights"]
