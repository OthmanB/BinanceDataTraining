"""Calibration analysis and post-hoc calibration utilities.

This module provides:
- compute_calibration_metrics: Compute ECE, Brier score, and reliability curves
- probs_to_logits_proxy: Convert probabilities to logit-like values
- TemperatureScaler: Post-hoc temperature scaling for probability calibration
- apply_temperature_scaling: Apply learned temperature to logits/probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, TYPE_CHECKING
import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


logger = logging.getLogger(__name__)


def compute_calibration_metrics(y_true: Any, y_prob: Any, *, num_bins: int) -> Dict[str, Any]:
    """Compute basic multi-class calibration metrics.

    Parameters
    ----------
    y_true:
        Array-like of shape (n_samples, n_classes) with one-hot encoded
        ground-truth labels.
    y_prob:
        Array-like of shape (n_samples, n_classes) with predicted
        probabilities.
    num_bins:
        Number of bins to use for reliability curve / ECE computation.
    """

    y_true_arr = np.asarray(y_true, dtype="float64")
    y_prob_arr = np.asarray(y_prob, dtype="float64")

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            "y_true and y_prob must have the same shape for calibration analysis, "
            f"got {y_true_arr.shape} and {y_prob_arr.shape}",
        )

    if y_true_arr.ndim != 2:
        raise ValueError("y_true and y_prob must be 2D arrays of shape (n_samples, n_classes)")

    if num_bins <= 0:
        raise ValueError("num_bins must be a positive integer for calibration analysis")

    n_samples, n_classes = y_prob_arr.shape
    if n_samples == 0 or n_classes == 0:
        return {
            "brier_score": 0.0,
            "ece": 0.0,
            "bin_edges": np.asarray([], dtype="float64"),
            "bin_confidence": np.asarray([], dtype="float64"),
            "bin_accuracy": np.asarray([], dtype="float64"),
            "bin_count": np.asarray([], dtype="int64"),
        }

    diff = y_prob_arr - y_true_arr
    brier_per_sample = np.sum(diff * diff, axis=1)
    brier_score = float(np.mean(brier_per_sample))

    probs_flat = y_prob_arr.ravel()
    true_flat = y_true_arr.ravel()

    edges = np.linspace(0.0, 1.0, num_bins + 1, dtype="float64")
    bin_confidence = np.zeros(num_bins, dtype="float64")
    bin_accuracy = np.zeros(num_bins, dtype="float64")
    bin_count = np.zeros(num_bins, dtype="int64")

    for i in range(num_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == num_bins - 1:
            mask = (probs_flat >= left) & (probs_flat <= right)
        else:
            mask = (probs_flat >= left) & (probs_flat < right)

        if not mask.any():
            bin_confidence[i] = 0.0
            bin_accuracy[i] = 0.0
            bin_count[i] = 0
            continue

        probs_bin = probs_flat[mask]
        true_bin = true_flat[mask]

        bin_confidence[i] = float(np.mean(probs_bin))
        bin_accuracy[i] = float(np.mean(true_bin))
        bin_count[i] = int(mask.sum())

    total = float(bin_count.sum())
    if total > 0.0:
        abs_diff = np.abs(bin_confidence - bin_accuracy)
        weights = bin_count.astype("float64") / total
        ece = float(np.sum(abs_diff * weights))
    else:
        ece = 0.0

    return {
        "brier_score": brier_score,
        "ece": ece,
        "bin_edges": edges,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
    }


def probs_to_logits_proxy(probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Convert probabilities to logit-like values suitable for temperature scaling.

    This is useful when the model outputs softmax probabilities instead of logits.
    The returned values preserve class ordering and can be used with temperature
    scaling because softmax(log(p) / T) yields a valid calibrated distribution.

    Parameters
    ----------
    probs : np.ndarray
        Array of probabilities with shape (n_samples, n_classes).
    eps : float
        Clipping epsilon to avoid log(0). Must be in (0, 0.5).

    Returns
    -------
    np.ndarray
        Logit-like array of shape (n_samples, n_classes).
    """
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")

    probs_arr = np.asarray(probs, dtype="float64")
    if probs_arr.ndim != 2:
        raise ValueError(
            "probs must be a 2D array of shape (n_samples, n_classes) for logits proxy conversion"
        )

    probs_arr = np.clip(probs_arr, eps, 1.0 - eps)
    row_sums = probs_arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    probs_arr = probs_arr / row_sums

    return np.log(probs_arr)


def logits_to_calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    return apply_temperature_scaling(logits, temperature)


__all__ = [
    "compute_calibration_metrics",
    "probs_to_logits_proxy",
    "TemperatureScaler",
    "apply_temperature_scaling",
    "logits_to_calibrated_probs",
    "fit_temperature",
]


@dataclass
class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration.

    Temperature scaling divides logits by a learned temperature parameter
    before softmax to improve calibration. A temperature > 1 softens
    predictions (reduces overconfidence), while temperature < 1 sharpens them.

    Attributes
    ----------
    temperature : float
        The learned temperature parameter. Default is 1.0 (no scaling).
    fitted : bool
        Whether the temperature has been fitted on validation data.
    pre_calibration_ece : float
        ECE before calibration (computed during fitting).
    post_calibration_ece : float
        ECE after calibration (computed during fitting).
    """

    temperature: float = 1.0
    fitted: bool = False
    pre_calibration_ece: float = 0.0
    post_calibration_ece: float = 0.0

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        *,
        num_bins: int = 15,
        bounds: Tuple[float, float] = (0.1, 10.0),
    ) -> "TemperatureScaler":
        """Fit the temperature parameter using validation data.

        Parameters
        ----------
        logits : np.ndarray
            Pre-softmax logits of shape (n_samples, n_classes).
        y_true : np.ndarray
            Ground truth labels, either integer class indices of shape (n_samples,)
            or one-hot encoded of shape (n_samples, n_classes).
        num_bins : int
            Number of bins for ECE computation during optimization.
        bounds : Tuple[float, float]
            Search bounds for temperature parameter.

        Returns
        -------
        TemperatureScaler
            Self, with fitted temperature.
        """
        logits_arr = np.asarray(logits, dtype="float64")
        y_true_arr = np.asarray(y_true)

        if logits_arr.ndim != 2:
            raise ValueError(
                f"logits must be 2D array of shape (n_samples, n_classes), got {logits_arr.ndim}D"
            )

        n_samples, n_classes = logits_arr.shape

        if n_samples == 0:
            logger.warning("Cannot fit temperature on empty dataset")
            self.fitted = False
            return self

        # Convert y_true to integer labels if one-hot encoded
        if y_true_arr.ndim == 2:
            y_labels = np.argmax(y_true_arr, axis=1)
        elif y_true_arr.ndim == 1:
            y_labels = y_true_arr.astype("int64")
        else:
            raise ValueError(f"y_true must be 1D or 2D array, got {y_true_arr.ndim}D")

        if y_labels.shape[0] != n_samples:
            raise ValueError(
                f"y_true length {y_labels.shape[0]} does not match logits samples {n_samples}"
            )

        # Compute pre-calibration ECE
        probs_before = softmax(logits_arr, axis=1)
        y_onehot = np.eye(n_classes, dtype="float64")[y_labels]
        pre_metrics = compute_calibration_metrics(y_onehot, probs_before, num_bins=num_bins)
        self.pre_calibration_ece = pre_metrics["ece"]

        def nll_loss(temperature: float) -> float:
            """Negative log-likelihood loss for temperature optimization."""
            scaled_logits = logits_arr / temperature
            probs = softmax(scaled_logits, axis=1)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
            # NLL = -sum(y_true * log(p))
            log_probs = np.log(probs)
            # Select the log prob for the true class
            nll = -np.mean(log_probs[np.arange(n_samples), y_labels])
            return float(nll)

        result = minimize_scalar(
            nll_loss,
            bounds=bounds,
            method="bounded",
        )

        # OptimizeResult has .x attribute for the optimal value
        optimal_temp = getattr(result, "x", 1.0)
        self.temperature = float(optimal_temp)
        self.fitted = True

        # Compute post-calibration ECE
        probs_after = softmax(logits_arr / self.temperature, axis=1)
        post_metrics = compute_calibration_metrics(y_onehot, probs_after, num_bins=num_bins)
        self.post_calibration_ece = post_metrics["ece"]

        logger.info(
            "Temperature scaling fitted: temperature=%.4f, pre_ECE=%.4f, post_ECE=%.4f",
            self.temperature,
            self.pre_calibration_ece,
            self.post_calibration_ece,
        )

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits and return calibrated probabilities.

        Parameters
        ----------
        logits : np.ndarray
            Pre-softmax logits of shape (n_samples, n_classes).

        Returns
        -------
        np.ndarray
            Calibrated probabilities of shape (n_samples, n_classes).
        """
        if not self.fitted:
            logger.warning("Temperature scaler not fitted, returning unscaled softmax")
            return softmax(np.asarray(logits, dtype="float64"), axis=1)

        logits_arr = np.asarray(logits, dtype="float64")
        scaled_logits = logits_arr / self.temperature
        return softmax(scaled_logits, axis=1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the scaler to a dictionary."""
        return {
            "temperature": self.temperature,
            "fitted": self.fitted,
            "pre_calibration_ece": self.pre_calibration_ece,
            "post_calibration_ece": self.post_calibration_ece,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureScaler":
        """Deserialize the scaler from a dictionary."""
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            fitted=bool(data.get("fitted", False)),
            pre_calibration_ece=float(data.get("pre_calibration_ece", 0.0)),
            post_calibration_ece=float(data.get("post_calibration_ece", 0.0)),
        )


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    num_bins: int = 15,
    bounds: Tuple[float, float] = (0.1, 10.0),
) -> TemperatureScaler:
    """Convenience function to fit temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Pre-softmax logits of shape (n_samples, n_classes).
    y_true : np.ndarray
        Ground truth labels, either integer class indices or one-hot encoded.
    num_bins : int
        Number of bins for ECE computation.
    bounds : Tuple[float, float]
        Search bounds for temperature.

    Returns
    -------
    TemperatureScaler
        Fitted temperature scaler.
    """
    scaler = TemperatureScaler()
    scaler.fit(logits, y_true, num_bins=num_bins, bounds=bounds)
    return scaler


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to logits.

    Parameters
    ----------
    logits : np.ndarray
        Pre-softmax logits of shape (n_samples, n_classes). Log-probability
        proxies from probs_to_logits_proxy are also acceptable.
    temperature : float
        Temperature parameter (must be positive).

    Returns
    -------
    np.ndarray
        Calibrated probabilities of shape (n_samples, n_classes).

    Raises
    ------
    ValueError
        If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    logits_arr = np.asarray(logits, dtype="float64")
    scaled_logits = logits_arr / temperature
    return softmax(scaled_logits, axis=1)
