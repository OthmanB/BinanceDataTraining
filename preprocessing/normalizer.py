"""Input normalization for order book features.

This module implements the Normalizer class that supports three scaling methods
as defined in the YAML configuration under preprocessing.normalization:
- min_max: scales features to [0, 1] range
- standard: zero-mean, unit-variance scaling
- robust: median-based scaling using IQR (robust to outliers)

All parameters must be provided via configuration; no implicit defaults are used.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


class Normalizer:
    """Feature normalizer supporting min_max, standard, and robust scaling.

    This class follows the fit/transform pattern:
    - fit(): compute statistics from training data only
    - transform(): apply the fitted normalization to any data
    - fit_transform(): convenience method combining both

    All configuration must come from YAML; this class does not define defaults.
    """

    SUPPORTED_METHODS = ("min_max", "standard", "robust")

    def __init__(self, method: str) -> None:
        """Initialize normalizer with the specified method.

        Parameters
        ----------
        method:
            Scaling method. Must be one of: 'min_max', 'standard', 'robust'.
            Raises ValueError if method is not supported.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported normalization method: {method!r}. "
                f"Must be one of: {self.SUPPORTED_METHODS}",
            )
        self._method = method
        self._is_fitted = False

        # Statistics populated by fit()
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._median: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None

    @property
    def method(self) -> str:
        """Return the normalization method."""
        return self._method

    @property
    def is_fitted(self) -> bool:
        """Return True if the normalizer has been fitted."""
        return self._is_fitted

    def fit(self, X: np.ndarray) -> "Normalizer":
        """Compute normalization statistics from training data.

        Parameters
        ----------
        X:
            Training data array of shape (n_samples, ...).
            Statistics are computed over the first axis (samples).

        Returns
        -------
        self:
            The fitted normalizer instance.
        """
        if X.size == 0:
            raise ValueError("Cannot fit normalizer on empty array")

        # Flatten all dimensions except the first (samples) for statistics
        X_flat = X.reshape(X.shape[0], -1)

        if self._method == "min_max":
            self._min = np.min(X_flat, axis=0)
            self._max = np.max(X_flat, axis=0)
        elif self._method == "standard":
            self._mean = np.mean(X_flat, axis=0)
            self._std = np.std(X_flat, axis=0)
        elif self._method == "robust":
            self._median = np.median(X_flat, axis=0)
            q75 = np.percentile(X_flat, 75, axis=0)
            q25 = np.percentile(X_flat, 25, axis=0)
            self._iqr = q75 - q25

        self._is_fitted = True

        logger.info(
            "Normalizer fitted with method=%s on data shape=%s",
            self._method,
            X.shape,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to data using fitted statistics.

        Parameters
        ----------
        X:
            Data array to transform. Must have the same feature dimensions
            as the data used for fitting.

        Returns
        -------
        X_normalized:
            Normalized data array with the same shape as input.

        Raises
        ------
        RuntimeError:
            If the normalizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before calling transform()")

        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)

        if self._method == "min_max":
            # Avoid division by zero for constant features
            denom = self._max - self._min
            denom = np.where(denom == 0, 1.0, denom)
            X_normalized = (X_flat - self._min) / denom

        elif self._method == "standard":
            # Avoid division by zero for constant features
            std = np.where(self._std == 0, 1.0, self._std)
            X_normalized = (X_flat - self._mean) / std

        elif self._method == "robust":
            # Avoid division by zero for constant features
            iqr = np.where(self._iqr == 0, 1.0, self._iqr)
            X_normalized = (X_flat - self._median) / iqr

        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown normalization method: {self._method}")

        return X_normalized.reshape(original_shape).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the normalizer and transform data in one step.

        Parameters
        ----------
        X:
            Training data array to fit and transform.

        Returns
        -------
        X_normalized:
            Normalized data array with the same shape as input.
        """
        return self.fit(X).transform(X)

    def get_statistics(self) -> Dict[str, Optional[np.ndarray]]:
        """Return the fitted statistics as a dictionary.

        Returns
        -------
        stats:
            Dictionary containing the statistics used for normalization.
            Keys depend on the method used.
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before calling get_statistics()")

        if self._method == "min_max":
            return {"min": self._min, "max": self._max}
        elif self._method == "standard":
            return {"mean": self._mean, "std": self._std}
        elif self._method == "robust":
            return {"median": self._median, "iqr": self._iqr}
        else:
            return {}


def create_normalizer_from_config(config: Dict[str, Any]) -> Normalizer:
    """Create a Normalizer instance from YAML configuration.

    Parameters
    ----------
    config:
        Full configuration dictionary. Expects preprocessing.normalization.method
        to be defined.

    Returns
    -------
    normalizer:
        Configured Normalizer instance (not yet fitted).

    Raises
    ------
    KeyError:
        If required configuration keys are missing.
    ValueError:
        If configuration values are invalid.
    """
    preprocessing_cfg = config["preprocessing"]
    normalization_cfg = preprocessing_cfg["normalization"]

    method = str(normalization_cfg["method"])

    logger.info(
        "Creating normalizer from config: method=%s, per_asset=%s, fit_on_train_only=%s",
        method,
        normalization_cfg.get("per_asset"),
        normalization_cfg.get("fit_on_train_only"),
    )

    return Normalizer(method=method)


__all__ = ["Normalizer", "create_normalizer_from_config"]
