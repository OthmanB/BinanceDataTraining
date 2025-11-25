"""Dataset hashing and NPZ caching utilities for the training pipeline."""

from typing import Any, Dict, List, Optional
import hashlib
import logging
import os

import numpy as np


logger = logging.getLogger(__name__)


def _update_hash_with_array(hasher: "hashlib._Hash", array: np.ndarray, name: str) -> None:
    """Update a hash object with array metadata and contents."""

    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.tobytes(order="C"))


def compute_dataset_hash(
    x_train: np.ndarray,
    y_train: List[np.ndarray],
    x_val: Optional[np.ndarray],
    y_val: Optional[List[np.ndarray]],
) -> str:
    """Compute a deterministic hash for the in-memory training dataset.

    The hash incorporates shapes, dtypes, and raw bytes of all arrays that
    participate in training (train and validation splits).
    """

    hasher = hashlib.sha256()

    _update_hash_with_array(hasher, x_train, "x_train")
    for idx, y_arr in enumerate(y_train):
        _update_hash_with_array(hasher, y_arr, f"y_train_{idx}")

    if x_val is not None and y_val is not None:
        _update_hash_with_array(hasher, x_val, "x_val")
        for idx, y_arr in enumerate(y_val):
            _update_hash_with_array(hasher, y_arr, f"y_val_{idx}")

    return hasher.hexdigest()


def cache_dataset_to_npz(
    config: Dict[str, Any],
    dataset_hash: str,
    x_train: np.ndarray,
    y_train: List[np.ndarray],
    x_val: Optional[np.ndarray],
    y_val: Optional[List[np.ndarray]],
) -> Optional[str]:
    """Optionally cache the dataset to an NPZ file based on configuration.

    Returns the path to the NPZ file if caching is enabled and succeeds,
    otherwise returns None.
    """

    training_cfg = config["training"]
    cache_cfg = training_cfg["dataset_cache"]

    enabled = bool(cache_cfg["enabled"])
    if not enabled:
        return None

    directory = str(cache_cfg["directory"])
    filename_pattern = str(cache_cfg["filename_pattern"])
    version = str(cache_cfg["version"])

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    model_cfg = config["model"]
    architecture_name = str(model_cfg["architecture"])

    filename = filename_pattern.format(
        asset=target_asset,
        model=architecture_name,
        dataset_version=version,
        dataset_hash=dataset_hash,
    )

    os.makedirs(directory, exist_ok=True)
    npz_path = os.path.join(directory, filename)

    x_val_array: np.ndarray
    if x_val is None:
        x_val_array = np.zeros((0,), dtype="float32")
    else:
        x_val_array = x_val

    if y_val is None:
        y_up_val = np.zeros((0,), dtype="float32")
        y_down_val = np.zeros((0,), dtype="float32")
    else:
        if len(y_val) != 2:
            raise ValueError("Expected y_val to contain exactly two arrays: [y_up_val, y_down_val]")
        y_up_val = y_val[0]
        y_down_val = y_val[1]

    if len(y_train) != 2:
        raise ValueError("Expected y_train to contain exactly two arrays: [y_up_train, y_down_train]")

    y_up_train = y_train[0]
    y_down_train = y_train[1]

    try:
        np.savez_compressed(
            npz_path,
            x_train=x_train,
            y_up_train=y_up_train,
            y_down_train=y_down_train,
            x_val=x_val_array,
            y_up_val=y_up_val,
            y_down_val=y_down_val,
            dataset_hash=dataset_hash,
            dataset_version=version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cache dataset to NPZ file %s: %s", npz_path, exc)
        return None

    logger.info("Cached training dataset to NPZ file: %s", npz_path)
    return npz_path


__all__ = ["compute_dataset_hash", "cache_dataset_to_npz"]
