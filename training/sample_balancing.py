"""Sample selection utilities for coarse class balancing.

This module implements a snapshot-native undersampling strategy driven by the
existing label discretisation (two-head intensity labels). It is designed to be
memory-safe on large datasets by streaming label arrays from snapshot chunks
without loading input tensors.

The balancing configuration lives under ``preprocessing.class_balancing`` and
is intentionally independent from training-time loss reweighting (see
``training.class_weights``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import logging
import math
import random

import numpy as np

from utils.config_loader import ConfigError
from training.snapshot_dataset import (
    CHUNK_STORAGE_FRAME_STORE_V1,
    CHUNK_STORAGE_NPY_SHARDS_V1,
    SnapshotChunk,
    SnapshotDataset,
)


logger = logging.getLogger(__name__)


_CRITERIA_MAX_INTENSITY = "max_intensity"
_CRITERIA_UP_INTENSITY = "up_intensity"
_CRITERIA_DOWN_INTENSITY = "down_intensity"

_POLICY_UNIFORM_TIME = "uniform_time"
_POLICY_RANDOM = "random"
_POLICY_KMEANS = "kmeans"


@dataclass(frozen=True)
class UndersamplingConfig:
    """Configuration for coarse undersampling."""

    labeling_criteria: str
    target_distribution: Tuple[float, ...]
    selection_policy: str
    min_samples_after_balance: int
    min_fraction_after_balance: float
    random_seed: int


def resolve_undersampling_config(config: Dict[str, Any]) -> Optional[UndersamplingConfig]:
    """Parse undersampling config, returning None when disabled."""

    preprocessing_cfg = config.get("preprocessing")
    if not isinstance(preprocessing_cfg, dict):
        raise ConfigError("preprocessing must be a dict")

    cb_cfg = preprocessing_cfg.get("class_balancing")
    if not isinstance(cb_cfg, dict):
        raise ConfigError("preprocessing.class_balancing must be a dict")

    enabled = bool(cb_cfg.get("enabled", False))
    if not enabled:
        return None

    method = str(cb_cfg.get("method") or "")
    if method != "undersampling":
        raise ConfigError("preprocessing.class_balancing.method must be 'undersampling'")

    undersampling_cfg = cb_cfg.get("undersampling")
    if not isinstance(undersampling_cfg, dict):
        raise ConfigError(
            "preprocessing.class_balancing.undersampling must be provided when enabled"
        )

    labeling_criteria = str(undersampling_cfg.get("labeling_criteria") or "")
    selection_policy = str(undersampling_cfg.get("selection_policy") or "")

    target_raw = undersampling_cfg.get("target_distribution")
    if isinstance(target_raw, str):
        if target_raw.strip().lower() != "auto":
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.target_distribution must be a list or 'auto'"
            )
        target_distribution = (0.0,)
    else:
        if not isinstance(target_raw, list) or not target_raw:
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.target_distribution must be a non-empty list"
            )
        try:
            target_distribution = tuple(float(v) for v in target_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.target_distribution must contain only numbers"
            ) from exc

    min_samples_raw = undersampling_cfg.get("min_samples_after_balance")
    if min_samples_raw is None:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_samples_after_balance is required when enabled"
        )
    min_samples = int(min_samples_raw)
    min_fraction_raw = undersampling_cfg.get("min_fraction_after_balance")
    if min_fraction_raw is None:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_fraction_after_balance is required when enabled"
        )
    min_fraction = float(min_fraction_raw)
    random_seed_raw = undersampling_cfg.get("random_seed", 0)
    random_seed = int(random_seed_raw)

    return UndersamplingConfig(
        labeling_criteria=labeling_criteria,
        target_distribution=target_distribution,
        selection_policy=selection_policy,
        min_samples_after_balance=min_samples,
        min_fraction_after_balance=min_fraction,
        random_seed=random_seed,
    )


def compute_undersample_counts(
    *,
    available_counts: Sequence[int],
    target_distribution: Sequence[float],
) -> List[int]:
    """Compute per-class sample counts to keep under an undersampling target.

    This maximizes total retained samples while keeping ratios as close as
    possible to ``target_distribution`` under the constraint ``keep[c] <=
    available_counts[c]``.

    Fail-fast policy
    ----------------
    If a target class weight is > 0 and the available count is 0, this raises
    ConfigError.
    """

    counts = [int(v) for v in available_counts]
    target = [float(v) for v in target_distribution]
    if len(target) == 1 and target[0] == 0.0:
        target = _auto_target_distribution(counts)
    if len(counts) != len(target):
        raise ValueError("available_counts and target_distribution must have the same length")
    if any(v < 0 for v in counts):
        raise ValueError("available_counts must be >= 0")
    if any(v < 0.0 for v in target):
        raise ConfigError("target_distribution values must be >= 0")

    total_weight = float(sum(target))
    if total_weight <= 0.0:
        raise ConfigError("target_distribution must have a positive sum")

    proportions = [v / total_weight for v in target]

    for idx, (p, n) in enumerate(zip(proportions, counts)):
        if p > 0.0 and n <= 0:
            raise ConfigError(
                "Undersampling target requests a non-zero share for a class with zero samples: "
                f"class={idx}, available=0"
            )

    limiting_totals: List[float] = []
    for p, n in zip(proportions, counts):
        if p <= 0.0:
            continue
        limiting_totals.append(float(n) / float(p))
    if not limiting_totals:
        return [0 for _ in counts]

    total_keep = int(math.floor(min(limiting_totals)))
    if total_keep <= 0:
        return [0 for _ in counts]

    desired = [p * float(total_keep) for p in proportions]
    keep = [int(math.floor(v)) for v in desired]

    # Ensure we never exceed availability due to floating point drift.
    keep = [min(k, n) for k, n in zip(keep, counts)]

    allocated = int(sum(keep))
    remaining = int(total_keep - allocated)
    if remaining <= 0:
        return keep

    remainders = [v - float(k) for v, k in zip(desired, keep)]
    order = sorted(range(len(keep)), key=lambda i: remainders[i], reverse=True)
    for idx in order:
        if remaining <= 0:
            break
        if proportions[idx] <= 0.0:
            continue
        if keep[idx] >= counts[idx]:
            continue
        keep[idx] += 1
        remaining -= 1

    return keep


def _auto_target_distribution(available_counts: Sequence[int]) -> List[float]:
    present = [int(v) > 0 for v in available_counts]
    present_count = int(sum(1 for v in present if v))
    if present_count <= 0:
        raise ConfigError("Undersampling auto target_distribution requires at least one non-empty class")
    weight = 100.0 / float(present_count)
    return [weight if is_present else 0.0 for is_present in present]


def _iter_chunk_ranges(
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
) -> Iterator[Tuple[SnapshotChunk, int, int]]:
    if start_index < 0 or end_index < 0 or start_index > end_index:
        raise ValueError("Invalid start_index/end_index")

    for chunk in dataset.chunks:
        chunk_start = int(chunk.start_index)
        chunk_end = int(chunk.start_index + chunk.num_samples)
        if end_index <= chunk_start:
            break
        if start_index >= chunk_end:
            continue

        local_start = max(0, int(start_index - chunk_start))
        local_end = min(int(chunk.num_samples), int(end_index - chunk_start))
        if local_end <= local_start:
            continue
        yield chunk, local_start, local_end


def _slice_chunk_labels(
    chunk: SnapshotChunk,
    local_start: int,
    local_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if chunk.storage_format == CHUNK_STORAGE_NPY_SHARDS_V1 or chunk.storage_format == CHUNK_STORAGE_FRAME_STORE_V1:
        if chunk.array_paths is None:
            raise ConfigError("Snapshot chunk array_paths are required for label slicing")
        y_up = np.load(chunk.array_paths["y_up"], mmap_mode="r")[local_start:local_end]
        y_down = np.load(chunk.array_paths["y_down"], mmap_mode="r")[local_start:local_end]
        return np.asarray(y_up, dtype="int64"), np.asarray(y_down, dtype="int64")

    with np.load(chunk.file_path) as npz:
        if "y_up" not in npz or "y_down" not in npz:
            raise ConfigError("Snapshot chunk missing y_up/y_down arrays")
        y_up = npz["y_up"][local_start:local_end]
        y_down = npz["y_down"][local_start:local_end]
        return np.asarray(y_up, dtype="int64"), np.asarray(y_down, dtype="int64")


def _slice_chunk_labels_for_indices(
    chunk: SnapshotChunk,
    local_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    local_indices = np.asarray(local_indices, dtype="int64")
    if local_indices.ndim != 1:
        raise ValueError("local_indices must be rank 1")
    if local_indices.size == 0:
        return np.zeros((0,), dtype="int64"), np.zeros((0,), dtype="int64")

    if chunk.storage_format == CHUNK_STORAGE_NPY_SHARDS_V1 or chunk.storage_format == CHUNK_STORAGE_FRAME_STORE_V1:
        if chunk.array_paths is None:
            raise ConfigError("Snapshot chunk array_paths are required for label slicing")
        y_up_all = np.load(chunk.array_paths["y_up"], mmap_mode="r")
        y_down_all = np.load(chunk.array_paths["y_down"], mmap_mode="r")
        y_up = np.asarray(y_up_all[local_indices], dtype="int64")
        y_down = np.asarray(y_down_all[local_indices], dtype="int64")
        return y_up, y_down

    with np.load(chunk.file_path) as npz:
        if "y_up" not in npz or "y_down" not in npz:
            raise ConfigError("Snapshot chunk missing y_up/y_down arrays")
        y_up = np.asarray(npz["y_up"][local_indices], dtype="int64")
        y_down = np.asarray(npz["y_down"][local_indices], dtype="int64")
        return y_up, y_down


def _balance_labels(y_up: np.ndarray, y_down: np.ndarray, labeling_criteria: str) -> np.ndarray:
    if labeling_criteria == _CRITERIA_MAX_INTENSITY:
        return np.maximum(y_up, y_down)
    if labeling_criteria == _CRITERIA_UP_INTENSITY:
        return y_up
    if labeling_criteria == _CRITERIA_DOWN_INTENSITY:
        return y_down
    raise ConfigError(
        "Unsupported undersampling labeling_criteria. Expected one of: "
        f"{_CRITERIA_MAX_INTENSITY!r}, {_CRITERIA_UP_INTENSITY!r}, {_CRITERIA_DOWN_INTENSITY!r}; "
        f"got {labeling_criteria!r}"
    )


def compute_available_class_counts(
    *,
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    num_classes: int,
    labeling_criteria: str,
) -> List[int]:
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    counts = np.zeros((num_classes,), dtype="int64")

    for chunk, local_start, local_end in _iter_chunk_ranges(dataset, start_index, end_index):
        y_up, y_down = _slice_chunk_labels(chunk, local_start, local_end)
        y_bal = _balance_labels(y_up, y_down, labeling_criteria)
        y_bal = np.asarray(y_bal, dtype="int64")

        # Ignore out-of-range labels instead of crashing.
        valid = (y_bal >= 0) & (y_bal < num_classes)
        if not bool(valid.any()):
            continue

        binc = np.bincount(y_bal[valid], minlength=num_classes)
        counts += binc.astype("int64")

    return [int(v) for v in counts.tolist()]


def _iter_indices_by_chunk(
    dataset: SnapshotDataset,
    indices: np.ndarray,
) -> Iterator[Tuple[SnapshotChunk, np.ndarray]]:
    indices = np.asarray(indices, dtype="int64")
    if indices.ndim != 1:
        raise ValueError("indices must be rank 1")

    if indices.size == 0:
        return

    pos = 0
    total = int(indices.shape[0])
    for chunk in dataset.chunks:
        if pos >= total:
            break
        chunk_start = int(chunk.start_index)
        chunk_end = int(chunk.start_index + chunk.num_samples)

        # Indices are expected to be sorted and within [0, dataset.total_samples).
        if int(indices[pos]) < chunk_start:
            raise ConfigError("Undersampling indices are not aligned with chunk boundaries")

        end = pos
        while end < total and int(indices[end]) < chunk_end:
            end += 1
        if end > pos:
            local = np.asarray(indices[pos:end] - chunk_start, dtype="int64")
            yield chunk, local
        pos = end

    if pos < total:
        raise ConfigError("Undersampling indices exceed snapshot dataset bounds")


def compute_class_counts_for_indices(
    *,
    dataset: SnapshotDataset,
    indices: np.ndarray,
    num_classes: int,
    labeling_criteria: str,
) -> List[int]:
    """Compute per-class counts for an explicit index list under a labeling criteria.

    This mirrors compute_available_class_counts but only for the given indices.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")

    indices = np.asarray(indices, dtype="int64")
    if indices.ndim != 1:
        raise ValueError("indices must be rank 1")
    if indices.size == 0:
        return [0 for _ in range(num_classes)]

    counts = np.zeros((num_classes,), dtype="int64")
    for chunk, local_indices in _iter_indices_by_chunk(dataset, indices):
        y_up, y_down = _slice_chunk_labels_for_indices(chunk, local_indices)
        y_bal = _balance_labels(y_up, y_down, labeling_criteria)
        y_bal = np.asarray(y_bal, dtype="int64")

        valid = (y_bal >= 0) & (y_bal < num_classes)
        if not bool(valid.any()):
            continue

        binc = np.bincount(y_bal[valid], minlength=num_classes)
        counts += binc.astype("int64")

    return [int(v) for v in counts.tolist()]


def select_undersampled_indices(
    *,
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    num_classes: int,
    labeling_criteria: str,
    keep_counts: Sequence[int],
    selection_policy: str,
    random_seed: int,
) -> np.ndarray:
    """Return sorted global indices to keep under the undersampling plan."""
    if len(keep_counts) != num_classes:
        raise ValueError("keep_counts length must equal num_classes")
    keep_counts_int = [int(v) for v in keep_counts]
    if any(v < 0 for v in keep_counts_int):
        raise ValueError("keep_counts must be >= 0")

    total_keep = int(sum(keep_counts_int))
    if total_keep <= 0:
        return np.zeros((0,), dtype="int64")

    if selection_policy == _POLICY_UNIFORM_TIME:
        return _select_uniform_time(
            dataset=dataset,
            start_index=start_index,
            end_index=end_index,
            num_classes=num_classes,
            labeling_criteria=labeling_criteria,
            keep_counts=keep_counts_int,
        )

    if selection_policy == _POLICY_RANDOM:
        return _select_random(
            dataset=dataset,
            start_index=start_index,
            end_index=end_index,
            num_classes=num_classes,
            labeling_criteria=labeling_criteria,
            keep_counts=keep_counts_int,
            random_seed=int(random_seed),
        )

    if selection_policy == _POLICY_KMEANS:
        raise ConfigError("preprocessing.class_balancing undersampling selection_policy='kmeans' is not implemented yet")

    raise ConfigError(
        "Unsupported undersampling selection_policy. Expected one of: "
        f"{_POLICY_UNIFORM_TIME!r}, {_POLICY_RANDOM!r}, {_POLICY_KMEANS!r}; got {selection_policy!r}"
    )


def _should_select_uniform(ordinal: int, total: int, keep: int) -> bool:
    """Evenly select exactly ``keep`` ordinals out of ``total``.

    Uses a Bresenham-style integer rule:
    select when floor((i+1)*keep/total) > floor(i*keep/total).
    """
    if keep <= 0 or total <= 0:
        return False
    if keep >= total:
        return True
    left = int(((ordinal + 1) * keep) // total)
    right = int((ordinal * keep) // total)
    return left > right


def _select_uniform_time(
    *,
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    num_classes: int,
    labeling_criteria: str,
    keep_counts: Sequence[int],
) -> np.ndarray:
    available = compute_available_class_counts(
        dataset=dataset,
        start_index=start_index,
        end_index=end_index,
        num_classes=num_classes,
        labeling_criteria=labeling_criteria,
    )

    target = [int(v) for v in keep_counts]
    for c in range(num_classes):
        if target[c] > available[c]:
            raise ConfigError(
                "Undersampling plan requests more samples than available: "
                f"class={c}, keep={target[c]}, available={available[c]}"
            )

    selected = np.empty((int(sum(target)),), dtype="int64")
    out_pos = 0
    seen_by_class = [0 for _ in range(num_classes)]
    kept_by_class = [0 for _ in range(num_classes)]

    for chunk, local_start, local_end in _iter_chunk_ranges(dataset, start_index, end_index):
        y_up, y_down = _slice_chunk_labels(chunk, local_start, local_end)
        y_bal = _balance_labels(y_up, y_down, labeling_criteria)
        y_bal = np.asarray(y_bal, dtype="int64")

        base_global = int(chunk.start_index + local_start)
        for offset, cls in enumerate(y_bal.tolist()):
            class_idx = int(cls)
            if class_idx < 0 or class_idx >= num_classes:
                continue

            ordinal = int(seen_by_class[class_idx])
            seen_by_class[class_idx] += 1

            if kept_by_class[class_idx] >= target[class_idx]:
                continue

            if not _should_select_uniform(ordinal, available[class_idx], target[class_idx]):
                continue

            selected[out_pos] = int(base_global + offset)
            out_pos += 1
            kept_by_class[class_idx] += 1

    if out_pos != selected.shape[0]:
        raise ConfigError(
            "Undersampling uniform_time selection did not produce expected number of samples: "
            f"expected={selected.shape[0]}, got={out_pos}"
        )
    if kept_by_class != target:
        raise ConfigError(
            "Undersampling uniform_time selection did not match per-class targets: "
            f"expected={target}, got={kept_by_class}"
        )
    return selected


def _select_random(
    *,
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    num_classes: int,
    labeling_criteria: str,
    keep_counts: Sequence[int],
    random_seed: int,
) -> np.ndarray:
    # Two-pass: first count, then reservoir-sample per class.
    available = compute_available_class_counts(
        dataset=dataset,
        start_index=start_index,
        end_index=end_index,
        num_classes=num_classes,
        labeling_criteria=labeling_criteria,
    )

    keep = [int(v) for v in keep_counts]
    for c in range(num_classes):
        if keep[c] > available[c]:
            raise ConfigError(
                "Undersampling plan requests more samples than available: "
                f"class={c}, keep={keep[c]}, available={available[c]}"
            )

    rng = random.Random(int(random_seed))
    reservoirs: List[List[int]] = [[] for _ in range(num_classes)]
    seen_by_class = [0 for _ in range(num_classes)]

    for chunk, local_start, local_end in _iter_chunk_ranges(dataset, start_index, end_index):
        y_up, y_down = _slice_chunk_labels(chunk, local_start, local_end)
        y_bal = _balance_labels(y_up, y_down, labeling_criteria)
        y_bal = np.asarray(y_bal, dtype="int64")

        base_global = int(chunk.start_index + local_start)
        for offset, cls in enumerate(y_bal.tolist()):
            class_idx = int(cls)
            if class_idx < 0 or class_idx >= num_classes:
                continue
            k = int(keep[class_idx])
            if k <= 0:
                seen_by_class[class_idx] += 1
                continue

            j = int(seen_by_class[class_idx])
            seen_by_class[class_idx] += 1
            global_idx = int(base_global + offset)

            res = reservoirs[class_idx]
            if len(res) < k:
                res.append(global_idx)
                continue

            # Replace an existing element with decreasing probability.
            r = rng.randrange(0, j + 1)
            if r < k:
                res[r] = global_idx

    flat: List[int] = []
    for c, res in enumerate(reservoirs):
        if len(res) != int(keep[c]):
            raise ConfigError(
                "Random undersampling did not fill reservoir for class: "
                f"class={c}, expected={keep[c]}, got={len(res)}"
            )
        flat.extend(res)

    out = np.asarray(sorted(flat), dtype="int64")
    if out.shape[0] != int(sum(keep)):
        raise ConfigError(
            "Random undersampling returned unexpected total size: "
            f"expected={sum(keep)}, got={out.shape[0]}"
        )
    return out


__all__ = [
    "UndersamplingConfig",
    "compute_available_class_counts",
    "compute_class_counts_for_indices",
    "compute_undersample_counts",
    "resolve_undersampling_config",
    "select_undersampled_indices",
]
