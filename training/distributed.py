"""Distributed training utilities for multi-GPU support.

Provides helpers to parse the ``training.runtime.distributed`` configuration,
create a ``tf.distribute.MirroredStrategy``, convert Python generators into
``tf.data.Dataset`` objects compatible with the strategy, and manage the
strategy scope for model building.

The distributed path is fully opt-in: when ``distributed.enabled`` is
``false`` (the default), the existing single-GPU generator path is used
unchanged.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from utils.config_loader import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedContext",
    "parse_distributed_config",
    "build_distributed_context",
    "wrap_generator_as_dataset",
]


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"mirrored"}

_RESOURCE_RE = re.compile(r"^(gpu|cpu):(\d+)$", re.IGNORECASE)


def _parse_resources(resources_raw: List[Any]) -> List[str]:
    """Validate and normalize the resource list.

    Returns TensorFlow-style device strings like ``/gpu:0``, ``/cpu:0``.
    """
    if not resources_raw:
        return []

    devices: List[str] = []
    for entry in resources_raw:
        text = str(entry).strip().lower()
        match = _RESOURCE_RE.match(text)
        if not match:
            raise ConfigError(
                f"training.runtime.distributed.resources entry {entry!r} "
                "is invalid; expected format like 'gpu:0' or 'cpu:0'"
            )
        dev_type = match.group(1).upper()
        dev_id = match.group(2)
        devices.append(f"/{dev_type}:{dev_id}")
    return devices


def parse_distributed_config(
    runtime_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Parse and validate ``training.runtime.distributed``.

    Returns ``None`` when distributed training is disabled, otherwise a dict
    with keys ``strategy``, ``devices``, ``num_replicas``.
    """
    dist_cfg = runtime_cfg.get("distributed")
    if not isinstance(dist_cfg, dict):
        return None

    enabled = dist_cfg.get("enabled", False)
    if not enabled:
        return None

    strategy_name = str(dist_cfg.get("strategy", "mirrored")).strip().lower()
    if strategy_name not in _VALID_STRATEGIES:
        raise ConfigError(
            f"training.runtime.distributed.strategy must be one of "
            f"{sorted(_VALID_STRATEGIES)}; got {strategy_name!r}"
        )

    resources_raw = dist_cfg.get("resources") or []
    if not isinstance(resources_raw, list):
        raise ConfigError(
            "training.runtime.distributed.resources must be a list"
        )

    devices = _parse_resources(resources_raw)

    return {
        "strategy": strategy_name,
        "devices": devices,
    }


# ---------------------------------------------------------------------------
# Strategy + context
# ---------------------------------------------------------------------------


class DistributedContext:
    """Holds the active distribution strategy and associated metadata.

    Attributes
    ----------
    strategy : tf.distribute.Strategy
        The TensorFlow distribution strategy instance.
    num_replicas : int
        Number of replicas (GPUs) managed by the strategy.
    devices : list[str]
        TF device strings used by the strategy.
    """

    def __init__(
        self,
        strategy: Any,
        num_replicas: int,
        devices: List[str],
    ) -> None:
        self.strategy = strategy
        self.num_replicas = num_replicas
        self.devices = devices

    def scope(self) -> contextlib.AbstractContextManager:
        """Return the strategy scope context manager."""
        return self.strategy.scope()

    def global_batch_size(self, per_replica_batch_size: int) -> int:
        """Compute global batch size from per-replica batch size."""
        return per_replica_batch_size * self.num_replicas


def build_distributed_context(
    runtime_cfg: Dict[str, Any],
) -> Optional[DistributedContext]:
    """Create a ``DistributedContext`` if distributed training is enabled.

    Returns ``None`` when distributed training is disabled.
    """
    parsed = parse_distributed_config(runtime_cfg)
    if parsed is None:
        return None

    import tensorflow as tf  # type: ignore[import]

    strategy_name = parsed["strategy"]
    devices = parsed["devices"]

    if strategy_name == "mirrored":
        if devices:
            strategy = tf.distribute.MirroredStrategy(devices=devices)
        else:
            strategy = tf.distribute.MirroredStrategy()
    else:
        raise ConfigError(
            f"Unsupported distribution strategy: {strategy_name!r}"
        )

    num_replicas = strategy.num_replicas_in_sync
    actual_devices = devices if devices else [
        f"/GPU:{i}" for i in range(num_replicas)
    ]

    logger.info(
        "Distributed training enabled: strategy=%s, num_replicas=%d, devices=%s",
        strategy_name,
        num_replicas,
        actual_devices,
    )

    return DistributedContext(
        strategy=strategy,
        num_replicas=num_replicas,
        devices=actual_devices,
    )


# ---------------------------------------------------------------------------
# Generator → tf.data.Dataset conversion
# ---------------------------------------------------------------------------


def _build_output_signature_no_lt(
    input_shape: Tuple[int, ...],
    num_classes: int,
) -> Any:
    """Build ``tf.data.Dataset`` output signature without long-term features.

    Generator yields: ``(x, (y_up, y_down), (sw_up, sw_down))``
    """
    import tensorflow as tf  # type: ignore[import]

    return (
        tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )


def _build_output_signature_with_lt(
    input_shape: Tuple[int, ...],
    num_classes: int,
    long_term_dim: int,
) -> Any:
    """Build ``tf.data.Dataset`` output signature with long-term features.

    Generator yields: ``((x, lt), (y_up, y_down), (sw_up, sw_down))``
    """
    import tensorflow as tf  # type: ignore[import]

    return (
        (
            tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None, long_term_dim), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        ),
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )


def wrap_generator_as_dataset(
    generator_factory: Callable[[], Iterator[Tuple[Any, ...]]],
    input_shape: Tuple[int, ...],
    num_classes: int,
    long_term_dim: Optional[int],
    global_batch_size: int,
    steps_per_epoch: int,
    distributed_ctx: Optional[DistributedContext] = None,
) -> Any:
    """Convert a generator factory into a ``tf.data.Dataset``.

    Parameters
    ----------
    generator_factory:
        A zero-argument callable that returns a fresh generator each time.
        The generator should yield **unbatched-style** tuples (the existing
        generators already yield fixed-size batches, so we treat each yield
        as one element).
    input_shape:
        Model input shape ``(T, H, W, C)`` (without batch dim).
    num_classes:
        Number of output classes per head.
    long_term_dim:
        Dimension of long-term features, or ``None`` if not enabled.
    global_batch_size:
        The global batch size (used only for prefetch sizing; the generator
        already yields batches of the correct size).
    steps_per_epoch:
        Number of steps per epoch (used to bound the dataset via ``.take()``).
    distributed_ctx:
        Optional distributed context. When provided, the dataset is
        distributed across replicas.

    Returns
    -------
    tf.data.Dataset
        A dataset suitable for ``model.fit()``.
    """
    import tensorflow as tf  # type: ignore[import]

    if long_term_dim is not None:
        output_sig = _build_output_signature_with_lt(
            input_shape, num_classes, long_term_dim,
        )
    else:
        output_sig = _build_output_signature_no_lt(input_shape, num_classes)

    dataset = tf.data.Dataset.from_generator(
        generator_factory,
        output_signature=output_sig,
    )

    # The generator already yields batches, so each element = 1 step.
    # Take steps_per_epoch elements per pass, then repeat forever for Keras.
    dataset = dataset.take(steps_per_epoch).repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if distributed_ctx is not None:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )
        dataset = dataset.with_options(options)

    logger.info(
        "Wrapped generator as tf.data.Dataset: steps_per_epoch=%d, "
        "global_batch_size=%d, long_term=%s",
        steps_per_epoch,
        global_batch_size,
        long_term_dim is not None,
    )

    return dataset
