"""Streaming snapshot dataset builder and batch generators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple
import logging
import math
import os

import numpy as np

from data.greptime_client import (
    OrderBookChunk,
    _generate_time_chunks,
    stream_order_book_chunks_by_time,
)
from preprocessing.depth_aggregator import aggregate_snapshot_to_hybrid, get_hybrid_output_shape
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.time_utils import normalize_timestamp_array
from utils.config_loader import ConfigError
from .snapshot_store import (
    SnapshotContext,
    load_or_create_manifest,
    maybe_evict_snapshots,
    resolve_snapshot_context,
    save_manifest,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotChunk:
    """Chunk metadata for a snapshot dataset."""

    start: str
    end: str
    file_path: str
    num_samples: int
    start_index: int


@dataclass(frozen=True)
class SnapshotDataset:
    """Resolved snapshot dataset metadata."""

    snapshot_dir: str
    manifest: Dict[str, Any]
    chunks: List[SnapshotChunk]
    total_samples: int
    config_hash: str


@dataclass
class SnapshotRecord:
    """Snapshot-level data used for streaming sample construction."""

    timestamp: np.datetime64
    snapshot_features: List[float]
    depth: Optional[Dict[str, np.ndarray]]
    mid_price: float
    hybrid_snapshot: Optional[np.ndarray]
    volume_proxy: float
    confidence: float
    gap_reset: bool
    observed: bool


@dataclass
class MultiAssetSnapshotRecord:
    """Snapshot-level data for multiple assets at a given timestamp."""

    timestamp: np.datetime64
    asset_snapshots: Dict[str, SnapshotRecord]


@dataclass(frozen=True)
class SampleRecord:
    """Derived sample record for model training."""

    x: np.ndarray
    y_up: int
    y_down: int
    anchor_ts_seconds: int
    duty_cycle: float


@dataclass(frozen=True)
class NormalizationStats:
    """Normalization statistics for snapshot datasets."""

    method: str
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None


class GapHandler:
    """Handle snapshot gaps based on configuration."""

    def __init__(
        self,
        cadence_seconds: int,
        validation_max_gap_seconds: int,
        alignment_max_gap_seconds: int,
        handle_gaps: str,
        check_missing_data: bool,
        fail_on_invalid: bool,
    ) -> None:
        self._cadence_seconds = cadence_seconds
        self._validation_max_gap_seconds = validation_max_gap_seconds
        self._alignment_max_gap_seconds = alignment_max_gap_seconds
        self._handle_gaps = handle_gaps
        self._check_missing_data = check_missing_data
        self._fail_on_invalid = fail_on_invalid
        self._prev_snapshot: Optional[SnapshotRecord] = None

    def iter_gap_handled(self, snapshots: Iterable[SnapshotRecord]) -> Iterator[SnapshotRecord]:
        for snapshot in snapshots:
            if self._prev_snapshot is None:
                self._prev_snapshot = snapshot
                yield snapshot
                continue

            prev = self._prev_snapshot
            gap_secs = int((snapshot.timestamp - prev.timestamp).astype("timedelta64[s]").astype("int64"))

            if gap_secs < 0:
                message = "Non-monotonic snapshot timestamps detected during streaming"
                if self._fail_on_invalid:
                    raise ValueError(message)
                logger.warning(message)
                self._prev_snapshot = snapshot
                yield snapshot
                continue

            if self._check_missing_data and gap_secs > self._validation_max_gap_seconds:
                message = (
                    "Snapshot timestamp gaps exceed data.validation.max_gap_seconds; "
                    f"max_gap_seconds={self._validation_max_gap_seconds}, observed_gap_seconds={gap_secs}"
                )
                if self._fail_on_invalid:
                    raise ValueError(message)
                logger.warning(message)

            gap_reset = gap_secs > self._alignment_max_gap_seconds
            if self._alignment_max_gap_seconds <= 0:
                raise ValueError("alignment.max_gap_seconds must be positive")
            base_confidence = max(0.0, 1.0 - (gap_secs / float(self._alignment_max_gap_seconds)))

            if (
                gap_secs > self._cadence_seconds
                and gap_secs <= self._alignment_max_gap_seconds
                and self._handle_gaps in {"forward_fill", "interpolate"}
            ):
                missing_steps = gap_secs // self._cadence_seconds - 1
                for step in range(1, missing_steps + 1):
                    ts_new = prev.timestamp + np.timedelta64(step * self._cadence_seconds, "s")
                    if self._handle_gaps == "forward_fill":
                        distance_fraction = step / float(missing_steps + 1)
                        confidence = max(0.0, base_confidence * (1.0 - distance_fraction))
                        yield SnapshotRecord(
                            timestamp=ts_new,
                            snapshot_features=list(prev.snapshot_features),
                            depth=_copy_depth(prev.depth),
                            mid_price=prev.mid_price,
                            hybrid_snapshot=None if prev.hybrid_snapshot is None else prev.hybrid_snapshot.copy(),
                            volume_proxy=prev.volume_proxy,
                            confidence=confidence,
                            gap_reset=False,
                            observed=False,
                        )
                    else:
                        alpha = step / float(missing_steps + 1)
                        distance_fraction = min(alpha, 1.0 - alpha)
                        confidence = max(0.0, base_confidence * (1.0 - distance_fraction))
                        yield SnapshotRecord(
                            timestamp=ts_new,
                            snapshot_features=_interpolate_features(prev.snapshot_features, snapshot.snapshot_features, alpha),
                            depth=_interpolate_depth(prev.depth, snapshot.depth, alpha),
                            mid_price=(1.0 - alpha) * prev.mid_price + alpha * snapshot.mid_price,
                            hybrid_snapshot=None,
                            volume_proxy=(1.0 - alpha) * prev.volume_proxy + alpha * snapshot.volume_proxy,
                            confidence=confidence,
                            gap_reset=False,
                            observed=False,
                        )

            self._prev_snapshot = snapshot
            if gap_reset:
                yield SnapshotRecord(
                    timestamp=snapshot.timestamp,
                    snapshot_features=list(snapshot.snapshot_features),
                    depth=_copy_depth(snapshot.depth),
                    mid_price=snapshot.mid_price,
                    hybrid_snapshot=None if snapshot.hybrid_snapshot is None else snapshot.hybrid_snapshot.copy(),
                    volume_proxy=snapshot.volume_proxy,
                    confidence=snapshot.confidence,
                    gap_reset=True,
                    observed=snapshot.observed,
                )
            else:
                yield snapshot

    @staticmethod
    def align_multi_asset(
        asset_records: Dict[str, List[SnapshotRecord]],
        assets: List[str],
        target_asset: str,
        alignment_cfg: Dict[str, Any],
        representation: str,
        cadence_seconds: int,
        hybrid_levels: Optional[int],
        fail_on_invalid: bool,
    ) -> List[MultiAssetSnapshotRecord]:
        return _align_multi_asset_records(
            asset_records,
            assets,
            target_asset,
            alignment_cfg,
            representation,
            cadence_seconds,
            hybrid_levels,
            fail_on_invalid,
        )


class StreamingSampleBuilder:
    """Streaming builder for training samples from snapshot records."""

    def __init__(
        self,
        config: Dict[str, Any],
        representation: str,
        height: int,
        width: int,
        assets: List[str],
        target_asset: str,
    ) -> None:
        self._config = config
        self._representation = representation
        self._height = height
        self._width = width
        self._assets = [str(a) for a in assets]
        if not self._assets:
            raise ValueError("At least one asset must be provided for snapshot streaming")
        if target_asset not in self._assets:
            raise ValueError("target_asset must be included in assets for snapshot streaming")
        self._target_asset = str(target_asset)
        self._asset_indices = {asset: idx for idx, asset in enumerate(self._assets)}
        alignment_cfg = config["data"]["asset_pairs"]["alignment"]
        self._include_mask_channel = bool(alignment_cfg["include_mask_channel"])

        data_cfg = config["data"]
        time_range_cfg = data_cfg["time_range"]
        cadence_seconds = int(time_range_cfg["cadence_seconds"])
        if cadence_seconds <= 0:
            raise ValueError("data.time_range.cadence_seconds must be positive")

        targets_cfg = config["targets"]
        visible_window_seconds = int(targets_cfg["visible_window_seconds"])
        prediction_horizon_seconds = int(targets_cfg["prediction_horizon_seconds"])

        if visible_window_seconds <= 0:
            raise ValueError("targets.visible_window_seconds must be positive")
        if prediction_horizon_seconds <= 0:
            raise ValueError("targets.prediction_horizon_seconds must be positive")
        if visible_window_seconds % cadence_seconds != 0:
            raise ValueError(
                "targets.visible_window_seconds must be an integer multiple of data.time_range.cadence_seconds",
            )

        self._window_steps = visible_window_seconds // cadence_seconds
        if self._window_steps <= 0:
            raise ValueError("Derived visible window steps must be positive")

        self._horizon_steps = prediction_horizon_seconds // cadence_seconds
        if self._horizon_steps <= 0:
            raise ValueError(
                "targets.prediction_horizon_seconds must be at least data.time_range.cadence_seconds",
            )

        price_classes_cfg = targets_cfg["price_classes"]
        boundaries = price_classes_cfg["boundaries"]
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("targets.price_classes.boundaries must be a non-empty list")
        self._boundaries = [float(b) for b in boundaries]

        model_cfg = config["model"]
        output_cfg = model_cfg["output"]
        output_type = str(output_cfg["type"])
        if output_type != "two_head_intensity":
            raise ValueError("Only model.output.type='two_head_intensity' is supported in snapshot mode")
        self._num_classes = int(output_cfg["num_classes"])
        expected_classes = len(self._boundaries) + 1
        if self._num_classes != expected_classes:
            raise ValueError(
                "model.output.num_classes must equal len(targets.price_classes.boundaries) + 1",
            )

        fe_cfg = config["preprocessing"]["feature_engineering"]
        self._feature_engineer = FeatureEngineer(config) if bool(fe_cfg["enabled"]) else None

        ir_cfg = model_cfg["input_representation"]
        tf_cfg = ir_cfg["temporal_features"]
        self._temporal_mode = str(tf_cfg["integration_mode"])
        self._use_local = bool(tf_cfg["use_local_features"])
        self._use_global = bool(tf_cfg["use_global_features"])
        self._local_features = data_cfg["temporal_features"]["local"] or []
        self._global_features = data_cfg["temporal_features"]["global"] or []
        self._market_session_cfg = data_cfg["temporal_features"]["market_session"]

        if self._temporal_mode not in {"none", "concat_channels"}:
            raise ValueError(
                "model.input_representation.temporal_features.integration_mode must be 'none' or 'concat_channels'",
            )

        self._buffer: deque[MultiAssetSnapshotRecord] = deque()
        self._buffer_start_idx = 0
        self._next_anchor_idx = self._window_steps - 1
        self._latest_idx = -1
        self._dataset_start_day: Optional[int] = None

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def add_snapshot(self, snapshot: MultiAssetSnapshotRecord) -> List[SampleRecord]:
        samples: List[SampleRecord] = []
        self._buffer.append(snapshot)
        self._latest_idx += 1

        target_record = snapshot.asset_snapshots.get(self._target_asset)
        if target_record is not None and target_record.gap_reset:
            self._buffer = deque([snapshot])
            self._buffer_start_idx = self._latest_idx
            self._next_anchor_idx = self._latest_idx + (self._window_steps - 1)
            return samples

        while self._latest_idx >= self._next_anchor_idx + self._horizon_steps:
            sample = self._build_sample(self._next_anchor_idx)
            if sample is not None:
                samples.append(sample)

            self._next_anchor_idx += 1
            earliest_needed = self._next_anchor_idx - (self._window_steps - 1)
            while self._buffer and self._buffer_start_idx < earliest_needed:
                self._buffer.popleft()
                self._buffer_start_idx += 1

        return samples

    def _build_sample(self, anchor_idx: int) -> Optional[SampleRecord]:
        anchor_pos = anchor_idx - self._buffer_start_idx
        if anchor_pos < 0 or anchor_pos >= len(self._buffer):
            return None

        window_start = anchor_pos - (self._window_steps - 1)
        if window_start < 0:
            return None

        window_records = list(self._buffer)[window_start : anchor_pos + 1]
        if len(window_records) != self._window_steps:
            return None

        mid_prices = np.array(
            [rec.asset_snapshots[self._target_asset].mid_price for rec in self._buffer],
            dtype="float64",
        )
        volumes = np.array(
            [rec.asset_snapshots[self._target_asset].volume_proxy for rec in self._buffer],
            dtype="float64",
        )

        future_start = anchor_pos + 1
        future_end = anchor_pos + self._horizon_steps + 1
        future_window = mid_prices[future_start:future_end]
        p0 = mid_prices[anchor_pos]

        if future_window.size == 0 or p0 <= 0:
            max_up = 0.0
            max_down = 0.0
        else:
            rel_moves = (future_window - p0) / p0 * 100.0
            max_up = float(np.max(rel_moves))
            max_down = float(np.min(rel_moves))

        y_up, y_down = _compute_intensity_bins(self._boundaries, max_up, max_down)

        x_seq = self._build_input_sequence(window_records)
        if x_seq is None:
            return None

        if self._feature_engineer is not None:
            target_record = window_records[anchor_pos - window_start].asset_snapshots[self._target_asset]
            fe_vector = _compute_feature_vector(
                self._feature_engineer,
                target_record,
                mid_prices,
                volumes,
                anchor_pos,
                self._config,
            )
            if fe_vector is not None and fe_vector.size > 0:
                x_seq = _concat_broadcast_features(x_seq, fe_vector)

        if self._temporal_mode == "concat_channels":
            temporal_vector = _compute_temporal_vector(
                anchor_ts=window_records[anchor_pos - window_start].timestamp,
                local_cfg=self._local_features if self._use_local else [],
                global_cfg=self._global_features if self._use_global else [],
                market_session_cfg=self._market_session_cfg,
                dataset_start_day=self._dataset_start_day,
            )
            if temporal_vector is not None and temporal_vector.size > 0:
                if self._dataset_start_day is None:
                    self._dataset_start_day = int(
                        window_records[anchor_pos - window_start]
                        .timestamp.astype("datetime64[D]")
                        .astype("int64")
                    )
                x_seq = _concat_broadcast_features(x_seq, temporal_vector)

        anchor_ts_seconds = int(
            window_records[anchor_pos - window_start]
            .timestamp.astype("datetime64[s]")
            .astype("int64")
        )

        observed_count = sum(
            1
            for rec in window_records
            if rec.asset_snapshots[self._target_asset].observed
        )
        duty_cycle = float(observed_count) / float(self._window_steps)

        return SampleRecord(
            x=x_seq.astype("float32"),
            y_up=int(y_up),
            y_down=int(y_down),
            anchor_ts_seconds=anchor_ts_seconds,
            duty_cycle=duty_cycle,
        )

    def _build_input_sequence(self, window_records: List[MultiAssetSnapshotRecord]) -> Optional[np.ndarray]:
        num_assets = len(self._assets)
        if self._representation == "hybrid":
            effective_levels = get_hybrid_output_shape(self._config)
            x_seq = np.zeros((self._window_steps, effective_levels, 4, num_assets), dtype="float32")
            for t_idx, rec in enumerate(window_records):
                for asset_idx, asset in enumerate(self._assets):
                    asset_rec = rec.asset_snapshots.get(asset)
                    if asset_rec is None:
                        return None
                    if asset_rec.hybrid_snapshot is None:
                        if asset_rec.depth is None:
                            return None
                        hybrid = aggregate_snapshot_to_hybrid(
                            bid_prices=asset_rec.depth["bid_prices"],
                            bid_quantities=asset_rec.depth["bid_quantities"],
                            ask_prices=asset_rec.depth["ask_prices"],
                            ask_quantities=asset_rec.depth["ask_quantities"],
                            config=self._config,
                        ).astype("float32")
                        x_seq[t_idx, :, :, asset_idx] = hybrid
                    else:
                        x_seq[t_idx, :, :, asset_idx] = asset_rec.hybrid_snapshot
            if self._include_mask_channel:
                mask_seq = np.zeros_like(x_seq)
                for t_idx, rec in enumerate(window_records):
                    for asset_idx, asset in enumerate(self._assets):
                        asset_rec = rec.asset_snapshots.get(asset)
                        if asset_rec is None:
                            return None
                        confidence = float(asset_rec.confidence)
                        mask_seq[t_idx, :, :, asset_idx] = confidence
                x_seq = np.concatenate([x_seq, mask_seq], axis=-1)
            return x_seq

        if self._representation == "top_of_book":
            x_seq = np.zeros((self._window_steps, self._height, self._width, num_assets), dtype="float32")
            for t_idx, rec in enumerate(window_records):
                for asset_idx, asset in enumerate(self._assets):
                    asset_rec = rec.asset_snapshots.get(asset)
                    if asset_rec is None:
                        return None
                    features = asset_rec.snapshot_features
                    if len(features) < 4:
                        continue
                    bid_price, bid_qty, ask_price, ask_qty = features[:4]
                    x_seq[t_idx, 0, 0, asset_idx] = float(bid_price)
                    if self._width > 1:
                        x_seq[t_idx, 0, 1, asset_idx] = float(bid_qty)
                    if self._height > 1:
                        x_seq[t_idx, 1, 0, asset_idx] = float(ask_price)
                    if self._height > 1 and self._width > 1:
                        x_seq[t_idx, 1, 1, asset_idx] = float(ask_qty)
            if self._include_mask_channel:
                mask_seq = np.zeros_like(x_seq)
                for t_idx, rec in enumerate(window_records):
                    for asset_idx, asset in enumerate(self._assets):
                        asset_rec = rec.asset_snapshots.get(asset)
                        if asset_rec is None:
                            return None
                        confidence = float(asset_rec.confidence)
                        mask_seq[t_idx, :, :, asset_idx] = confidence
                x_seq = np.concatenate([x_seq, mask_seq], axis=-1)
            return x_seq

        raise ValueError(f"Unsupported order book representation: {self._representation}")


def prepare_snapshot_dataset(config: Dict[str, Any]) -> SnapshotDataset:
    """Prepare (build or load) a snapshot dataset for streaming training."""

    context = resolve_snapshot_context(config)
    max_snapshots = int(config["snapshot"]["max_snapshots"])
    maybe_evict_snapshots(context, max_snapshots)

    manifest = load_or_create_manifest(context, config)

    complete = bool(manifest.get("complete"))
    chunk_entries = manifest.get("chunks", []) or []
    if complete and not chunk_entries:
        logger.warning(
            "Snapshot manifest marked complete but contains no chunks; rebuilding snapshot.",
        )
        complete = False

    if complete:
        for entry in chunk_entries:
            file_rel = entry.get("file")
            if not file_rel:
                complete = False
                break
            file_path = os.path.join(context.snapshot_dir, file_rel)
            if not os.path.exists(file_path):
                complete = False
                break
    if not complete:
        manifest = _build_snapshot_chunks(config, context, manifest)

    dataset = load_snapshot_dataset(context, config)
    return dataset


def load_snapshot_dataset(context: SnapshotContext, config: Dict[str, Any]) -> SnapshotDataset:
    manifest = load_or_create_manifest(context, config)
    if manifest.get("config_hash") != context.config_hash:
        raise ConfigError("Snapshot manifest config_hash does not match current configuration")

    chunks_meta = []
    for entry in manifest.get("chunks", []) or []:
        file_path = os.path.join(context.snapshot_dir, entry["file"])
        if not os.path.exists(file_path):
            continue
        chunks_meta.append(
            {
                "start": entry["start"],
                "end": entry["end"],
                "file": file_path,
                "num_samples": int(entry["num_samples"]),
            }
        )

    chunks_meta.sort(key=lambda item: item["start"])
    chunks: List[SnapshotChunk] = []
    start_idx = 0
    for entry in chunks_meta:
        chunks.append(
            SnapshotChunk(
                start=entry["start"],
                end=entry["end"],
                file_path=entry["file"],
                num_samples=entry["num_samples"],
                start_index=start_idx,
            )
        )
        start_idx += entry["num_samples"]

    return SnapshotDataset(
        snapshot_dir=context.snapshot_dir,
        manifest=manifest,
        chunks=chunks,
        total_samples=start_idx,
        config_hash=context.config_hash,
    )


def iter_snapshot_batches(
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Iterate slices of snapshot chunks for a global index range.

    Returns x, y_up, y_down, anchor_ts, duty_cycle arrays.
    """

    if start_index < 0 or end_index < 0 or start_index > end_index:
        raise ValueError("Invalid start_index/end_index for snapshot batch iteration")

    for chunk in dataset.chunks:
        chunk_start = chunk.start_index
        chunk_end = chunk.start_index + chunk.num_samples
        if end_index <= chunk_start:
            break
        if start_index >= chunk_end:
            continue

        local_start = max(0, start_index - chunk_start)
        local_end = min(chunk.num_samples, end_index - chunk_start)
        if local_end <= local_start:
            continue

        with np.load(chunk.file_path) as npz:
            x = npz["x"][local_start:local_end]
            y_up = npz["y_up"][local_start:local_end]
            y_down = npz["y_down"][local_start:local_end]
            anchor_ts = npz["anchor_ts"][local_start:local_end]
            if "duty_cycle" not in npz:
                raise ConfigError(
                    "Snapshot chunk missing duty_cycle. Rebuild snapshot dataset to enable duty-cycle weighting."
                )
            duty_cycle = npz["duty_cycle"][local_start:local_end]

        if duty_cycle.shape[0] != x.shape[0]:
            raise ConfigError(
                "Snapshot chunk duty_cycle length does not match x length. Rebuild snapshot dataset."
            )

        yield x, y_up, y_down, anchor_ts, duty_cycle


def get_mask_channel_info(config: Dict[str, Any]) -> Tuple[int, int]:
    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    correlated_assets = asset_pairs_cfg["correlated_assets"] or []
    num_assets = 1 + len(correlated_assets)

    alignment_cfg = asset_pairs_cfg["alignment"]
    include_mask = bool(alignment_cfg["include_mask_channel"])
    if not include_mask:
        return 0, 0
    return num_assets, num_assets


def build_training_generator(
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    batch_size: int,
    num_classes: int,
    normalization: Optional[NormalizationStats],
    sample_weight_cfg: Optional[Dict[str, Any]],
    mask_start: int = 0,
    mask_count: int = 0,
    class_weights_up: Optional[Dict[int, float]] = None,
    class_weights_down: Optional[Dict[int, float]] = None,
) -> Tuple[Iterator[Tuple[Any, ...]], int]:
    """Create a generator for model.fit from snapshot chunks.

    Parameters
    ----------
    dataset:
        Snapshot dataset to iterate.
    start_index:
        Start sample index (inclusive).
    end_index:
        End sample index (exclusive).
    batch_size:
        Number of samples per batch.
    num_classes:
        Number of output classes.
    normalization:
        Optional normalization stats to apply.
    sample_weight_cfg:
        Optional sample weighting config (exponential decay).
    mask_start:
        Start index for mask channels (for normalization).
    mask_count:
        Number of mask channels (for normalization).
    class_weights_up:
        Optional dict mapping class index to weight for up-head.
        When provided, class weights are applied as sample weights.
    class_weights_down:
        Optional dict mapping class index to weight for down-head.
        When provided, class weights are applied as sample weights.

    Returns
    -------
    Tuple[Iterator, int]:
        Generator and number of steps per epoch.

    Notes
    -----
    Duty-cycle weighting is always applied. When both sample_weight_cfg
    (exponential decay) and class weights are provided, the final sample
    weight is the product of duty_cycle * decay * class weight.
    Class weights are converted to per-sample weights by looking up each
    sample's label in the class_weights dict.
    """
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    total_samples = max(0, end_index - start_index)
    steps = int(math.ceil(total_samples / float(batch_size))) if total_samples > 0 else 0

    # Determine if exponential decay weighting is enabled
    current_day = None
    decay_const = None
    use_decay_weights = False
    if sample_weight_cfg and sample_weight_cfg.get("enabled"):
        apply_to = str(sample_weight_cfg.get("apply_to", "loss_function"))
        if apply_to != "loss_function":
            raise ValueError(
                "training.sample_weighting.apply_to must be 'loss_function' when enabled; "
                f"got {apply_to!r}"
            )
        method = str(sample_weight_cfg["method"])
        if method != "exponential_decay":
            raise ValueError("training.sample_weighting.method must be 'exponential_decay'")
        half_life_days = int(sample_weight_cfg["half_life_days"])
        if half_life_days <= 0:
            raise ValueError("training.sample_weighting.half_life_days must be positive")
        decay_const = float(np.log(2.0) / float(half_life_days))
        current_day = _compute_current_day(dataset, start_index, end_index)
        use_decay_weights = True

    # Determine if class weighting is enabled
    use_class_weights = class_weights_up is not None or class_weights_down is not None

    # Duty-cycle weighting is always applied
    use_weights = True

    eye = np.eye(num_classes, dtype="float32")

    # Helper to compute class-based sample weights
    def _compute_class_sample_weights(
        y_up: np.ndarray,
        y_down: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert class weights to per-sample weights."""
        batch_len = y_up.shape[0]

        if class_weights_up is not None:
            weights_up = np.array(
                [class_weights_up.get(int(label), 1.0) for label in y_up],
                dtype="float32",
            )
        else:
            weights_up = np.ones(batch_len, dtype="float32")

        if class_weights_down is not None:
            weights_down = np.array(
                [class_weights_down.get(int(label), 1.0) for label in y_down],
                dtype="float32",
            )
        else:
            weights_down = np.ones(batch_len, dtype="float32")

        return weights_up, weights_down

    def _generator_with_weights() -> Iterator[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        while True:
            for x_chunk, y_up_chunk, y_down_chunk, anchor_ts, duty_cycle_chunk in iter_snapshot_batches(
                dataset, start_index, end_index
            ):
                x_chunk = _apply_normalization(x_chunk, normalization, mask_start, mask_count)

                n_chunk = x_chunk.shape[0]
                for offset in range(0, n_chunk, batch_size):
                    x_batch = x_chunk[offset : offset + batch_size]
                    y_up = y_up_chunk[offset : offset + batch_size]
                    y_down = y_down_chunk[offset : offset + batch_size]
                    duty_cycle = duty_cycle_chunk[offset : offset + batch_size].astype("float32")

                    y_up_oh = eye[y_up]
                    y_down_oh = eye[y_down]
                    y_batch = (y_up_oh, y_down_oh)

                    # Start with duty-cycle weights
                    weights_up = duty_cycle
                    weights_down = duty_cycle.copy()

                    # Apply exponential decay if enabled
                    if use_decay_weights:
                        if current_day is None or decay_const is None:
                            raise ValueError("Sample weighting requires current_day and decay_const")
                        anchor_slice = anchor_ts[offset : offset + batch_size]
                        anchor_days = (anchor_slice // 86400).astype("float64")
                        age_days = float(current_day) - anchor_days
                        decay_weights = np.exp(-age_days * float(decay_const)).astype("float32")
                        weights_up = weights_up * decay_weights
                        weights_down = weights_down * decay_weights

                    # Apply class weights if enabled
                    if use_class_weights:
                        class_w_up, class_w_down = _compute_class_sample_weights(y_up, y_down)
                        weights_up = weights_up * class_w_up
                        weights_down = weights_down * class_w_down

                    sample_weight = (weights_up, weights_down)
                    yield x_batch, y_batch, sample_weight

    return _generator_with_weights(), steps


def compute_normalization_stats(
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    method: str,
    mask_start: int = 0,
    mask_count: int = 0,
) -> NormalizationStats:
    """Compute normalization stats in a streaming pass."""

    if method not in {"min_max", "standard"}:
        raise ConfigError(
            "Snapshot streaming supports normalization methods 'min_max' and 'standard' only",
        )

    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    mean_vals: Optional[np.ndarray] = None
    m2_vals: Optional[np.ndarray] = None
    count = 0

    for x_chunk, _, _, _, _ in iter_snapshot_batches(dataset, start_index, end_index):
        if x_chunk.size == 0:
            continue
        x_non_mask, _ = _strip_mask_channels(x_chunk, mask_start, mask_count)
        x_flat = x_non_mask.reshape(x_non_mask.shape[0], -1).astype("float64")

        if method == "min_max":
            batch_min = np.min(x_flat, axis=0)
            batch_max = np.max(x_flat, axis=0)
            if min_vals is None or max_vals is None:
                min_vals = batch_min
                max_vals = batch_max
            else:
                assert min_vals is not None
                assert max_vals is not None
                min_vals = np.minimum(min_vals, batch_min)
                max_vals = np.maximum(max_vals, batch_max)
        else:
            batch_count = x_flat.shape[0]
            batch_mean = np.mean(x_flat, axis=0)
            batch_m2 = np.sum((x_flat - batch_mean) ** 2, axis=0)

            if mean_vals is None:
                mean_vals = batch_mean
                m2_vals = batch_m2
                count = batch_count
            else:
                delta = batch_mean - mean_vals
                total = count + batch_count
                mean_vals = mean_vals + delta * (batch_count / float(total))
                if m2_vals is None:
                    m2_vals = batch_m2
                else:
                    m2_vals = m2_vals + batch_m2 + delta * delta * (count * batch_count / float(total))
                count = total

    if method == "min_max":
        if min_vals is None or max_vals is None:
            raise ValueError("Failed to compute min/max normalization stats")
        return NormalizationStats(method=method, min=min_vals, max=max_vals)

    if mean_vals is None or m2_vals is None or count <= 0:
        raise ValueError("Failed to compute standard normalization stats")
    std_vals = np.sqrt(m2_vals / float(count))
    return NormalizationStats(method=method, mean=mean_vals, std=std_vals)


def save_normalization_stats(path: str, stats: NormalizationStats) -> None:
    payload = {"method": np.asarray([stats.method])}
    if stats.min is not None:
        payload["min"] = stats.min
    if stats.max is not None:
        payload["max"] = stats.max
    if stats.mean is not None:
        payload["mean"] = stats.mean
    if stats.std is not None:
        payload["std"] = stats.std
    np.savez_compressed(path, **payload)


def load_normalization_stats(path: str) -> NormalizationStats:
    with np.load(path) as npz:
        method_arr = npz["method"]
        method = str(method_arr[0]) if method_arr.size > 0 else ""
        min_vals = npz["min"] if "min" in npz else None
        max_vals = npz["max"] if "max" in npz else None
        mean_vals = npz["mean"] if "mean" in npz else None
        std_vals = npz["std"] if "std" in npz else None
    return NormalizationStats(method=method, min=min_vals, max=max_vals, mean=mean_vals, std=std_vals)


@dataclass(frozen=True)
class LabelDistribution:
    """Label distribution for two-head classification outputs."""

    up_counts: Dict[int, int]
    down_counts: Dict[int, int]
    total_samples: int
    num_classes: int


def compute_label_distribution(
    dataset: SnapshotDataset,
    start_index: int,
    end_index: int,
    num_classes: int,
) -> LabelDistribution:
    """Compute label distribution from snapshot dataset by streaming chunks.

    This function counts labels in a single pass without loading all labels
    into memory, making it suitable for large datasets.

    Parameters
    ----------
    dataset:
        Snapshot dataset to iterate.
    start_index:
        Start sample index (inclusive).
    end_index:
        End sample index (exclusive).
    num_classes:
        Number of classes per head (must match model output).

    Returns
    -------
    LabelDistribution:
        Counts for each class index in both up and down heads.

    Raises
    ------
    ValueError:
        If num_classes < 1 or if no samples are found.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")

    up_counts: Dict[int, int] = {c: 0 for c in range(num_classes)}
    down_counts: Dict[int, int] = {c: 0 for c in range(num_classes)}
    total_samples = 0

    for _, y_up_chunk, y_down_chunk, _, _ in iter_snapshot_batches(
        dataset, start_index, end_index
    ):
        for y_up_val in y_up_chunk:
            class_idx = int(y_up_val)
            if 0 <= class_idx < num_classes:
                up_counts[class_idx] += 1
            total_samples += 1

        for y_down_val in y_down_chunk:
            class_idx = int(y_down_val)
            if 0 <= class_idx < num_classes:
                down_counts[class_idx] += 1

    if total_samples == 0:
        raise ValueError(
            "No samples found in dataset range; cannot compute label distribution"
        )

    logger.info(
        "Computed label distribution: total_samples=%s, up_counts=%s, down_counts=%s",
        total_samples,
        up_counts,
        down_counts,
    )

    return LabelDistribution(
        up_counts=up_counts,
        down_counts=down_counts,
        total_samples=total_samples,
        num_classes=num_classes,
    )


def save_label_stats_to_manifest(
    context: SnapshotContext,
    manifest: Dict[str, Any],
    distribution: LabelDistribution,
    split_name: str,
) -> None:
    """Save label distribution to manifest for a given split.

    Parameters
    ----------
    context:
        Snapshot context for saving manifest.
    manifest:
        Manifest dict to update.
    distribution:
        Label distribution to save.
    split_name:
        Name of split (e.g., 'train', 'val').
    """
    label_stats = manifest.get("label_stats", {})
    if not isinstance(label_stats, dict):
        label_stats = {}

    label_stats[split_name] = {
        "up_counts": {str(k): v for k, v in distribution.up_counts.items()},
        "down_counts": {str(k): v for k, v in distribution.down_counts.items()},
        "total_samples": distribution.total_samples,
        "num_classes": distribution.num_classes,
    }

    manifest["label_stats"] = label_stats
    save_manifest(context, manifest)

    logger.info(
        "Saved label stats for split '%s' to manifest: total_samples=%s",
        split_name,
        distribution.total_samples,
    )


def load_label_stats_from_manifest(
    manifest: Dict[str, Any],
    split_name: str,
) -> Optional[LabelDistribution]:
    """Load label distribution from manifest for a given split.

    Parameters
    ----------
    manifest:
        Manifest dict to read from.
    split_name:
        Name of split (e.g., 'train', 'val').

    Returns
    -------
    Optional[LabelDistribution]:
        Label distribution if found in manifest, None otherwise.
    """
    label_stats = manifest.get("label_stats", {})
    if not isinstance(label_stats, dict):
        return None

    split_stats = label_stats.get(split_name)
    if not isinstance(split_stats, dict):
        return None

    try:
        up_counts = {int(k): int(v) for k, v in split_stats["up_counts"].items()}
        down_counts = {int(k): int(v) for k, v in split_stats["down_counts"].items()}
        total_samples = int(split_stats["total_samples"])
        num_classes = int(split_stats["num_classes"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse label stats for split '%s': %s",
            split_name,
            exc,
        )
        return None

    logger.info(
        "Loaded label stats for split '%s' from manifest: total_samples=%s",
        split_name,
        total_samples,
    )

    return LabelDistribution(
        up_counts=up_counts,
        down_counts=down_counts,
        total_samples=total_samples,
        num_classes=num_classes,
    )


def _apply_normalization(
    x: np.ndarray,
    stats: Optional[NormalizationStats],
    mask_start: int,
    mask_count: int,
) -> np.ndarray:
    if stats is None:
        return x

    x_non_mask, mask = _strip_mask_channels(x, mask_start, mask_count)
    x_flat = x_non_mask.reshape(x_non_mask.shape[0], -1)

    if stats.method == "min_max":
        if stats.min is None or stats.max is None:
            raise ValueError("Missing min/max normalization stats")
        denom = stats.max - stats.min
        denom = np.where(denom == 0, 1.0, denom)
        x_norm = (x_flat - stats.min) / denom
    elif stats.method == "standard":
        if stats.mean is None or stats.std is None:
            raise ValueError("Missing mean/std normalization stats")
        std = np.where(stats.std == 0, 1.0, stats.std)
        x_norm = (x_flat - stats.mean) / std
    else:
        raise ValueError(f"Unsupported normalization method for snapshot dataset: {stats.method}")

    x_norm = x_norm.reshape(x_non_mask.shape).astype("float32")
    if mask is None:
        return x_norm

    left = x_norm[..., :mask_start]
    right = x_norm[..., mask_start:]
    return np.concatenate([left, mask, right], axis=-1)


def _strip_mask_channels(
    x: np.ndarray,
    mask_start: int,
    mask_count: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mask_count <= 0:
        return x, None
    mask_end = mask_start + mask_count
    mask = x[..., mask_start:mask_end]
    x_non_mask = np.concatenate([x[..., :mask_start], x[..., mask_end:]], axis=-1)
    return x_non_mask, mask


def _compute_current_day(dataset: SnapshotDataset, start_index: int, end_index: int) -> int:
    max_day = None
    for _, _, _, anchor_ts, _ in iter_snapshot_batches(dataset, start_index, end_index):
        if anchor_ts.size == 0:
            continue
        days = (anchor_ts // 86400).astype("int64")
        batch_max = int(days.max())
        if max_day is None or batch_max > max_day:
            max_day = batch_max
    if max_day is None:
        raise ValueError("Failed to compute current_day for sample weighting")
    return max_day


def _align_multi_asset_records(
    asset_records: Dict[str, List[SnapshotRecord]],
    assets: List[str],
    target_asset: str,
    alignment_cfg: Dict[str, Any],
    representation: str,
    cadence_seconds: int,
    hybrid_levels: Optional[int],
    fail_on_invalid: bool,
) -> List[MultiAssetSnapshotRecord]:
    target_records = asset_records.get(target_asset, [])
    if not target_records:
        return []

    method = str(alignment_cfg["method"])
    missing_policy = str(alignment_cfg["missing_policy"])
    max_gap_seconds = int(alignment_cfg["max_gap_seconds"])
    bucket_tolerance_seconds = float(alignment_cfg["bucket_tolerance_seconds"])

    if method not in {"interpolate", "bucket"}:
        raise ValueError("data.asset_pairs.alignment.method must be 'interpolate' or 'bucket'")
    if missing_policy not in {"forward_fill", "skip", "error"}:
        raise ValueError(
            "data.asset_pairs.alignment.missing_policy must be 'forward_fill', 'skip', or 'error'",
        )
    if max_gap_seconds <= 0:
        raise ValueError("data.asset_pairs.alignment.max_gap_seconds must be positive")
    if bucket_tolerance_seconds < 0:
        raise ValueError("data.asset_pairs.alignment.bucket_tolerance_seconds must be >= 0")

    target_times = np.asarray([rec.timestamp for rec in target_records], dtype="datetime64[ns]")

    aligned_by_asset: Dict[str, List[Optional[SnapshotRecord]]] = {}
    for asset in assets:
        if asset == target_asset:
            continue
        records = asset_records.get(asset, [])
        aligned_by_asset[asset] = _align_asset_records(
            records=records,
            target_times=target_times,
            method=method,
            representation=representation,
            missing_policy=missing_policy,
            max_gap_seconds=max_gap_seconds,
            bucket_tolerance_seconds=bucket_tolerance_seconds,
            cadence_seconds=cadence_seconds,
            hybrid_levels=hybrid_levels,
            fail_on_invalid=fail_on_invalid,
            asset_name=asset,
        )

    multi_records: List[MultiAssetSnapshotRecord] = []
    for idx, target_rec in enumerate(target_records):
        asset_snapshots: Dict[str, SnapshotRecord] = {target_asset: target_rec}
        missing_assets = []
        for asset in assets:
            if asset == target_asset:
                continue
            aligned_rec = aligned_by_asset[asset][idx]
            if aligned_rec is None:
                missing_assets.append(asset)
                continue
            asset_snapshots[asset] = aligned_rec

        if missing_assets:
            message = (
                "Missing aligned correlated asset snapshots for timestamp="
                f"{target_rec.timestamp}; missing_assets={missing_assets}"
            )
            if missing_policy == "error" or fail_on_invalid:
                raise ValueError(message)
            logger.warning(message)
            continue

        multi_records.append(
            MultiAssetSnapshotRecord(
                timestamp=target_rec.timestamp,
                asset_snapshots=asset_snapshots,
            )
        )

    return multi_records


def _align_asset_records(
    *,
    records: List[SnapshotRecord],
    target_times: np.ndarray,
    method: str,
    representation: str,
    missing_policy: str,
    max_gap_seconds: int,
    bucket_tolerance_seconds: float,
    cadence_seconds: int,
    hybrid_levels: Optional[int],
    fail_on_invalid: bool,
    asset_name: str,
) -> List[Optional[SnapshotRecord]]:
    if not records:
        if missing_policy in {"forward_fill", "skip"}:
            return [None for _ in range(len(target_times))]
        raise ValueError(f"No records available for asset {asset_name} during alignment")

    if method == "interpolate":
        return _align_asset_interpolate(
            records=records,
            target_times=target_times,
            representation=representation,
            missing_policy=missing_policy,
            max_gap_seconds=max_gap_seconds,
            hybrid_levels=hybrid_levels,
            fail_on_invalid=fail_on_invalid,
            asset_name=asset_name,
        )

    return _align_asset_bucket(
        records=records,
        target_times=target_times,
        representation=representation,
        missing_policy=missing_policy,
        max_gap_seconds=max_gap_seconds,
        bucket_tolerance_seconds=bucket_tolerance_seconds,
        cadence_seconds=cadence_seconds,
        hybrid_levels=hybrid_levels,
        fail_on_invalid=fail_on_invalid,
        asset_name=asset_name,
    )


def _align_asset_interpolate(
    *,
    records: List[SnapshotRecord],
    target_times: np.ndarray,
    representation: str,
    missing_policy: str,
    max_gap_seconds: int,
    hybrid_levels: Optional[int],
    fail_on_invalid: bool,
    asset_name: str,
) -> List[Optional[SnapshotRecord]]:
    times = np.asarray([rec.timestamp for rec in records], dtype="datetime64[ms]")
    times_ms = times.astype("int64")
    target_ms = target_times.astype("datetime64[ms]").astype("int64")

    values, confidences = _extract_record_values(records, representation)
    if values is None:
        if missing_policy in {"forward_fill", "skip"}:
            return [None for _ in range(len(target_times))]
        raise ValueError(f"Missing values for asset {asset_name} during alignment")

    aligned: List[Optional[SnapshotRecord]] = []
    max_gap_ms = int(max_gap_seconds * 1000)

    for t_ms, t_dt in zip(target_ms, target_times):
        idx = int(np.searchsorted(times_ms, t_ms, side="left"))
        if idx < len(times_ms) and times_ms[idx] == t_ms:
            record = records[idx]
            value, confidence = _extract_single_record_value(record, representation)
            aligned.append(
                _build_aligned_record(
                    t_dt,
                    value,
                    representation,
                    confidence,
                    observed=record.observed,
                )
            )
            continue

        left = idx - 1
        right = idx
        if left < 0 or right >= len(times_ms):
            if missing_policy == "forward_fill" and left >= 0:
                gap_from_left = int(t_ms - times_ms[left])
                if gap_from_left <= max_gap_ms:
                    value = values[left]
                    confidence = confidences[left]
                    aligned.append(
                        _build_aligned_record(
                            t_dt,
                            value,
                            representation,
                            float(confidence),
                            observed=False,
                        )
                    )
                    continue
            aligned.append(
                _handle_missing_alignment(
                    t_dt,
                    representation,
                    missing_policy,
                    fail_on_invalid,
                    asset_name,
                    hybrid_levels,
                )
            )
            continue

        gap_ms = int(times_ms[right] - times_ms[left])
        if gap_ms <= 0 or gap_ms > max_gap_ms:
            if missing_policy == "forward_fill":
                gap_from_left = int(t_ms - times_ms[left])
                if gap_from_left <= max_gap_ms:
                    value = values[left]
                    confidence = confidences[left]
                    aligned.append(
                        _build_aligned_record(
                            t_dt,
                            value,
                            representation,
                            float(confidence),
                            observed=False,
                        )
                    )
                    continue
            aligned.append(
                _handle_missing_alignment(
                    t_dt,
                    representation,
                    missing_policy,
                    fail_on_invalid,
                    asset_name,
                    hybrid_levels,
                )
            )
            continue

        alpha = float(t_ms - times_ms[left]) / float(gap_ms)
        value = values[left] + alpha * (values[right] - values[left])
        confidence = float(confidences[left] + alpha * (confidences[right] - confidences[left]))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        aligned.append(
            _build_aligned_record(
                t_dt,
                value,
                representation,
                confidence,
                observed=False,
            )
        )

    return aligned


def _align_asset_bucket(
    *,
    records: List[SnapshotRecord],
    target_times: np.ndarray,
    representation: str,
    missing_policy: str,
    max_gap_seconds: int,
    bucket_tolerance_seconds: float,
    cadence_seconds: int,
    hybrid_levels: Optional[int],
    fail_on_invalid: bool,
    asset_name: str,
) -> List[Optional[SnapshotRecord]]:
    if cadence_seconds <= 0:
        raise ValueError("data.time_range.cadence_seconds must be positive")

    cadence_ms = int(cadence_seconds * 1000)
    tolerance_ms = float(bucket_tolerance_seconds * 1000.0)

    times = np.asarray([rec.timestamp for rec in records], dtype="datetime64[ms]")
    times_ms = times.astype("int64")

    bucket_map: Dict[int, Tuple[float, SnapshotRecord]] = {}
    for rec, t_ms in zip(records, times_ms):
        bucket = int((t_ms // cadence_ms) * cadence_ms)
        distance = float(abs(t_ms - bucket))
        if distance > tolerance_ms:
            continue
        existing = bucket_map.get(bucket)
        if existing is None or distance < existing[0]:
            bucket_map[bucket] = (distance, rec)

    aligned: List[Optional[SnapshotRecord]] = []
    target_ms = target_times.astype("datetime64[ms]").astype("int64")
    max_gap_ms = int(max_gap_seconds * 1000)
    last_record: Optional[SnapshotRecord] = None
    last_record_time_ms: Optional[int] = None
    for t_ms, t_dt in zip(target_ms, target_times):
        bucket = int((t_ms // cadence_ms) * cadence_ms)
        record_entry = bucket_map.get(bucket)
        if record_entry is None:
            if missing_policy == "forward_fill" and last_record is not None and last_record_time_ms is not None:
                gap_ms = int(t_ms - last_record_time_ms)
                if 0 <= gap_ms <= max_gap_ms:
                    value, confidence = _extract_single_record_value(last_record, representation)
                    aligned.append(
                        _build_aligned_record(
                            t_dt,
                            value,
                            representation,
                            confidence,
                            observed=False,
                        )
                    )
                    continue
            aligned.append(
                _handle_missing_alignment(
                    t_dt,
                    representation,
                    missing_policy,
                    fail_on_invalid,
                    asset_name,
                    hybrid_levels,
                )
            )
            continue
        record = record_entry[1]
        value, confidence = _extract_single_record_value(record, representation)
        aligned.append(
            _build_aligned_record(
                t_dt,
                value,
                representation,
                confidence,
                observed=record.observed,
            )
        )
        last_record = record
        last_record_time_ms = int(record.timestamp.astype("datetime64[ms]").astype("int64"))

    return aligned


def _extract_record_values(
    records: List[SnapshotRecord],
    representation: str,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    values_list = []
    confidences = np.array([rec.confidence for rec in records], dtype="float64")

    for rec in records:
        value, _ = _extract_single_record_value(rec, representation)
        if value is None:
            return None, confidences
        values_list.append(value)

    values = np.asarray(values_list, dtype="float64")
    return values, confidences


def _extract_single_record_value(
    record: SnapshotRecord,
    representation: str,
) -> Tuple[np.ndarray, float]:
    if representation == "hybrid":
        if record.hybrid_snapshot is None:
            raise ValueError("Missing hybrid data for alignment")
        value = np.asarray(record.hybrid_snapshot, dtype="float64")
        return value, float(record.confidence)

    features = record.snapshot_features
    if len(features) < 4:
        raise ValueError("snapshot_features must contain at least 4 values for alignment")
    value = np.asarray(features[:4], dtype="float64")
    return value, float(record.confidence)


def _build_aligned_record(
    timestamp: np.datetime64,
    value: np.ndarray,
    representation: str,
    confidence: float,
    observed: bool,
) -> SnapshotRecord:
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if representation == "hybrid":
        hybrid_snapshot = np.asarray(value, dtype="float32")
        if hybrid_snapshot.ndim != 2 or hybrid_snapshot.shape[1] != 4:
            raise ValueError("Aligned hybrid snapshot must have shape (L, 4)")
        bid_price = float(hybrid_snapshot[0, 0]) if hybrid_snapshot.shape[0] > 0 else 0.0
        bid_qty = float(hybrid_snapshot[0, 1]) if hybrid_snapshot.shape[0] > 0 else 0.0
        ask_price = float(hybrid_snapshot[0, 2]) if hybrid_snapshot.shape[0] > 0 else 0.0
        ask_qty = float(hybrid_snapshot[0, 3]) if hybrid_snapshot.shape[0] > 0 else 0.0
        mid_price = 0.5 * (bid_price + ask_price) if bid_price > 0 and ask_price > 0 else 0.0
        return SnapshotRecord(
            timestamp=timestamp,
            snapshot_features=[bid_price, bid_qty, ask_price, ask_qty],
            depth=None,
            mid_price=mid_price,
            hybrid_snapshot=hybrid_snapshot.astype("float32"),
            volume_proxy=0.0,
            confidence=confidence,
            gap_reset=False,
            observed=observed,
        )

    features = np.asarray(value, dtype="float32")
    if features.shape[0] < 4:
        raise ValueError("Aligned top_of_book features must have at least 4 values")
    bid_price = float(features[0])
    bid_qty = float(features[1])
    ask_price = float(features[2])
    ask_qty = float(features[3])
    mid_price = 0.5 * (bid_price + ask_price) if bid_price > 0 and ask_price > 0 else 0.0
    return SnapshotRecord(
        timestamp=timestamp,
        snapshot_features=[bid_price, bid_qty, ask_price, ask_qty],
        depth=None,
        mid_price=mid_price,
        hybrid_snapshot=None,
        volume_proxy=0.0,
        confidence=confidence,
        gap_reset=False,
        observed=observed,
    )


def _populate_hybrid_snapshots(
    records: List[SnapshotRecord],
    config: Dict[str, Any],
    *,
    fail_on_invalid: bool,
    asset_name: str,
) -> None:
    hybrid_levels = get_hybrid_output_shape(config)
    for rec in records:
        if rec.hybrid_snapshot is not None:
            continue
        if rec.depth is None:
            message = f"Missing depth data for asset={asset_name} while building hybrid snapshots"
            if fail_on_invalid:
                raise ValueError(message)
            logger.warning(message)
            fallback = np.zeros((hybrid_levels, 4), dtype="float32")
            if len(rec.snapshot_features) >= 4:
                bid_price, bid_qty, ask_price, ask_qty = rec.snapshot_features[:4]
                fallback[0, 0] = float(bid_price)
                fallback[0, 1] = float(bid_qty)
                fallback[0, 2] = float(ask_price)
                fallback[0, 3] = float(ask_qty)
            rec.hybrid_snapshot = fallback
            continue
        rec.hybrid_snapshot = aggregate_snapshot_to_hybrid(
            bid_prices=rec.depth["bid_prices"],
            bid_quantities=rec.depth["bid_quantities"],
            ask_prices=rec.depth["ask_prices"],
            ask_quantities=rec.depth["ask_quantities"],
            config=config,
        ).astype("float32")


def _zero_pad_record(
    timestamp: np.datetime64,
    representation: str,
    hybrid_levels: Optional[int],
) -> SnapshotRecord:
    if representation == "hybrid":
        if hybrid_levels is None or hybrid_levels <= 0:
            raise ValueError("hybrid_levels must be provided for zero-padding hybrid snapshots")
        return SnapshotRecord(
            timestamp=timestamp,
            snapshot_features=[0.0, 0.0, 0.0, 0.0],
            depth=None,
            mid_price=0.0,
            hybrid_snapshot=np.zeros((hybrid_levels, 4), dtype="float32"),
            volume_proxy=0.0,
            confidence=0.0,
            gap_reset=False,
            observed=False,
        )

    return SnapshotRecord(
        timestamp=timestamp,
        snapshot_features=[0.0, 0.0, 0.0, 0.0],
        depth=None,
        mid_price=0.0,
        hybrid_snapshot=None,
        volume_proxy=0.0,
        confidence=0.0,
        gap_reset=False,
        observed=False,
    )


def _handle_missing_alignment(
    timestamp: np.datetime64,
    representation: str,
    missing_policy: str,
    fail_on_invalid: bool,
    asset_name: str,
    hybrid_levels: Optional[int],
) -> Optional[SnapshotRecord]:
    message = f"Missing aligned data for asset={asset_name} at timestamp={timestamp}"
    if missing_policy == "error" or fail_on_invalid:
        raise ValueError(message)
    logger.warning(message)
    return None


class _DutyCycleAccumulator:
    def __init__(self, bins: int = 200) -> None:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self._bins = bins
        self._counts = np.zeros(bins, dtype="int64")
        self._min: Optional[float] = None
        self._total = 0

    def add_values(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        values = np.clip(values, 0.0, 1.0)
        current_min = float(values.min())
        self._min = current_min if self._min is None else min(self._min, current_min)
        self._total += int(values.size)
        indices = np.clip((values * (self._bins - 1)).astype("int64"), 0, self._bins - 1)
        counts = np.bincount(indices, minlength=self._bins).astype("int64")
        self._counts += counts

    def summary(self) -> Optional[Tuple[float, float, float]]:
        if self._total == 0 or self._min is None:
            return None
        median = self._quantile(0.5)
        p95 = self._quantile(0.95)
        return float(self._min), float(median), float(p95)

    def _quantile(self, q: float) -> float:
        target = max(1, int(math.ceil(q * self._total)))
        cumulative = np.cumsum(self._counts)
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), self._bins - 1)
        return idx / float(self._bins - 1)


def _build_snapshot_chunks(
    config: Dict[str, Any],
    context: SnapshotContext,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    writer = None
    try:
        from observability.run_state import get_run_state_writer

        writer = get_run_state_writer()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize run-state writer for snapshot build: %s", exc)
        writer = None
    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    start_date = str(time_range_cfg["start_date"])
    end_date = str(time_range_cfg["end_date"])
    chunk_hours = int(data_cfg["ingestion"]["chunk_hours"])
    cadence_seconds = int(time_range_cfg["cadence_seconds"])

    if chunk_hours <= 0:
        raise ValueError("data.ingestion.chunk_hours must be positive")

    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])
    correlated_assets = [str(a) for a in asset_pairs_cfg["correlated_assets"]]
    assets = [target_asset] + correlated_assets
    if not assets:
        raise ValueError("data.asset_pairs must define at least one asset for snapshot building")

    output_chunks = _generate_time_chunks(start_date, end_date, chunk_hours)
    output_boundaries = _build_output_boundaries(output_chunks)
    chunks_total = len(output_boundaries)
    chunks_processed = 0

    if writer is not None:
        try:
            writer.update_snapshot_progress(processed=0, total=chunks_total)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish initial snapshot progress: %s", exc)

    duty_cycle_stats = _DutyCycleAccumulator()

    def _update_duty_cycle_writer() -> None:
        if writer is None:
            return
        summary = duty_cycle_stats.summary()
        if summary is None:
            return
        try:
            writer.update_duty_cycle_stats(*summary)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish duty-cycle summary to run-state", exc_info=True)

    manifest["complete"] = False
    save_manifest(context, manifest)

    os.makedirs(os.path.join(context.snapshot_dir, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(context.snapshot_dir, "series"), exist_ok=True)

    existing_entries = _existing_chunk_entries(context, manifest)
    existing_series_entries = _existing_series_entries(context, manifest)
    for boundary in output_boundaries:
        key = (boundary["start_str"], boundary["end_str"])
        if key in existing_entries:
            boundary["cached"] = True

    series_meta = manifest.get("series")
    if not isinstance(series_meta, dict):
        series_meta = {}
    series_meta["target_asset"] = target_asset
    series_meta["cadence_seconds"] = cadence_seconds
    manifest["series"] = series_meta

    sample_builder = _create_sample_builder(config)
    gap_handlers = {asset: _create_gap_handler(config) for asset in assets}
    validation_cfg = data_cfg["validation"]
    fail_on_invalid = bool(validation_cfg["fail_on_invalid"])

    current_chunk_idx = 0
    chunk_samples: List[SampleRecord] = []

    def flush_chunk(index: int) -> None:
        nonlocal chunk_samples
        nonlocal chunks_processed
        if index >= len(output_boundaries):
            chunk_samples = []
            return

        boundary = output_boundaries[index]
        if boundary["cached"]:
            key = (boundary["start_str"], boundary["end_str"])
            entry = existing_entries.get(key, {})
            file_rel = entry.get("file")
            if file_rel:
                file_path = os.path.join(context.snapshot_dir, file_rel)
                try:
                    with np.load(file_path) as npz:
                        if "duty_cycle" in npz:
                            duty_cycle_stats.add_values(npz["duty_cycle"])
                            _update_duty_cycle_writer()
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to load duty_cycle from cached chunk: %s", file_path)
            chunk_samples = []
            chunks_processed += 1
            if writer is not None:
                try:
                    writer.update_snapshot_progress(processed=chunks_processed, total=chunks_total)
                except Exception:
                    pass
            return

        if not chunk_samples:
            chunk_samples = []
            chunks_processed += 1
            if writer is not None:
                try:
                    writer.update_snapshot_progress(processed=chunks_processed, total=chunks_total)
                except Exception:
                    pass
            return

        chunk_start = boundary["start_str"]
        chunk_end = boundary["end_str"]
        filename = _chunk_filename(chunk_start, chunk_end)
        file_rel = os.path.join("chunks", filename)
        file_path = os.path.join(context.snapshot_dir, file_rel)

        x = np.stack([s.x for s in chunk_samples]).astype("float32")
        y_up = np.asarray([s.y_up for s in chunk_samples], dtype="int64")
        y_down = np.asarray([s.y_down for s in chunk_samples], dtype="int64")
        anchor_ts = np.asarray([s.anchor_ts_seconds for s in chunk_samples], dtype="int64")
        duty_cycle = np.asarray([s.duty_cycle for s in chunk_samples], dtype="float32")

        duty_cycle_stats.add_values(duty_cycle)
        _update_duty_cycle_writer()

        np.savez_compressed(
            file_path,
            x=x,
            y_up=y_up,
            y_down=y_down,
            anchor_ts=anchor_ts,
            duty_cycle=duty_cycle,
        )

        entry = {
            "start": chunk_start,
            "end": chunk_end,
            "file": file_rel,
            "num_samples": int(x.shape[0]),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        _upsert_chunk_entry(manifest, entry)
        save_manifest(context, manifest)
        chunk_samples = []

        chunks_processed += 1
        if writer is not None:
            try:
                writer.update_snapshot_progress(processed=chunks_processed, total=chunks_total)
            except Exception:
                pass

    current_chunk_key: Optional[Tuple[str, str]] = None
    chunk_rows: Dict[str, List[List[Any]]] = {}

    def process_chunk(chunk_key: Tuple[str, str], chunk_rows_by_asset: Dict[str, List[List[Any]]]) -> None:
        nonlocal current_chunk_idx
        missing_assets = [asset for asset in assets if asset not in chunk_rows_by_asset]
        if missing_assets:
            message = f"Missing chunk data for assets={missing_assets} in chunk {chunk_key}"
            if fail_on_invalid:
                raise ValueError(message)
            logger.warning(message)
            return

        order_book_cfg = data_cfg["order_book"]
        representation = str(order_book_cfg["representation"])
        if representation == "full":
            representation = "hybrid"

        asset_records: Dict[str, List[SnapshotRecord]] = {}
        for asset in assets:
            rows = chunk_rows_by_asset.get(asset, [])
            chunk = OrderBookChunk(
                asset=asset,
                chunk_start=chunk_key[0],
                chunk_end=chunk_key[1],
                rows=rows,
            )
            compute_volume_proxy = asset == target_asset
            records = _build_snapshots_from_rows(chunk, config, compute_volume_proxy=compute_volume_proxy)
            filled_records = list(gap_handlers[asset].iter_gap_handled(records))
            if representation == "hybrid":
                _populate_hybrid_snapshots(
                    filled_records,
                    config,
                    fail_on_invalid=fail_on_invalid,
                    asset_name=asset,
                )
            asset_records[asset] = filled_records

        hybrid_levels = get_hybrid_output_shape(config) if representation == "hybrid" else None

        alignment_cfg = asset_pairs_cfg["alignment"]
        multi_records = GapHandler.align_multi_asset(
            asset_records=asset_records,
            assets=assets,
            target_asset=target_asset,
            alignment_cfg=alignment_cfg,
            representation=representation,
            cadence_seconds=cadence_seconds,
            hybrid_levels=hybrid_levels,
            fail_on_invalid=fail_on_invalid,
        )
        series_timestamps: List[int] = []
        series_mid_prices: List[float] = []
        series_volumes: List[float] = []

        for snapshot in multi_records:
            target_snapshot = snapshot.asset_snapshots.get(target_asset)
            if target_snapshot is None:
                raise ValueError(
                    f"Aligned snapshot missing target asset '{target_asset}' for chunk {chunk_key}",
                )

            series_timestamps.append(
                int(snapshot.timestamp.astype("datetime64[s]").astype("int64"))
            )
            series_mid_prices.append(float(target_snapshot.mid_price))
            series_volumes.append(float(target_snapshot.volume_proxy))

            for sample in sample_builder.add_snapshot(snapshot):
                anchor_ts = sample.anchor_ts_seconds

                while current_chunk_idx < len(output_boundaries) and anchor_ts > output_boundaries[current_chunk_idx][
                    "end_ts"
                ]:
                    flush_chunk(current_chunk_idx)
                    current_chunk_idx += 1

                if current_chunk_idx >= len(output_boundaries):
                    break

                if output_boundaries[current_chunk_idx]["cached"]:
                    continue

                chunk_samples.append(sample)

        series_filename = _chunk_filename(chunk_key[0], chunk_key[1])
        series_rel = os.path.join("series", series_filename)
        series_path = os.path.join(context.snapshot_dir, series_rel)
        series_key = (chunk_key[0], chunk_key[1])

        if series_key not in existing_series_entries or not os.path.exists(series_path):
            series_ts_arr = np.asarray(series_timestamps, dtype="int64")
            series_mid_arr = np.asarray(series_mid_prices, dtype="float64")
            series_vol_arr = np.asarray(series_volumes, dtype="float64")

            np.savez_compressed(
                series_path,
                timestamps=series_ts_arr,
                mid_prices=series_mid_arr,
                volumes=series_vol_arr,
            )

        series_entry = {
            "start": chunk_key[0],
            "end": chunk_key[1],
            "file": series_rel,
            "num_snapshots": int(len(series_timestamps)),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        _upsert_series_entry(manifest, series_entry)
        save_manifest(context, manifest)

    for chunk in stream_order_book_chunks_by_time(config, assets_override=assets):
        key = (chunk.chunk_start, chunk.chunk_end)
        if current_chunk_key is None:
            current_chunk_key = key

        if key != current_chunk_key:
            process_chunk(current_chunk_key, chunk_rows)
            chunk_rows = {}
            current_chunk_key = key

        chunk_rows[chunk.asset] = chunk.rows

        if len(chunk_rows) == len(assets):
            process_chunk(current_chunk_key, chunk_rows)
            chunk_rows = {}
            current_chunk_key = None

    if current_chunk_key is not None and chunk_rows:
        process_chunk(current_chunk_key, chunk_rows)

    if current_chunk_idx < len(output_boundaries):
        flush_chunk(current_chunk_idx)

    manifest["complete"] = True
    save_manifest(context, manifest)
    return manifest




def _build_output_boundaries(output_chunks: List[Tuple[str, str, bool]]) -> List[Dict[str, Any]]:
    boundaries: List[Dict[str, Any]] = []
    for start_str, end_str, _ in output_chunks:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        boundaries.append(
            {
                "start_str": start_str,
                "end_str": end_str,
                "start_ts": int(start_dt.timestamp()),
                "end_ts": int(end_dt.timestamp()),
                "cached": False,
            }
        )
    return boundaries


def _existing_chunk_entries(context: SnapshotContext, manifest: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in manifest.get("chunks", []) or []:
        start = entry.get("start")
        end = entry.get("end")
        file_rel = entry.get("file")
        if not start or not end or not file_rel:
            continue
        file_path = os.path.join(context.snapshot_dir, file_rel)
        if not os.path.exists(file_path):
            continue
        existing[(start, end)] = entry
    return existing


def _existing_series_entries(context: SnapshotContext, manifest: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    series_meta = manifest.get("series", {})
    if not isinstance(series_meta, dict):
        return existing
    for entry in series_meta.get("chunks", []) or []:
        start = entry.get("start")
        end = entry.get("end")
        file_rel = entry.get("file")
        if not start or not end or not file_rel:
            continue
        file_path = os.path.join(context.snapshot_dir, file_rel)
        if not os.path.exists(file_path):
            continue
        existing[(start, end)] = entry
    return existing


def _chunk_filename(start_str: str, end_str: str) -> str:
    safe_start = start_str.replace(" ", "_").replace(":", "-")
    safe_end = end_str.replace(" ", "_").replace(":", "-")
    return f"{safe_start}_{safe_end}.npz"


def _upsert_chunk_entry(manifest: Dict[str, Any], entry: Dict[str, Any]) -> None:
    chunks = manifest.get("chunks", []) or []
    replaced = False
    for idx, existing in enumerate(chunks):
        if existing.get("start") == entry.get("start") and existing.get("end") == entry.get("end"):
            chunks[idx] = entry
            replaced = True
            break
    if not replaced:
        chunks.append(entry)
    chunks.sort(key=lambda item: item.get("start") or "")
    manifest["chunks"] = chunks


def _upsert_series_entry(manifest: Dict[str, Any], entry: Dict[str, Any]) -> None:
    series_meta = manifest.get("series")
    if not isinstance(series_meta, dict):
        series_meta = {}
    chunks = series_meta.get("chunks", []) or []
    replaced = False
    for idx, existing in enumerate(chunks):
        if existing.get("start") == entry.get("start") and existing.get("end") == entry.get("end"):
            chunks[idx] = entry
            replaced = True
            break
    if not replaced:
        chunks.append(entry)
    chunks.sort(key=lambda item: item.get("start") or "")
    series_meta["chunks"] = chunks
    manifest["series"] = series_meta


def _create_sample_builder(config: Dict[str, Any]) -> StreamingSampleBuilder:
    model_cfg = config["model"]
    cnn_cfg = model_cfg["cnn"]
    cnn_layers = cnn_cfg["layers"]

    kernel_sizes = [l["kernel_size"] for l in cnn_layers]
    pool_sizes = [l["pool_size"] for l in cnn_layers if l.get("pool_size") is not None]

    heights = [int(k[0]) for k in kernel_sizes]
    widths = [int(k[1]) for k in kernel_sizes]

    pool_heights = [int(p[0]) for p in pool_sizes] if pool_sizes else [1]
    pool_widths = [int(p[1]) for p in pool_sizes] if pool_sizes else [1]

    min_height = 1
    for ph in pool_heights:
        min_height *= ph

    min_width = 1
    for pw in pool_widths:
        min_width *= pw

    height = max(max(heights), min_height)
    width = max(max(widths), min_width)

    data_cfg = config["data"]
    representation = str(data_cfg["order_book"]["representation"])
    if representation not in {"top_of_book", "hybrid", "full"}:
        raise ValueError("Unsupported data.order_book.representation in snapshot mode")
    if representation == "full":
        representation = "hybrid"

    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])
    correlated_assets = [str(a) for a in asset_pairs_cfg["correlated_assets"]]
    assets = [target_asset] + correlated_assets

    return StreamingSampleBuilder(
        config,
        representation=representation,
        height=height,
        width=width,
        assets=assets,
        target_asset=target_asset,
    )


def _create_gap_handler(config: Dict[str, Any]) -> GapHandler:
    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    validation_cfg = data_cfg["validation"]
    targets_cfg = config["targets"]
    labeling_cfg = targets_cfg["labeling"]
    alignment_cfg = data_cfg["asset_pairs"]["alignment"]

    cadence_seconds = int(time_range_cfg["cadence_seconds"])
    validation_max_gap_seconds = int(validation_cfg["max_gap_seconds"])
    check_missing_data = bool(validation_cfg["check_missing_data"])
    fail_on_invalid = bool(validation_cfg["fail_on_invalid"])
    handle_gaps = str(labeling_cfg["handle_gaps"])
    alignment_max_gap_seconds = int(alignment_cfg["max_gap_seconds"])

    if handle_gaps not in {"skip", "forward_fill", "interpolate"}:
        raise ValueError("targets.labeling.handle_gaps must be 'skip', 'forward_fill', or 'interpolate'")
    if alignment_max_gap_seconds <= 0:
        raise ValueError("data.asset_pairs.alignment.max_gap_seconds must be positive")

    return GapHandler(
        cadence_seconds=cadence_seconds,
        validation_max_gap_seconds=validation_max_gap_seconds,
        alignment_max_gap_seconds=alignment_max_gap_seconds,
        handle_gaps=handle_gaps,
        check_missing_data=check_missing_data,
        fail_on_invalid=fail_on_invalid,
    )


def _build_snapshots_from_rows(
    chunk: OrderBookChunk,
    config: Dict[str, Any],
    *,
    compute_volume_proxy: bool,
) -> List[SnapshotRecord]:
    data_cfg = config["data"]
    order_book_cfg = data_cfg["order_book"]
    representation = str(order_book_cfg["representation"])
    collect_full_depth = representation in {"hybrid", "full"}
    depth_levels = int(order_book_cfg["depth_levels"])
    if collect_full_depth and depth_levels <= 0:
        raise ValueError("data.order_book.depth_levels must be positive for hybrid representation")

    snapshots: Dict[Any, Dict[str, Any]] = {}

    for row_index, row in enumerate(chunk.rows):
        if len(row) < 6:
            raise ValueError(
                f"Order book row {row_index} has insufficient columns: expected >=6, got {len(row)}"
            )
        ts_value = row[0]
        state = snapshots.get(ts_value)
        if state is None:
            state = {
                "best_bid_price": None,
                "best_bid_qty": None,
                "best_ask_price": None,
                "best_ask_qty": None,
                "bid_levels": [],
                "ask_levels": [],
            }
            snapshots[ts_value] = state

        try:
            bid_price = float(row[1])
            bid_qty = float(row[2])
            ask_price = float(row[3])
            ask_qty = float(row[4])
        except (TypeError, ValueError, IndexError) as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse row {row_index}: {exc}") from exc

        if bid_price > 0.0 and bid_qty >= 0.0:
            current_bid = state["best_bid_price"]
            if current_bid is None or bid_price > current_bid:
                state["best_bid_price"] = bid_price
                state["best_bid_qty"] = bid_qty
            if collect_full_depth:
                state["bid_levels"].append((bid_price, bid_qty))

        if ask_price > 0.0 and ask_qty >= 0.0:
            current_ask = state["best_ask_price"]
            if current_ask is None or ask_price < current_ask:
                state["best_ask_price"] = ask_price
                state["best_ask_qty"] = ask_qty
            if collect_full_depth:
                state["ask_levels"].append((ask_price, ask_qty))

    if not snapshots:
        return []

    sorted_keys = sorted(snapshots.keys())
    normalized_ts = normalize_timestamp_array(sorted_keys)

    fe_cfg = config["preprocessing"]["feature_engineering"]
    feature_engineer = FeatureEngineer(config) if bool(fe_cfg["enabled"]) and compute_volume_proxy else None

    records: List[SnapshotRecord] = []
    for key, ts_norm in zip(sorted_keys, normalized_ts):
        state = snapshots[key]
        best_bid = state["best_bid_price"]
        best_ask = state["best_ask_price"]
        if best_bid is None or best_ask is None:
            continue
        if best_bid <= 0.0 or best_ask <= 0.0:
            continue

        best_bid_qty = state["best_bid_qty"] or 0.0
        best_ask_qty = state["best_ask_qty"] or 0.0

        mid_price = 0.5 * (best_bid + best_ask)
        snapshot_features = [
            float(best_bid),
            float(best_bid_qty),
            float(best_ask),
            float(best_ask_qty),
        ]

        depth_data = None
        hybrid_snapshot = None
        volume_proxy = 0.0
        if collect_full_depth:
            bid_levels_sorted = sorted(state["bid_levels"], key=lambda x: -x[0])
            ask_levels_sorted = sorted(state["ask_levels"], key=lambda x: x[0])

            bid_prices = np.zeros(depth_levels, dtype="float64")
            bid_quantities = np.zeros(depth_levels, dtype="float64")
            ask_prices = np.zeros(depth_levels, dtype="float64")
            ask_quantities = np.zeros(depth_levels, dtype="float64")

            for i, (p, q) in enumerate(bid_levels_sorted[:depth_levels]):
                bid_prices[i] = p
                bid_quantities[i] = q
            for i, (p, q) in enumerate(ask_levels_sorted[:depth_levels]):
                ask_prices[i] = p
                ask_quantities[i] = q

            depth_data = {
                "bid_prices": bid_prices,
                "bid_quantities": bid_quantities,
                "ask_prices": ask_prices,
                "ask_quantities": ask_quantities,
            }

            if representation in {"hybrid", "full"}:
                hybrid_snapshot = aggregate_snapshot_to_hybrid(
                    bid_prices=bid_prices,
                    bid_quantities=bid_quantities,
                    ask_prices=ask_prices,
                    ask_quantities=ask_quantities,
                    config=config,
                ).astype("float32")

            if feature_engineer is not None:
                volume_proxy = float(feature_engineer.compute_volume_proxy(depth_data))

        records.append(
            SnapshotRecord(
                timestamp=ts_norm,
                snapshot_features=snapshot_features,
                depth=depth_data,
                mid_price=mid_price,
                hybrid_snapshot=hybrid_snapshot,
                volume_proxy=volume_proxy,
                confidence=1.0,
                gap_reset=False,
                observed=True,
            )
        )

    return records


def _compute_intensity_bins(boundaries: List[float], max_up: float, max_down: float) -> Tuple[int, int]:
    up_intensity = max(max_up, 0.0)
    down_intensity = max(-max_down, 0.0)

    up_bin = 0
    for idx, boundary in enumerate(boundaries):
        if up_intensity <= boundary:
            up_bin = idx
            break
    else:
        up_bin = len(boundaries)

    down_bin = 0
    for idx, boundary in enumerate(boundaries):
        if down_intensity <= boundary:
            down_bin = idx
            break
    else:
        down_bin = len(boundaries)

    return int(up_bin), int(down_bin)


def _compute_feature_vector(
    feature_engineer: FeatureEngineer,
    anchor_record: SnapshotRecord,
    mid_prices: np.ndarray,
    volumes: np.ndarray,
    anchor_pos: int,
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    if anchor_record.depth is None:
        return None

    order_book_features = feature_engineer.compute_order_book_features(anchor_record.depth)

    momentum_window_seconds = int(
        config["preprocessing"]["feature_engineering"]["momentum_window_seconds"]
    )
    cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
    momentum_steps = momentum_window_seconds // cadence_seconds
    momentum_features = feature_engineer.compute_momentum_features(
        mid_prices, volumes, anchor_pos, momentum_steps
    )

    features: List[float] = []
    for feat_name in feature_engineer._order_book_features:
        features.append(float(order_book_features.get(feat_name, 0.0)))
    for feat_name in feature_engineer._derived_features:
        features.append(float(momentum_features.get(feat_name, 0.0)))

    return np.asarray(features, dtype="float32") if features else None


def _compute_temporal_vector(
    anchor_ts: np.datetime64,
    local_cfg: List[str],
    global_cfg: List[str],
    market_session_cfg: Dict[str, Any],
    dataset_start_day: Optional[int],
) -> Optional[np.ndarray]:
    if not local_cfg and not global_cfg:
        return None

    local_vec = (
        _compute_local_temporal_features(anchor_ts, local_cfg) if local_cfg else np.zeros((0,), dtype="float32")
    )
    if global_cfg:
        global_vec = _compute_global_temporal_features(
            anchor_ts, global_cfg, market_session_cfg, dataset_start_day
        )
    else:
        global_vec = np.zeros((0,), dtype="float32")
    if global_vec is None:
        global_vec = np.zeros((0,), dtype="float32")

    combined = np.concatenate([local_vec, global_vec]).astype("float32")
    return combined


def _compute_local_temporal_features(anchor_ts: np.datetime64, local_cfg: List[str]) -> np.ndarray:
    ts_sec = anchor_ts.astype("datetime64[s]").astype("int64")
    seconds_per_day = 24 * 60 * 60
    seconds_in_day = ts_sec % seconds_per_day
    hours = float(seconds_in_day // 3600)
    minutes = float((seconds_in_day % 3600) // 60)

    days_since_epoch = anchor_ts.astype("datetime64[D]").astype("int64")
    day_of_week = float((days_since_epoch + 3) % 7)

    two_pi = 2.0 * np.pi
    features: List[float] = []

    for name in local_cfg:
        key = str(name)
        if key == "hour_of_day":
            angle = two_pi * (hours / 24.0)
            features.extend([math.sin(angle), math.cos(angle)])
        elif key == "day_of_week":
            angle = two_pi * (day_of_week / 7.0)
            features.extend([math.sin(angle), math.cos(angle)])
        elif key == "minute_of_hour":
            angle = two_pi * (minutes / 60.0)
            features.extend([math.sin(angle), math.cos(angle)])
        else:
            raise ValueError(
                "Unsupported local temporal feature name in data.temporal_features.local: "
                f"{key!r}",
            )

    return np.asarray(features, dtype="float32")


def _compute_global_temporal_features(
    anchor_ts: np.datetime64,
    global_cfg: List[str],
    market_session_cfg: Dict[str, Any],
    dataset_start_day: Optional[int],
) -> Optional[np.ndarray]:
    features: List[float] = []

    for name in global_cfg:
        key = str(name)
        if key == "days_since_start":
            day = int(anchor_ts.astype("datetime64[D]").astype("int64"))
            if dataset_start_day is None:
                dataset_start_day = day
            features.append(float(day - dataset_start_day))
        elif key == "market_session":
            utc_offset_hours = int(market_session_cfg.get("utc_offset_hours", 0))
            sessions_cfg = market_session_cfg.get("sessions")
            if not isinstance(sessions_cfg, list) or not sessions_cfg:
                raise ValueError("data.temporal_features.market_session.sessions must be a non-empty list")

            ranges: List[Tuple[int, int]] = []
            for sess in sessions_cfg:
                start_hour = int(sess["start_hour"])
                end_hour = int(sess["end_hour"])
                ranges.append((start_hour, end_hour))

            ts_sec = anchor_ts.astype("datetime64[s]").astype("int64")
            seconds_per_day = 24 * 60 * 60
            hours_utc = int((ts_sec % seconds_per_day) // 3600)
            hours_local = (hours_utc + utc_offset_hours) % 24

            for start_hour, end_hour in ranges:
                features.append(1.0 if start_hour <= hours_local < end_hour else 0.0)
        else:
            raise ValueError(
                "Unsupported global temporal feature name in data.temporal_features.global: "
                f"{key!r}",
            )

    return np.asarray(features, dtype="float32") if features else None


def _concat_broadcast_features(x_seq: np.ndarray, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return x_seq
    t_steps, h_dim, w_dim, _ = x_seq.shape
    feat = features.astype("float32")
    feat_exp = feat[None, None, None, :]
    feat_broadcast = np.broadcast_to(feat_exp, (t_steps, h_dim, w_dim, feat.shape[0]))
    return np.concatenate([x_seq, feat_broadcast], axis=-1)


def _copy_depth(depth: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if depth is None:
        return None
    return {
        "bid_prices": depth["bid_prices"].copy(),
        "bid_quantities": depth["bid_quantities"].copy(),
        "ask_prices": depth["ask_prices"].copy(),
        "ask_quantities": depth["ask_quantities"].copy(),
    }


def _interpolate_features(a: List[float], b: List[float], alpha: float) -> List[float]:
    out: List[float] = []
    for v0, v1 in zip(a, b):
        out.append((1.0 - alpha) * float(v0) + alpha * float(v1))
    return out


def _interpolate_depth(
    a: Optional[Dict[str, np.ndarray]],
    b: Optional[Dict[str, np.ndarray]],
    alpha: float,
) -> Optional[Dict[str, np.ndarray]]:
    if a is None or b is None:
        return _copy_depth(a) if a is not None else _copy_depth(b)
    return {
        "bid_prices": (1.0 - alpha) * a["bid_prices"] + alpha * b["bid_prices"],
        "bid_quantities": (1.0 - alpha) * a["bid_quantities"] + alpha * b["bid_quantities"],
        "ask_prices": (1.0 - alpha) * a["ask_prices"] + alpha * b["ask_prices"],
        "ask_quantities": (1.0 - alpha) * a["ask_quantities"] + alpha * b["ask_quantities"],
    }


__all__ = [
    "LabelDistribution",
    "NormalizationStats",
    "SnapshotChunk",
    "SnapshotDataset",
    "build_training_generator",
    "compute_label_distribution",
    "compute_normalization_stats",
    "get_mask_channel_info",
    "iter_snapshot_batches",
    "load_label_stats_from_manifest",
    "load_normalization_stats",
    "load_snapshot_dataset",
    "prepare_snapshot_dataset",
    "save_label_stats_to_manifest",
    "save_normalization_stats",
]
