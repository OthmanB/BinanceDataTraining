"""Series-only dataset builder.

This builds a disk-cached series dataset containing only:
- timestamps (int64 seconds)
- mid_prices (float64)
- volumes (float64) for target asset (if configured; may be zeros)

It reuses the same gap handling and multi-asset alignment logic as the snapshot
dataset builder, but it does not require class boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import os

import numpy as np

from data.greptime_client import OrderBookChunk, _generate_time_chunks, stream_order_book_chunks_by_time
from preprocessing.depth_aggregator import get_hybrid_output_shape
from training.series_store import (
    SeriesContext,
    load_or_create_series_manifest,
    maybe_evict_series_caches,
    resolve_series_context,
    save_series_manifest,
)
from training.snapshot_dataset import (
    GapHandler,
    _build_snapshots_from_rows,
    _chunk_filename,
    _format_bytes,
    _populate_hybrid_snapshots,
    _safe_file_size,
)
from utils.config_loader import ConfigError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesChunk:
    start: str
    end: str
    file_path: str
    num_snapshots: int


@dataclass(frozen=True)
class SeriesDataset:
    series_dir: str
    manifest: Dict[str, Any]
    chunks: List[SeriesChunk]
    total_snapshots: int
    config_hash: str


def prepare_series_dataset(config: Dict[str, Any]) -> SeriesDataset:
    """Prepare (build or load) the series-only dataset."""

    context = resolve_series_context(config)
    max_caches = int(config["snapshot"].get("max_snapshots", 0))
    maybe_evict_series_caches(context, max_caches)

    manifest = load_or_create_series_manifest(context, config)
    complete = bool(manifest.get("complete"))
    chunk_entries = manifest.get("chunks", []) or []
    if complete and not chunk_entries:
        logger.warning("Series manifest marked complete but contains no chunks; rebuilding series cache.")
        complete = False

    if complete:
        for entry in chunk_entries:
            file_rel = entry.get("file")
            if not file_rel:
                complete = False
                break
            file_path = os.path.join(context.series_dir, file_rel)
            if not os.path.exists(file_path):
                complete = False
                break

    if not complete:
        manifest = _build_series_chunks(config, context, manifest)

    return _materialize_series_dataset(context, manifest)


def _materialize_series_dataset(context: SeriesContext, manifest: Dict[str, Any]) -> SeriesDataset:
    if manifest.get("config_hash") != context.config_hash:
        raise ConfigError("Series manifest config_hash does not match current configuration")

    chunks: List[SeriesChunk] = []
    total = 0
    for entry in manifest.get("chunks", []) or []:
        start = str(entry.get("start") or "")
        end = str(entry.get("end") or "")
        file_rel = entry.get("file")
        if not start or not end or not isinstance(file_rel, str) or not file_rel:
            continue
        file_path = os.path.join(context.series_dir, file_rel)
        if not os.path.exists(file_path):
            continue
        n = int(entry.get("num_snapshots") or 0)
        chunks.append(SeriesChunk(start=start, end=end, file_path=file_path, num_snapshots=n))
        total += n
    chunks.sort(key=lambda c: c.start)
    return SeriesDataset(
        series_dir=context.series_dir,
        manifest=manifest,
        chunks=chunks,
        total_snapshots=int(total),
        config_hash=context.config_hash,
    )


def _existing_series_entries(context: SeriesContext, manifest: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in manifest.get("chunks", []) or []:
        start = entry.get("start")
        end = entry.get("end")
        file_rel = entry.get("file")
        if not start or not end or not file_rel:
            continue
        file_path = os.path.join(context.series_dir, file_rel)
        if not os.path.exists(file_path):
            continue
        existing[(start, end)] = entry
    return existing


def _upsert_series_entry(manifest: Dict[str, Any], entry: Dict[str, Any]) -> None:
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


def _build_series_chunks(config: Dict[str, Any], context: SeriesContext, manifest: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    start_date = str(time_range_cfg["start_date"])
    end_date = str(time_range_cfg["end_date"])
    cadence_seconds = int(time_range_cfg["cadence_seconds"])

    chunk_hours = int(data_cfg["ingestion"]["chunk_hours"])
    if chunk_hours <= 0:
        raise ValueError("data.ingestion.chunk_hours must be positive")

    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])
    correlated_assets = [str(a) for a in asset_pairs_cfg["correlated_assets"]]
    assets = [target_asset] + correlated_assets
    if not assets:
        raise ValueError("data.asset_pairs must define at least one asset")

    output_chunks = _generate_time_chunks(start_date, end_date, chunk_hours)
    output_boundaries = [(c[0], c[1]) for c in output_chunks]

    os.makedirs(os.path.join(context.series_dir, "chunks"), exist_ok=True)

    manifest["complete"] = False
    manifest["target_asset"] = target_asset
    manifest["cadence_seconds"] = cadence_seconds
    save_series_manifest(context, manifest)

    existing = _existing_series_entries(context, manifest)
    for start_str, end_str in output_boundaries:
        key = (start_str, end_str)
        if key in existing:
            existing[key]["cached"] = True

    gap_handlers = {asset: _create_gap_handler(config) for asset in assets}
    validation_cfg = data_cfg["validation"]
    fail_on_invalid = bool(validation_cfg["fail_on_invalid"])

    order_book_cfg = data_cfg["order_book"]
    representation = str(order_book_cfg["representation"])
    if representation == "full":
        representation = "hybrid"
    hybrid_levels = get_hybrid_output_shape(config) if representation == "hybrid" else None
    alignment_cfg = asset_pairs_cfg["alignment"]

    current_chunk_key: Optional[Tuple[str, str]] = None
    chunk_rows: Dict[str, List[List[Any]]] = {}

    def process_chunk(chunk_key: Tuple[str, str], chunk_rows_by_asset: Dict[str, List[List[Any]]]) -> None:
        missing_assets = [asset for asset in assets if asset not in chunk_rows_by_asset]
        if missing_assets:
            message = f"Missing chunk data for assets={missing_assets} in chunk {chunk_key}"
            if fail_on_invalid:
                raise ValueError(message)
            logger.warning(message)
            return

        asset_records: Dict[str, List[Any]] = {}
        for asset in assets:
            rows = chunk_rows_by_asset.get(asset, [])
            chunk = OrderBookChunk(asset=asset, chunk_start=chunk_key[0], chunk_end=chunk_key[1], rows=rows)
            compute_volume_proxy = asset == target_asset
            records = _build_snapshots_from_rows(chunk, config, compute_volume_proxy=compute_volume_proxy)
            filled = list(gap_handlers[asset].iter_gap_handled(records))
            if representation == "hybrid":
                _populate_hybrid_snapshots(
                    filled,
                    config,
                    fail_on_invalid=fail_on_invalid,
                    asset_name=asset,
                )
            asset_records[asset] = filled

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
                raise ValueError(f"Aligned snapshot missing target asset '{target_asset}' for chunk {chunk_key}")
            series_timestamps.append(int(snapshot.timestamp.astype("datetime64[s]").astype("int64")))
            series_mid_prices.append(float(target_snapshot.mid_price))
            series_volumes.append(float(target_snapshot.volume_proxy))

        series_filename = _chunk_filename(chunk_key[0], chunk_key[1])
        series_rel = os.path.join("chunks", series_filename)
        series_path = os.path.join(context.series_dir, series_rel)

        ts_arr = np.asarray(series_timestamps, dtype="int64")
        mid_arr = np.asarray(series_mid_prices, dtype="float64")
        vol_arr = np.asarray(series_volumes, dtype="float64")

        # Always write if missing; never overwrite existing.
        if not os.path.exists(series_path):
            np.savez_compressed(series_path, timestamps=ts_arr, mid_prices=mid_arr, volumes=vol_arr)
            logger.info(
                "Series chunk written: %s -> %s snapshots=%s size=%s",
                chunk_key[0],
                chunk_key[1],
                int(ts_arr.shape[0]),
                _format_bytes(_safe_file_size(series_path)),
            )

        entry = {
            "start": chunk_key[0],
            "end": chunk_key[1],
            "file": series_rel,
            "num_snapshots": int(ts_arr.shape[0]),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        _upsert_series_entry(manifest, entry)
        save_series_manifest(context, manifest)

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

    manifest["complete"] = True
    save_series_manifest(context, manifest)
    return manifest


def _create_gap_handler(config: Dict[str, Any]) -> GapHandler:
    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    validation_cfg = data_cfg["validation"]
    alignment_cfg = data_cfg["asset_pairs"]["alignment"]
    labeling_cfg = config["targets"]["labeling"]

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
