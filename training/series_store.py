"""Series-only store utilities.

This module manages disk-cached, series-only datasets (timestamps/mid_prices/volumes)
used for tasks like fitting automatic price-class boundaries.

Important: the series cache hash intentionally excludes targets.price_classes.boundaries
so that auto boundary fitting can run before snapshot hashing/building.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import os
import shutil

from utils.config_loader import ConfigError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesContext:
    """Resolved directory and manifest metadata for a series-only cache."""

    series_dir: str
    manifest_path: str
    config_hash: str
    config_snapshot: Dict[str, Any]
    series_name: str
    root_name: str


def _series_config_subset(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    preprocessing_cfg = config["preprocessing"]
    targets_cfg = config["targets"]

    asset_pairs_cfg = data_cfg["asset_pairs"]
    time_range_cfg = data_cfg["time_range"]
    order_book_cfg = data_cfg["order_book"]

    # Keep this subset limited to keys that change the derived series.
    # Do NOT include targets.price_classes or model.output.*.
    subset = {
        "data": {
            "asset_pairs": {
                "target_asset": asset_pairs_cfg["target_asset"],
                "correlated_assets": asset_pairs_cfg["correlated_assets"],
                "alignment": asset_pairs_cfg["alignment"],
            },
            "time_range": time_range_cfg,
            "ingestion": {
                "chunk_hours": data_cfg["ingestion"]["chunk_hours"],
            },
            "order_book": {
                "depth_levels": order_book_cfg["depth_levels"],
                "representation": order_book_cfg["representation"],
                "hybrid": order_book_cfg["hybrid"],
            },
        },
        "targets": {
            "labeling": targets_cfg["labeling"],
        },
        "preprocessing": {
            "feature_engineering": preprocessing_cfg["feature_engineering"],
        },
    }
    return subset


def compute_series_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash for the series-cache-relevant configuration."""

    subset = _series_config_subset(config)
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_component(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("-")
    return "".join(safe_chars)


def _build_auto_series_name(config: Dict[str, Any], root_name: str, config_hash: str) -> str:
    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    time_range_cfg = data_cfg["time_range"]
    start_date = str(time_range_cfg["start_date"])
    end_date = str(time_range_cfg["end_date"])

    parts = [
        _sanitize_component(root_name),
        config_hash[:8],
        _sanitize_component(target_asset),
        _sanitize_component(start_date),
        _sanitize_component(end_date),
    ]
    return "_".join(parts)


def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load series manifest %s: %s", path, exc)
        return None


def _write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def resolve_series_context(config: Dict[str, Any]) -> SeriesContext:
    """Resolve series cache directory and manifest for the current config."""

    snapshot_cfg = config.get("snapshot")
    if not isinstance(snapshot_cfg, dict):
        raise ConfigError("snapshot must be a dict to resolve series context")

    root_dir_base = str(snapshot_cfg.get("directory") or "")
    root_name_base = str(snapshot_cfg.get("root_name") or "")
    name_base = str(snapshot_cfg.get("name") or "auto")
    on_mismatch = str(snapshot_cfg.get("on_config_mismatch") or "create_new")

    if not root_dir_base.strip():
        raise ConfigError("snapshot.directory must be a non-empty string")
    if not root_name_base.strip():
        raise ConfigError("snapshot.root_name must be a non-empty string")
    if on_mismatch not in {"create_new", "error"}:
        raise ConfigError("snapshot.on_config_mismatch must be 'create_new' or 'error'")

    # Keep series caches separate from full snapshot datasets.
    root_dir = os.path.join(root_dir_base, "series_cache")
    root_name = f"{root_name_base}-series"

    config_hash = compute_series_config_hash(config)
    config_snapshot = _series_config_subset(config)

    if name_base == "auto":
        resolved_name = _build_auto_series_name(config, root_name, config_hash)
    else:
        resolved_name = f"{str(name_base)}_series"

    series_dir = os.path.join(root_dir, resolved_name)
    manifest_path = os.path.join(series_dir, "manifest.json")
    os.makedirs(root_dir, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    if manifest is not None:
        manifest_hash = str(manifest.get("config_hash") or "")
        if manifest_hash and manifest_hash != config_hash:
            if on_mismatch == "error":
                raise ConfigError(
                    "Series cache configuration hash mismatch for existing series cache. "
                    "Set snapshot.on_config_mismatch='create_new' or choose a new snapshot name.",
                )
            suffix = config_hash[:8]
            resolved_name = f"{resolved_name}_{suffix}"
            series_dir = os.path.join(root_dir, resolved_name)
            manifest_path = os.path.join(series_dir, "manifest.json")

    os.makedirs(series_dir, exist_ok=True)
    return SeriesContext(
        series_dir=series_dir,
        manifest_path=manifest_path,
        config_hash=config_hash,
        config_snapshot=config_snapshot,
        series_name=resolved_name,
        root_name=root_name,
    )


def initialize_series_manifest(context: SeriesContext, config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    time_range_cfg = data_cfg["time_range"]
    cadence_seconds = int(time_range_cfg["cadence_seconds"])
    target_asset = str(data_cfg["asset_pairs"]["target_asset"])

    manifest = {
        "version": 1,
        "series_name": context.series_name,
        "root_name": context.root_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": context.config_hash,
        "config_snapshot": context.config_snapshot,
        "time_range": {
            "start_date": time_range_cfg["start_date"],
            "end_date": time_range_cfg["end_date"],
        },
        "assets": [target_asset],
        "cadence_seconds": cadence_seconds,
        "chunks": [],
        "complete": False,
    }
    _write_manifest(context.manifest_path, manifest)
    return manifest


def load_or_create_series_manifest(context: SeriesContext, config: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _load_manifest(context.manifest_path)
    if manifest is None:
        return initialize_series_manifest(context, config)
    return manifest


def save_series_manifest(context: SeriesContext, manifest: Dict[str, Any]) -> None:
    _write_manifest(context.manifest_path, manifest)


def maybe_evict_series_caches(context: SeriesContext, max_caches: int) -> None:
    if max_caches <= 0:
        return

    root_dir = os.path.dirname(context.series_dir)
    prefix = f"{_sanitize_component(context.root_name)}_"

    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(root_dir):
        if not name.startswith(prefix):
            continue
        cache_path = os.path.join(root_dir, name)
        manifest_path = os.path.join(cache_path, "manifest.json")
        if not os.path.isdir(cache_path) or not os.path.exists(manifest_path):
            continue
        manifest = _load_manifest(manifest_path)
        created_at = None
        if manifest is not None:
            created_at = manifest.get("created_at")
        if not created_at:
            created_at = datetime.fromtimestamp(os.path.getmtime(manifest_path), timezone.utc).isoformat()
        candidates.append((cache_path, str(created_at)))

    candidates.sort(key=lambda item: item[1])

    while len(candidates) > max_caches:
        cache_path, _ = candidates.pop(0)
        if os.path.abspath(cache_path) == os.path.abspath(context.series_dir):
            continue
        try:
            shutil.rmtree(cache_path)
            logger.info("Evicted old series cache: %s", cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to evict old series cache %s: %s", cache_path, exc)
