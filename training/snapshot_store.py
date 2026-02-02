"""Snapshot store utilities for derived datasets.

This module manages snapshot directories, manifests, and configuration hashes
for derived, preprocessed datasets used in streaming training.
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
class SnapshotContext:
    """Resolved snapshot directory and manifest metadata."""

    snapshot_dir: str
    manifest_path: str
    config_hash: str
    config_snapshot: Dict[str, Any]
    snapshot_name: str
    root_name: str


def _snapshot_config_subset(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    preprocessing_cfg = config.get("preprocessing", {})
    targets_cfg = config.get("targets", {})

    subset = {
        "data": {
            "asset_pairs": data_cfg.get("asset_pairs"),
            "time_range": data_cfg.get("time_range"),
            "order_book": {
                "depth_levels": data_cfg.get("order_book", {}).get("depth_levels"),
                "representation": data_cfg.get("order_book", {}).get("representation"),
                "hybrid": data_cfg.get("order_book", {}).get("hybrid"),
            },
            "temporal_features": data_cfg.get("temporal_features"),
            "alignment": data_cfg.get("asset_pairs", {}).get("alignment"),
        },
        "targets": {
            "prediction_horizon_seconds": targets_cfg.get("prediction_horizon_seconds"),
            "visible_window_seconds": targets_cfg.get("visible_window_seconds"),
            "price_classes": targets_cfg.get("price_classes"),
            "labeling": targets_cfg.get("labeling"),
        },
        "preprocessing": {
            "normalization": preprocessing_cfg.get("normalization"),
            "feature_engineering": preprocessing_cfg.get("feature_engineering"),
        },
        "model": {
            "architecture": model_cfg.get("architecture"),
            "input_representation": model_cfg.get("input_representation"),
            "cnn": {
                "kernel_sizes": model_cfg.get("cnn", {}).get("kernel_sizes"),
                "pool_sizes": model_cfg.get("cnn", {}).get("pool_sizes"),
            },
            "output": model_cfg.get("output"),
        },
    }

    return subset


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash for the snapshot-relevant configuration."""

    subset = _snapshot_config_subset(config)
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


def _build_auto_snapshot_name(config: Dict[str, Any], root_name: str, config_hash: str) -> str:
    data_cfg = config.get("data", {})
    asset_pairs_cfg = data_cfg.get("asset_pairs", {})
    target_asset = str(asset_pairs_cfg.get("target_asset") or "asset")

    time_range_cfg = data_cfg.get("time_range", {})
    start_date = str(time_range_cfg.get("start_date") or "start")
    end_date = str(time_range_cfg.get("end_date") or "end")

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
        logger.warning("Failed to load snapshot manifest %s: %s", path, exc)
        return None


def _write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def resolve_snapshot_context(config: Dict[str, Any]) -> SnapshotContext:
    """Resolve the snapshot directory and manifest for the current config."""

    snapshot_cfg = config.get("snapshot")
    if not isinstance(snapshot_cfg, dict):
        raise ConfigError("snapshot section must be defined in configuration")

    enabled = bool(snapshot_cfg.get("enabled"))
    if not enabled:
        raise ConfigError("snapshot.enabled must be true to resolve snapshot context")

    root_dir = str(snapshot_cfg["directory"])
    root_name = str(snapshot_cfg["root_name"])
    snapshot_name = str(snapshot_cfg["name"])
    on_mismatch = str(snapshot_cfg["on_config_mismatch"])

    if on_mismatch not in {"create_new", "error"}:
        raise ConfigError(
            "snapshot.on_config_mismatch must be 'create_new' or 'error'",
        )

    if not root_name.strip():
        raise ConfigError("snapshot.root_name must be a non-empty string")

    config_hash = compute_config_hash(config)
    config_snapshot = _snapshot_config_subset(config)

    if snapshot_name == "auto":
        resolved_name = _build_auto_snapshot_name(config, root_name, config_hash)
    else:
        resolved_name = snapshot_name

    snapshot_dir = os.path.join(root_dir, resolved_name)
    manifest_path = os.path.join(snapshot_dir, "manifest.json")

    os.makedirs(root_dir, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    if manifest is not None:
        manifest_hash = str(manifest.get("config_hash") or "")
        if manifest_hash and manifest_hash != config_hash:
            if on_mismatch == "error":
                raise ConfigError(
                    "Snapshot configuration hash mismatch for existing snapshot. "
                    "Set snapshot.on_config_mismatch='create_new' or choose a new snapshot name.",
                )

            suffix = config_hash[:8]
            resolved_name = f"{resolved_name}_{suffix}"
            snapshot_dir = os.path.join(root_dir, resolved_name)
            manifest_path = os.path.join(snapshot_dir, "manifest.json")

    os.makedirs(snapshot_dir, exist_ok=True)

    return SnapshotContext(
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        config_hash=config_hash,
        config_snapshot=config_snapshot,
        snapshot_name=resolved_name,
        root_name=root_name,
    )


def initialize_manifest(context: SnapshotContext, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new manifest structure for a snapshot."""

    data_cfg = config.get("data", {})
    time_range_cfg = data_cfg.get("time_range", {})

    manifest = {
        "version": 1,
        "snapshot_name": context.snapshot_name,
        "root_name": context.root_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": context.config_hash,
        "config_snapshot": context.config_snapshot,
        "time_range": {
            "start_date": time_range_cfg.get("start_date"),
            "end_date": time_range_cfg.get("end_date"),
        },
        "assets": [str(data_cfg.get("asset_pairs", {}).get("target_asset") or "")],
        "chunks": [],
        "normalization_stats": {},
    }

    _write_manifest(context.manifest_path, manifest)
    return manifest


def load_or_create_manifest(context: SnapshotContext, config: Dict[str, Any]) -> Dict[str, Any]:
    manifest = _load_manifest(context.manifest_path)
    if manifest is None:
        return initialize_manifest(context, config)
    return manifest


def save_manifest(context: SnapshotContext, manifest: Dict[str, Any]) -> None:
    _write_manifest(context.manifest_path, manifest)


def maybe_evict_snapshots(context: SnapshotContext, max_snapshots: int) -> None:
    """Evict old snapshots beyond max_snapshots if configured."""

    if max_snapshots <= 0:
        return

    root_dir = os.path.dirname(context.snapshot_dir)
    prefix = f"{_sanitize_component(context.root_name)}_"

    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(root_dir):
        if not name.startswith(prefix):
            continue
        snapshot_path = os.path.join(root_dir, name)
        manifest_path = os.path.join(snapshot_path, "manifest.json")
        if not os.path.isdir(snapshot_path) or not os.path.exists(manifest_path):
            continue
        manifest = _load_manifest(manifest_path)
        created_at = None
        if manifest is not None:
            created_at = manifest.get("created_at")
        if not created_at:
            created_at = datetime.fromtimestamp(os.path.getmtime(manifest_path), timezone.utc).isoformat()
        candidates.append((snapshot_path, str(created_at)))

    candidates.sort(key=lambda item: item[1])

    while len(candidates) > max_snapshots:
        snapshot_path, _ = candidates.pop(0)
        if os.path.abspath(snapshot_path) == os.path.abspath(context.snapshot_dir):
            continue
        try:
            shutil.rmtree(snapshot_path)
            logger.info("Evicted old snapshot directory: %s", snapshot_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to evict snapshot directory %s: %s", snapshot_path, exc)


__all__ = [
    "SnapshotContext",
    "compute_config_hash",
    "initialize_manifest",
    "load_or_create_manifest",
    "maybe_evict_snapshots",
    "resolve_snapshot_context",
    "save_manifest",
]
