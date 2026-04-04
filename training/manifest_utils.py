"""Shared helpers for manifest persistence used by store modules."""

from __future__ import annotations

from typing import Any, Dict, Optional
import json
import logging
import os


def sanitize_component(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("-")
    return "".join(safe_chars)


def load_manifest(path: str, logger: logging.Logger, store_name: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s manifest %s: %s", store_name, path, exc)
        return None


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


__all__ = ["load_manifest", "sanitize_component", "write_manifest"]
