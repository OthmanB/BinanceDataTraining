"""Temporal feature construction stubs.

Phase 2 defines a placeholder for attaching temporal features to DataObject
instances. Detailed implementations will follow the temporal encoding
configuration in later phases.
"""

from typing import Any, Dict, List, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


def _compute_anchor_timestamps(
    data_object: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (anchor_indices, anchor_timestamps) for the target asset.

    anchor_indices has shape (N,) and contains integer snapshot indices. The
    corresponding anchor_timestamps array has dtype datetime64[ns] and shape
    (N,), where each element is the timestamp of the snapshot referenced by
    the anchor index.
    """

    metadata = data_object.get("metadata", {})
    anchor_indices: List[int] = metadata.get("anchor_indices") or []

    if not anchor_indices:
        raise ValueError(
            "metadata.anchor_indices must be populated by the preprocessing pipeline before attaching temporal features",
        )

    n_samples = int(metadata.get("num_samples", 0))
    if n_samples <= 0:
        raise ValueError(
            "metadata.num_samples must be positive when anchor_indices are present for temporal feature construction",
        )

    if len(anchor_indices) != n_samples:
        raise ValueError(
            "Length of metadata.anchor_indices must match metadata.num_samples for temporal feature construction; "
            f"got len(anchor_indices)={len(anchor_indices)}, num_samples={n_samples}",
        )

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    order_books = data_object.get("order_books", {})
    target_book = order_books.get(target_asset, {})
    snapshot_timestamps = target_book.get("snapshot_timestamps") or []

    if not snapshot_timestamps:
        raise ValueError(
            "order_books[target_asset].snapshot_timestamps must be populated before attaching temporal features",
        )

    ts_array = np.asarray(snapshot_timestamps, dtype="datetime64[ns]")
    if ts_array.ndim != 1:
        raise ValueError("snapshot_timestamps must be a one-dimensional sequence of timestamps")

    anchor_arr = np.asarray(anchor_indices, dtype="int64")
    if anchor_arr.ndim != 1:
        raise ValueError("metadata.anchor_indices must be a one-dimensional list of integers")

    if anchor_arr.min() < 0 or anchor_arr.max() >= ts_array.shape[0]:
        raise ValueError(
            "metadata.anchor_indices must reference valid snapshot indices; "
            f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_snapshots={ts_array.shape[0]}",
        )

    anchor_ts = ts_array[anchor_arr]
    return anchor_arr, anchor_ts


def _build_local_temporal_features(
    local_cfg: List[str],
    anchor_timestamps: np.ndarray,
) -> np.ndarray:
    """Construct local temporal feature matrix for each anchor timestamp.

    Supported feature names in local_cfg:
    - "hour_of_day"      -> sin/cos pair
    - "day_of_week"      -> sin/cos pair (Monday=0..Sunday=6)
    - "minute_of_hour"   -> sin/cos pair
    """

    if anchor_timestamps.size == 0:
        return np.zeros((0, 0), dtype="float32")

    ts_sec = anchor_timestamps.astype("datetime64[s]").astype("int64")
    seconds_per_day = 24 * 60 * 60
    seconds_in_day = ts_sec % seconds_per_day

    hours = (seconds_in_day // 3600).astype("float64")
    minutes = ((seconds_in_day % 3600) // 60).astype("float64")

    days_since_epoch = anchor_timestamps.astype("datetime64[D]").astype("int64")
    # 1970-01-01 is a Thursday (3), so (days + 3) % 7 gives Monday=0..Sunday=6.
    day_of_week = ((days_since_epoch + 3) % 7).astype("float64")

    two_pi = 2.0 * np.pi

    feature_columns: List[np.ndarray] = []

    for name in local_cfg:
        key = str(name)
        if key == "hour_of_day":
            angle = two_pi * (hours / 24.0)
            feature_columns.append(np.sin(angle))
            feature_columns.append(np.cos(angle))
        elif key == "day_of_week":
            angle = two_pi * (day_of_week / 7.0)
            feature_columns.append(np.sin(angle))
            feature_columns.append(np.cos(angle))
        elif key == "minute_of_hour":
            angle = two_pi * (minutes / 60.0)
            feature_columns.append(np.sin(angle))
            feature_columns.append(np.cos(angle))
        else:
            raise ValueError(
                "Unsupported local temporal feature name in data.temporal_features.local: "
                f"{key!r}. Supported values are 'hour_of_day', 'day_of_week', 'minute_of_hour'",
            )

    if not feature_columns:
        return np.zeros((anchor_timestamps.shape[0], 0), dtype="float32")

    local_matrix = np.stack(feature_columns, axis=1).astype("float32")
    return local_matrix


def _build_global_temporal_features(
    global_cfg: List[str],
    anchor_timestamps: np.ndarray,
) -> np.ndarray:
    """Construct global temporal feature matrix for each anchor timestamp.

    Supported feature names in global_cfg:
    - "days_since_start"  -> scalar days since first anchor timestamp

    Unsupported names (e.g., "market_session") will trigger a warning and be
    ignored until the corresponding configuration and design are finalized.
    """

    if anchor_timestamps.size == 0:
        return np.zeros((0, 0), dtype="float32")

    feature_columns: List[np.ndarray] = []

    for name in global_cfg:
        key = str(name)
        if key == "days_since_start":
            days = anchor_timestamps.astype("datetime64[D]").astype("int64")
            first_day = days[0]
            days_since_start = (days - first_day).astype("float64")
            feature_columns.append(days_since_start)
        elif key == "market_session":
            logger.warning(
                "Global temporal feature 'market_session' is listed in configuration "
                "but is not yet implemented; it will be ignored in this phase.",
            )
        else:
            raise ValueError(
                "Unsupported global temporal feature name in data.temporal_features.global: "
                f"{key!r}. Supported values currently include 'days_since_start' and 'market_session' (ignored).",
            )

    if not feature_columns:
        return np.zeros((anchor_timestamps.shape[0], 0), dtype="float32")

    global_matrix = np.stack(feature_columns, axis=1).astype("float32")
    return global_matrix


def attach_temporal_features(config: Dict[str, Any], data_object: Dict[str, Any]) -> Dict[str, Any]:
    """Attach temporal features to a DataObject.

    This implementation constructs temporal feature matrices for each effective
    sample based on the configuration-driven lists in data.temporal_features:

    - data.temporal_features.local: local cyclical encodings (e.g., hour_of_day,
      day_of_week, minute_of_hour), producing a (N, D_local) float32 array.
    - data.temporal_features.global: coarse trend-style features (e.g.,
      days_since_start), producing a (N, D_global) float32 array.

    The per-sample timestamps are derived from metadata.anchor_indices and the
    target asset's snapshot_timestamps built during preprocessing.
    """

    metadata = data_object.get("metadata", {})
    n_samples = int(metadata.get("num_samples", 0))
    if n_samples <= 0:
        logger.info(
            "Temporal feature attachment skipped: metadata.num_samples=%s.",
            n_samples,
        )
        return data_object

    try:
        anchor_indices, anchor_ts = _compute_anchor_timestamps(data_object, config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Temporal feature attachment failed to compute anchor timestamps: %s", exc)
        return data_object

    data_cfg = config["data"]
    tf_cfg = data_cfg.get("temporal_features", {})
    local_cfg = tf_cfg.get("local", []) or []
    global_cfg = tf_cfg.get("global", []) or []

    if not isinstance(local_cfg, list) or not isinstance(global_cfg, list):
        raise ValueError("data.temporal_features.local and data.temporal_features.global must be lists in configuration")

    local_matrix = _build_local_temporal_features(local_cfg, anchor_ts)
    global_matrix = _build_global_temporal_features(global_cfg, anchor_ts)

    if local_matrix.shape[0] != n_samples or global_matrix.shape[0] != n_samples:
        raise ValueError(
            "Temporal feature matrices must have one row per sample; "
            f"got local.shape={local_matrix.shape}, global.shape={global_matrix.shape}, num_samples={n_samples}",
        )

    temporal_features = data_object.get("temporal_features", {})
    if not isinstance(temporal_features, dict):
        temporal_features = {}

    temporal_features["local"] = local_matrix
    temporal_features["global"] = global_matrix
    data_object["temporal_features"] = temporal_features

    logger.info(
        "Temporal features attached: n_samples=%s, local_dim=%s, global_dim=%s",
        n_samples,
        local_matrix.shape[1],
        global_matrix.shape[1],
    )

    return data_object


__all__ = ["attach_temporal_features"]
