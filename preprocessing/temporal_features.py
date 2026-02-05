"""Temporal feature construction stubs.

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
    anchor_indices = metadata.get("anchor_indices")

    if anchor_indices is None:
        raise ValueError(
            "metadata.anchor_indices must be populated by the preprocessing pipeline before attaching temporal features",
        )

    anchor_arr = np.asarray(anchor_indices, dtype="int64")
    if anchor_arr.ndim != 1:
        raise ValueError("metadata.anchor_indices must be a one-dimensional list or array of integers")
    if anchor_arr.size == 0:
        raise ValueError(
            "metadata.anchor_indices must be non-empty when attaching temporal features",
        )

    n_samples = int(metadata.get("num_samples", 0))
    if n_samples <= 0:
        raise ValueError(
            "metadata.num_samples must be positive when anchor_indices are present for temporal feature construction",
        )

    if anchor_arr.size != n_samples:
        raise ValueError(
            "Length of metadata.anchor_indices must match metadata.num_samples for temporal feature construction; "
            f"got len(anchor_indices)={anchor_arr.size}, num_samples={n_samples}",
        )

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    order_books = data_object.get("order_books", {})
    target_book = order_books.get(target_asset, {})
    snapshot_timestamps = target_book.get("snapshot_timestamps")

    if snapshot_timestamps is None:
        raise ValueError(
            "order_books[target_asset].snapshot_timestamps must be populated before attaching temporal features",
        )

    ts_array = np.asarray(snapshot_timestamps, dtype="datetime64[ns]")
    if ts_array.ndim != 1:
        raise ValueError("snapshot_timestamps must be a one-dimensional sequence of timestamps")
    if ts_array.size == 0:
        raise ValueError(
            "order_books[target_asset].snapshot_timestamps must be non-empty before attaching temporal features",
        )

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
    config: Dict[str, Any],
    global_cfg: List[str],
    anchor_timestamps: np.ndarray,
) -> np.ndarray:
    """Construct global temporal feature matrix for each anchor timestamp.

    Supported feature names in global_cfg:
    - "days_since_start"  -> scalar days since first anchor timestamp
    - "market_session"     -> one-hot encoding of configured market sessions
    """

    if anchor_timestamps.size == 0:
        return np.zeros((0, 0), dtype="float32")

    feature_columns: List[np.ndarray] = []

    data_cfg = config["data"]
    tf_cfg_all = data_cfg["temporal_features"]

    for name in global_cfg:
        key = str(name)
        if key == "days_since_start":
            days = anchor_timestamps.astype("datetime64[D]").astype("int64")
            first_day = days[0]
            days_since_start = (days - first_day).astype("float64")
            feature_columns.append(days_since_start)
        elif key == "market_session":
            ms_cfg = tf_cfg_all["market_session"]

            try:
                utc_offset_hours = int(ms_cfg["utc_offset_hours"])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "data.temporal_features.market_session.utc_offset_hours must be an integer in configuration",
                ) from exc

            sessions_cfg = ms_cfg["sessions"]
            if not isinstance(sessions_cfg, list) or not sessions_cfg:
                raise ValueError(
                    "data.temporal_features.market_session.sessions must be a non-empty list in configuration",
                )

            num_sessions = len(sessions_cfg)

            # Validate session definitions and check for overlapping hour ranges.
            hour_coverage = np.zeros(24, dtype="int64")
            session_ranges: List[Tuple[int, int]] = []

            for sess in sessions_cfg:
                if not isinstance(sess, dict):
                    raise ValueError(
                        "Each entry in data.temporal_features.market_session.sessions must be a mapping",
                    )

                try:
                    start_hour = int(sess["start_hour"])
                    end_hour = int(sess["end_hour"])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        "Each session in data.temporal_features.market_session.sessions must define "
                        "integer start_hour and end_hour values",
                    ) from exc

                if start_hour < 0 or start_hour >= 24 or end_hour <= 0 or end_hour > 24 or start_hour >= end_hour:
                    raise ValueError(
                        "Session hour ranges in data.temporal_features.market_session.sessions must satisfy "
                        "0 <= start_hour < end_hour <= 24; "
                        f"got start_hour={start_hour}, end_hour={end_hour}",
                    )

                hour_coverage[start_hour:end_hour] += 1
                session_ranges.append((start_hour, end_hour))

            if np.any(hour_coverage > 1):
                raise ValueError(
                    "Session hour ranges in data.temporal_features.market_session.sessions must not overlap.",
                )

            # Compute local hour-of-day in the configured time zone using a
            # simple integer UTC offset.
            ts_sec = anchor_timestamps.astype("datetime64[s]").astype("int64")
            seconds_per_day = 24 * 60 * 60
            seconds_in_day = ts_sec % seconds_per_day
            hours_utc = (seconds_in_day // 3600).astype("int64")
            hours_local = (hours_utc + utc_offset_hours) % 24

            session_matrix = np.zeros((anchor_timestamps.shape[0], num_sessions), dtype="float64")
            for idx, (start_hour, end_hour) in enumerate(session_ranges):
                mask = (hours_local >= start_hour) & (hours_local < end_hour)
                session_matrix[mask, idx] = 1.0

            for idx in range(num_sessions):
                feature_columns.append(session_matrix[:, idx])
        else:
            raise ValueError(
                "Unsupported global temporal feature name in data.temporal_features.global: "
                f"{key!r}. Supported values currently include 'days_since_start' and 'market_session'.",
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
    tf_cfg = data_cfg["temporal_features"]
    local_cfg = tf_cfg["local"] or []
    global_cfg = tf_cfg["global"] or []

    if not isinstance(local_cfg, list) or not isinstance(global_cfg, list):
        raise ValueError("data.temporal_features.local and data.temporal_features.global must be lists in configuration")

    local_matrix = _build_local_temporal_features(local_cfg, anchor_ts)
    global_matrix = _build_global_temporal_features(config, global_cfg, anchor_ts)

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
