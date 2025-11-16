from typing import Any, Sequence

import numpy as np


def normalize_timestamp_array(raw_timestamps: Sequence[Any]) -> np.ndarray:
    """Normalize raw Greptime-style timestamps to a datetime64[ns] array.

    This centralizes unit inference (seconds / ms / µs / ns) so that all
    consumers (diagnostics, training, evaluation) share exactly the same
    timestamp normalization logic.
    """

    if not raw_timestamps:
        return np.asarray([], dtype="datetime64[ns]")

    first_ts = raw_timestamps[0]

    if isinstance(first_ts, (np.datetime64, str)):
        return np.asarray(raw_timestamps, dtype="datetime64[ns]")

    if isinstance(first_ts, (int, np.integer, float)):
        ts_int = np.asarray(raw_timestamps, dtype="int64")
        abs_val = int(abs(ts_int[0]))

        # Magnitude-based heuristic for epoch-based integer timestamps.
        if abs_val >= 1_000_000_000_000_000_000:
            return ts_int.astype("datetime64[ns]")
        if abs_val >= 1_000_000_000_000_000:
            return ts_int.astype("datetime64[us]").astype("datetime64[ns]")
        if abs_val >= 1_000_000_000_000:
            return ts_int.astype("datetime64[ms]").astype("datetime64[ns]")
        return ts_int.astype("datetime64[s]").astype("datetime64[ns]")

    # Fallback for other types (e.g., Python datetime objects).
    return np.asarray(raw_timestamps, dtype="datetime64[ns]")
