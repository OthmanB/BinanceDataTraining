"""Minimal GreptimeDB HTTP client.

Phase: connectivity check only.

This module uses the existing YAML configuration fields:
- data.connection.database_uri
- data.connection.table_prefix
- data.asset_pairs.target_asset
- data.asset_pairs.correlated_assets

to derive per-asset table names (e.g. ``orderbook_btcusdt``) and issue
lightweight SQL queries via the GreptimeDB HTTP API (``/v1/sql``). It logs
whether the endpoint is reachable but does **not** yet fetch real training
samples into the DataObject.
"""

from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

import requests


logger = logging.getLogger(__name__)


def _get_timeout_config(conn_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract HTTP timeout configuration from connection config.

    All timeout parameters are required in YAML; no defaults are applied here.
    """
    return {
        "request_timeout_seconds": int(conn_cfg["request_timeout_seconds"]),
        "connect_timeout_seconds": int(conn_cfg["connect_timeout_seconds"]),
        "max_retries": int(conn_cfg["max_retries"]),
        "retry_backoff_factor": float(conn_cfg["retry_backoff_factor"]),
    }


def _generate_time_chunks(
    start_date: str,
    end_date: str,
    chunk_hours: int,
) -> List[Tuple[str, str, bool]]:
    """Generate non-overlapping time chunks covering [start_date, end_date].

    Parameters
    ----------
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format (inclusive).
    chunk_hours:
        Size of each chunk in hours. Must be positive.

    Returns
    -------
    List[Tuple[str, str, bool]]:
        List of (chunk_start_datetime, chunk_end_datetime, is_last_chunk) tuples.
        Datetime strings are in 'YYYY-MM-DD HH:MM:SS' format.
        The is_last_chunk flag indicates whether to use inclusive end boundary.

    Raises
    ------
    ValueError:
        If chunk_hours <= 0 or dates are invalid.
    """
    if chunk_hours <= 0:
        raise ValueError(f"chunk_hours must be positive; got {chunk_hours}")

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format; expected YYYY-MM-DD: start_date={start_date!r}, end_date={end_date!r}"
        ) from exc

    # End of day for end_date (23:59:59)
    end_dt = end_dt.replace(hour=23, minute=59, second=59)

    if start_dt > end_dt:
        raise ValueError(
            f"start_date must be <= end_date; got start_date={start_date!r}, end_date={end_date!r}"
        )

    chunks: List[Tuple[str, str, bool]] = []
    chunk_delta = timedelta(hours=chunk_hours)

    current_start = start_dt
    while current_start <= end_dt:
        current_end = current_start + chunk_delta - timedelta(seconds=1)
        is_last = current_end >= end_dt
        if is_last:
            current_end = end_dt

        chunk_start_str = current_start.strftime("%Y-%m-%d %H:%M:%S")
        chunk_end_str = current_end.strftime("%Y-%m-%d %H:%M:%S")
        chunks.append((chunk_start_str, chunk_end_str, is_last))

        current_start = current_end + timedelta(seconds=1)

    return chunks


def check_greptime_connectivity(config: Dict[str, Any]) -> None:
    """Check connectivity to GreptimeDB using the HTTP SQL API.

    This function is intentionally conservative:
    - It constructs the base URL from ``data.connection.database_uri``.
    - It derives table names as ``data.connection.table_prefix + asset.lower()``
      for all assets defined in ``data.asset_pairs``.
    - For each derived table, it sends a simple ``SELECT 1 FROM <table> LIMIT 1``
      query to ``/v1/sql``.
    - It logs HTTP status and does not raise on network errors.

    All connection details come from YAML; there are no additional parameters
    defined in code.
    """

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]

    target_asset = asset_pairs_cfg["target_asset"]
    correlated_assets = asset_pairs_cfg["correlated_assets"]

    assets = [str(target_asset)] + [str(a) for a in correlated_assets]

    if not assets:
        logger.warning("Greptime connectivity check skipped: no asset pairs configured.")
        return

    multi_db_cfg = data_cfg.get("multi_database")
    if isinstance(multi_db_cfg, dict) and multi_db_cfg.get("enabled"):
        connections = multi_db_cfg["connections"]
        if not isinstance(connections, list) or not connections:
            raise ValueError(
                "data.multi_database.connections must be a non-empty list when multi_database.enabled is true",
            )

        for conn in connections:
            base_uri = conn["database_uri"]
            table_prefix = conn["table_prefix"]
            timeout_cfg = _get_timeout_config(conn)

            _check_greptime_connectivity_for_connection(str(base_uri), str(table_prefix), assets, timeout_cfg)
    else:
        conn_cfg = data_cfg["connection"]
        base_uri = conn_cfg["database_uri"]
        table_prefix = conn_cfg["table_prefix"]
        timeout_cfg = _get_timeout_config(conn_cfg)

        _check_greptime_connectivity_for_connection(str(base_uri), str(table_prefix), assets, timeout_cfg)


def _check_greptime_connectivity_for_connection(
    base_uri: str,
    table_prefix: str,
    assets: List[str],
    timeout_cfg: Dict[str, Any],
) -> None:
    url = base_uri.rstrip("/") + "/v1/sql"
    connect_timeout = timeout_cfg["connect_timeout_seconds"]
    request_timeout = timeout_cfg["request_timeout_seconds"]

    for asset in assets:
        table_name = f"{table_prefix}{asset.lower()}"
        sql = f"SELECT 1 FROM {table_name} LIMIT 1"

        logger.info(
            "Checking GreptimeDB connectivity at %s (table=%s)",
            url,
            table_name,
        )

        try:
            resp = requests.post(
                url,
                data={"sql": sql},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=(connect_timeout, request_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GreptimeDB connectivity check failed for table %s: %s", table_name, exc)
            continue

        if not resp.ok:
            logger.warning(
                "GreptimeDB connectivity check returned non-OK status for table %s: %s %s",
                table_name,
                resp.status_code,
                resp.text,
            )
            continue

        try:
            payload = resp.json()
        except ValueError:
            logger.info(
                "GreptimeDB connectivity check succeeded for table %s (HTTP %s) but response is not JSON.",
                table_name,
                resp.status_code,
            )
            continue

        exec_time = payload.get("execution_time_ms")
        logger.info(
            "GreptimeDB connectivity check succeeded for table %s. execution_time_ms=%s",
            table_name,
            exec_time,
        )


def fetch_order_book_rows(config: Dict[str, Any]) -> Dict[str, List[List[Any]]]:
    """Fetch raw order book rows from GreptimeDB for all configured asset pairs.

    Uses only YAML-defined parameters:
    - data.connection.database_uri
    - data.connection.table_prefix
    - data.asset_pairs (target_asset + correlated_assets)
    - data.time_range.start_date / end_date (time window in physical units)
    - data.order_book.schema.* (column names)

    Returns a mapping from asset symbol to a list of row values as returned by
    GreptimeDB (no type conversion is performed at this stage).
    """

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    time_range_cfg = data_cfg["time_range"]
    order_book_cfg = data_cfg["order_book"]
    schema_cfg = order_book_cfg["schema"]
    ingestion_cfg = data_cfg["ingestion"]

    chunk_hours = int(ingestion_cfg["chunk_hours"])
    if chunk_hours <= 0:
        raise ValueError(
            f"data.ingestion.chunk_hours must be positive; got {chunk_hours}"
        )

    target_asset = asset_pairs_cfg["target_asset"]
    correlated_assets = asset_pairs_cfg["correlated_assets"]

    assets: List[str] = [str(target_asset)] + [str(a) for a in correlated_assets]

    if not assets:
        return {}

    global_start_date = str(time_range_cfg["start_date"])
    global_end_date = str(time_range_cfg["end_date"])

    rows_by_asset: Dict[str, List[List[Any]]] = {asset: [] for asset in assets}

    multi_db_cfg = data_cfg.get("multi_database")
    if isinstance(multi_db_cfg, dict) and multi_db_cfg.get("enabled"):
        connections = multi_db_cfg["connections"]
        if not isinstance(connections, list) or not connections:
            raise ValueError(
                "data.multi_database.connections must be a non-empty list when multi_database.enabled is true",
            )

        intervals: List[Tuple[str, str, Dict[str, Any]]] = []
        for conn in connections:
            conn_time_range = conn["time_range"]
            conn_start = str(conn_time_range["start_date"])
            conn_end = str(conn_time_range["end_date"])

            if conn_start > conn_end:
                raise ValueError(
                    "Connection-level time_range.start_date must be <= time_range.end_date for data.multi_database.connections; "
                    f"got start_date={conn_start!r}, end_date={conn_end!r}",
                )

            intervals.append((conn_start, conn_end, conn))

        intervals.sort(key=lambda item: item[0])

        prev_end = None
        for conn_start, conn_end, _ in intervals:
            if prev_end is not None and conn_start < prev_end:
                raise ValueError(
                    "Overlapping time ranges are not supported for data.multi_database.connections; "
                    "ensure per-connection time_range intervals are ordered and do not have interior overlap.",
                )
            prev_end = conn_end

        for conn_start, conn_end, conn in intervals:
            constrained_start = max(global_start_date, conn_start)
            constrained_end = min(global_end_date, conn_end)
            if constrained_start > constrained_end:
                continue

            base_uri = conn["database_uri"]
            table_prefix = conn["table_prefix"]
            timeout_cfg = _get_timeout_config(conn)

            _fetch_order_book_rows_for_connection(
                str(base_uri),
                str(table_prefix),
                assets,
                constrained_start,
                constrained_end,
                schema_cfg,
                rows_by_asset,
                timeout_cfg,
                chunk_hours,
            )
    else:
        conn_cfg = data_cfg["connection"]
        base_uri = conn_cfg["database_uri"]
        table_prefix = conn_cfg["table_prefix"]
        timeout_cfg = _get_timeout_config(conn_cfg)

        _fetch_order_book_rows_for_connection(
            str(base_uri),
            str(table_prefix),
            assets,
            global_start_date,
            global_end_date,
            schema_cfg,
            rows_by_asset,
            timeout_cfg,
            chunk_hours,
        )

    return rows_by_asset


def _fetch_order_book_rows_for_connection(
    base_uri: str,
    table_prefix: str,
    assets: List[str],
    start_date: str,
    end_date: str,
    schema_cfg: Dict[str, Any],
    rows_by_asset: Dict[str, List[List[Any]]],
    timeout_cfg: Dict[str, Any],
    chunk_hours: int,
) -> None:
    """Fetch order book rows using time-based chunking to handle large time ranges.

    Internally splits the [start_date, end_date] range into chunks of `chunk_hours`
    and fetches each chunk separately to avoid memory exhaustion and HTTP timeouts.
    Uses exclusive end boundary for interior chunks to avoid duplicate rows.
    """
    ts_col = schema_cfg["timestamp_column"]
    bid_price_col = schema_cfg["bid_price_column"]
    bid_qty_col = schema_cfg["bid_quantity_column"]
    ask_price_col = schema_cfg["ask_price_column"]
    ask_qty_col = schema_cfg["ask_quantity_column"]
    batch_id_col = schema_cfg["batch_id_column"]

    url = base_uri.rstrip("/") + "/v1/sql"
    connect_timeout = timeout_cfg["connect_timeout_seconds"]
    request_timeout = timeout_cfg["request_timeout_seconds"]

    # Generate time chunks for the entire date range
    chunks = _generate_time_chunks(start_date, end_date, chunk_hours)
    total_chunks = len(chunks)

    for asset in assets:
        table_name = f"{table_prefix}{asset.lower()}"
        asset_total_rows = 0

        logger.info(
            "Fetching order book rows for asset=%s from %s (table=%s, chunks=%s)",
            asset,
            url,
            table_name,
            total_chunks,
        )

        for chunk_idx, (chunk_start, chunk_end, is_last_chunk) in enumerate(chunks):
            # Use exclusive end for interior chunks, inclusive for last chunk
            if is_last_chunk:
                end_operator = "<="
            else:
                end_operator = "<"

            sql = (
                f"SELECT {ts_col}, {bid_price_col}, {bid_qty_col}, {ask_price_col}, {ask_qty_col}, {batch_id_col} "
                f"FROM {table_name} "
                f"WHERE {ts_col} >= '{chunk_start}' AND {ts_col} {end_operator} '{chunk_end}' "
                f"AND {bid_price_col} > 0 AND {ask_price_col} > 0 "
                f"ORDER BY {ts_col} ASC"
            )

            logger.info(
                "Fetching chunk %s/%s for asset %s (%s to %s)...",
                chunk_idx + 1,
                total_chunks,
                asset,
                chunk_start,
                chunk_end,
            )

            try:
                resp = requests.post(
                    url,
                    data={"sql": sql},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=(connect_timeout, request_timeout),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GreptimeDB data fetch failed for table %s (asset=%s, chunk=%s/%s): %s",
                    table_name,
                    asset,
                    chunk_idx + 1,
                    total_chunks,
                    exc,
                )
                continue

            if not resp.ok:
                logger.warning(
                    "GreptimeDB data fetch returned non-OK status for table %s (asset=%s, chunk=%s/%s): %s %s",
                    table_name,
                    asset,
                    chunk_idx + 1,
                    total_chunks,
                    resp.status_code,
                    resp.text,
                )
                continue

            try:
                payload = resp.json()
            except ValueError:
                logger.warning(
                    "GreptimeDB data fetch succeeded for table %s (asset=%s, chunk=%s/%s, HTTP %s) but response is not JSON.",
                    table_name,
                    asset,
                    chunk_idx + 1,
                    total_chunks,
                    resp.status_code,
                )
                continue

            output = payload.get("output")
            if not output:
                logger.debug(
                    "GreptimeDB data fetch returned empty output for table %s (asset=%s, chunk=%s/%s).",
                    table_name,
                    asset,
                    chunk_idx + 1,
                    total_chunks,
                )
                continue

            records = output[0].get("records") if isinstance(output, list) and output else None
            if not records:
                logger.debug(
                    "GreptimeDB data fetch returned no records for table %s (asset=%s, chunk=%s/%s).",
                    table_name,
                    asset,
                    chunk_idx + 1,
                    total_chunks,
                )
                continue

            rows = records.get("rows") or []
            if rows:
                rows_by_asset[asset].extend(rows)
                asset_total_rows += len(rows)

                logger.debug(
                    "Fetched %s rows for chunk %s/%s (asset=%s). execution_time_ms=%s",
                    len(rows),
                    chunk_idx + 1,
                    total_chunks,
                    asset,
                    payload.get("execution_time_ms"),
                )

        logger.info(
            "Completed fetching for asset=%s. total_rows=%s, chunks=%s",
            asset,
            asset_total_rows,
            total_chunks,
        )


__all__ = ["check_greptime_connectivity", "fetch_order_book_rows"]
