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

from typing import Any, Dict, List
import logging

import requests


logger = logging.getLogger(__name__)


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

    # Strict access to required configuration fields (no code-level defaults)
    data_cfg = config["data"]
    conn_cfg = data_cfg["connection"]
    asset_pairs_cfg = data_cfg["asset_pairs"]

    base_uri = conn_cfg["database_uri"]
    table_prefix = conn_cfg["table_prefix"]

    target_asset = asset_pairs_cfg["target_asset"]
    correlated_assets = asset_pairs_cfg["correlated_assets"]

    assets = [str(target_asset)] + [str(a) for a in correlated_assets]

    if not assets:
        logger.warning("Greptime connectivity check skipped: no asset pairs configured.")
        return

    url = base_uri.rstrip("/") + "/v1/sql"

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
    conn_cfg = data_cfg["connection"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    time_range_cfg = data_cfg["time_range"]
    order_book_cfg = data_cfg["order_book"]
    schema_cfg = order_book_cfg["schema"]

    base_uri = conn_cfg["database_uri"]
    table_prefix = conn_cfg["table_prefix"]

    target_asset = asset_pairs_cfg["target_asset"]
    correlated_assets = asset_pairs_cfg["correlated_assets"]

    assets: List[str] = [str(target_asset)] + [str(a) for a in correlated_assets]

    start_date = time_range_cfg["start_date"]
    end_date = time_range_cfg["end_date"]

    ts_col = schema_cfg["timestamp_column"]
    bid_price_col = schema_cfg["bid_price_column"]
    bid_qty_col = schema_cfg["bid_quantity_column"]
    ask_price_col = schema_cfg["ask_price_column"]
    ask_qty_col = schema_cfg["ask_quantity_column"]
    batch_id_col = schema_cfg["batch_id_column"]

    url = base_uri.rstrip("/") + "/v1/sql"

    rows_by_asset: Dict[str, List[List[Any]]] = {}

    for asset in assets:
        table_name = f"{table_prefix}{asset.lower()}"

        # Time window based on physical units (dates) from configuration
        start_ts_literal = f"{start_date} 00:00:00"
        end_ts_literal = f"{end_date} 23:59:59"

        sql = (
            f"SELECT {ts_col}, {bid_price_col}, {bid_qty_col}, {ask_price_col}, {ask_qty_col}, {batch_id_col} "
            f"FROM {table_name} "
            f"WHERE {ts_col} >= '{start_ts_literal}' AND {ts_col} <= '{end_ts_literal}' "
            f"AND {bid_price_col} > 0 AND {ask_price_col} > 0 "
            f"ORDER BY {ts_col} ASC"
        )

        logger.info(
            "Fetching order book rows from GreptimeDB at %s (table=%s, asset=%s, start=%s, end=%s)",
            url,
            table_name,
            asset,
            start_ts_literal,
            end_ts_literal,
        )

        try:
            resp = requests.post(
                url,
                data={"sql": sql},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GreptimeDB data fetch failed for table %s (asset=%s): %s",
                table_name,
                asset,
                exc,
            )
            rows_by_asset[asset] = []
            continue

        if not resp.ok:
            logger.warning(
                "GreptimeDB data fetch returned non-OK status for table %s (asset=%s): %s %s",
                table_name,
                asset,
                resp.status_code,
                resp.text,
            )
            rows_by_asset[asset] = []
            continue

        try:
            payload = resp.json()
        except ValueError:
            logger.warning(
                "GreptimeDB data fetch succeeded for table %s (asset=%s, HTTP %s) but response is not JSON.",
                table_name,
                asset,
                resp.status_code,
            )
            rows_by_asset[asset] = []
            continue

        output = payload.get("output")
        if not output:
            logger.warning(
                "GreptimeDB data fetch returned empty output for table %s (asset=%s).",
                table_name,
                asset,
            )
            rows_by_asset[asset] = []
            continue

        records = output[0].get("records") if isinstance(output, list) and output else None
        if not records:
            logger.warning(
                "GreptimeDB data fetch returned no records for table %s (asset=%s).",
                table_name,
                asset,
            )
            rows_by_asset[asset] = []
            continue

        rows = records.get("rows") or []
        rows_by_asset[asset] = rows

        logger.info(
            "Fetched %s rows from GreptimeDB for table %s (asset=%s). execution_time_ms=%s",
            len(rows),
            table_name,
            asset,
            payload.get("execution_time_ms"),
        )

    return rows_by_asset


__all__ = ["check_greptime_connectivity", "fetch_order_book_rows"]
