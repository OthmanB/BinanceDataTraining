"""Data loading utilities for Binance ML Training Platform (Phase 2 skeleton).

In later phases this module will:
- Connect to configured data sources (e.g., GreptimeDB instances)
- Fetch raw order book and auxiliary data
- Build fully populated DataObject instances

For Phase 2, it constructs an empty-but-structurally valid DataObject from configuration
metadata, to be used for wiring and testing the pipeline.
"""

from typing import Any, Dict, List
import logging

from .data_object import DataObject, build_data_object
from .greptime_client import check_greptime_connectivity, fetch_order_book_rows


logger = logging.getLogger(__name__)


def _get_asset_pairs_from_config(config: Dict[str, Any]) -> List[str]:
    data_cfg = config.get("data", {})
    asset_pairs_cfg = data_cfg.get("asset_pairs", {})
    target_asset = asset_pairs_cfg.get("target_asset")
    correlated_assets = asset_pairs_cfg.get("correlated_assets", []) or []

    assets: List[str] = []
    if target_asset is not None:
        assets.append(str(target_asset))
    for asset in correlated_assets:
        assets.append(str(asset))
    return assets


def create_empty_data_object_from_config(config: Dict[str, Any]) -> DataObject:
    """Create an empty DataObject using configuration metadata.

    This is a Phase 2 placeholder that sets up the expected structure but does not
    fetch any real data yet.
    """

    data_cfg = config.get("data", {})
    time_range_cfg = data_cfg.get("time_range", {})
    order_book_cfg = data_cfg.get("order_book", {})
    targets_cfg = config.get("targets", {})

    asset_pairs = _get_asset_pairs_from_config(config)

    metadata: Dict[str, Any] = {
        "asset_pairs": asset_pairs,
        "time_range": {
            "start": time_range_cfg.get("start_date"),
            "end": time_range_cfg.get("end_date"),
            "cadence_seconds": time_range_cfg.get("cadence_seconds"),
        },
        "num_samples": 0,
        "order_book_depth": order_book_cfg.get("depth_levels"),
    }

    temporal_features: Dict[str, Any] = {
        "local": None,
        "global": None,
    }

    targets: Dict[str, Any] = {
        "asset": data_cfg.get("asset_pairs", {}).get("target_asset"),
        "labels": None,
        "price_changes": None,
        "delta_t_seconds": targets_cfg.get("prediction_horizon_seconds"),
    }

    external_data: Dict[str, Any] = {}

    data_object = build_data_object(
        metadata=metadata,
        order_books={},
        temporal_features=temporal_features,
        targets=targets,
        external_data=external_data,
    )

    logger.info(
        "Created empty DataObject from configuration: asset_pairs=%s, time_range=(%s, %s)",
        metadata.get("asset_pairs"),
        metadata.get("time_range", {}).get("start"),
        metadata.get("time_range", {}).get("end"),
    )

    return data_object


def load_order_book_data(config: Dict[str, Any]) -> DataObject:
    """Load order book data and return a DataObject.

    Phase 2 implementation:
    - Performs a minimal GreptimeDB connectivity check using the HTTP SQL API
    - Returns a structurally valid but empty DataObject.
    Future phases will connect to real data sources based on the configuration
    and populate the DataObject with actual samples.
    """

    logger.info("Loading order book data using Phase 2.5 real ingestion from GreptimeDB.")

    # Connectivity check to the configured GreptimeDB instance
    check_greptime_connectivity(config)

    # Fetch raw rows per asset over the configured physical time window
    rows_by_asset = fetch_order_book_rows(config)

    # Start from the configuration-derived empty DataObject and inject real data
    data_object = create_empty_data_object_from_config(config)

    order_books: Dict[str, Any] = {}

    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])

    target_rows = rows_by_asset.get(target_asset, [])

    for asset, rows in rows_by_asset.items():
        order_books[str(asset)] = {"rows": rows}

    data_object["order_books"] = order_books

    # num_samples is defined as the number of rows for the target asset at this phase
    metadata = data_object.get("metadata", {})
    metadata["num_samples"] = len(target_rows)
    data_object["metadata"] = metadata

    logger.info(
        "Order book data loaded from GreptimeDB: target_asset=%s, num_samples=%s, assets=%s",
        target_asset,
        metadata["num_samples"],
        list(order_books.keys()),
    )

    return data_object


__all__ = ["create_empty_data_object_from_config", "load_order_book_data"]
