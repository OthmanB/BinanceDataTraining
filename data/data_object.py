"""DataObject utilities for Binance ML Training Platform.

DataObject is represented as a Python dictionary with a strict, validated schema,
consistent with the technical specifications.
"""

from typing import Any, Dict


class DataObjectError(Exception):
    """Raised when a DataObject is structurally invalid."""


DataObject = Dict[str, Any]


def build_data_object(
    metadata: Dict[str, Any],
    order_books: Dict[str, Any],
    temporal_features: Dict[str, Any],
    targets: Dict[str, Any],
    external_data: Dict[str, Any],
) -> DataObject:
    """Construct a DataObject from its main components."""

    return {
        "metadata": metadata,
        "order_books": order_books,
        "temporal_features": temporal_features,
        "targets": targets,
        "external_data": external_data,
    }


def validate_data_object_structure(data_object: DataObject) -> None:
    """Validate that a DataObject has the required top-level structure.

    This checks for presence and basic types of the top-level keys:
    - metadata
    - order_books
    - temporal_features
    - targets
    - external_data
    """

    required_keys = (
        "metadata",
        "order_books",
        "temporal_features",
        "targets",
        "external_data",
    )

    for key in required_keys:
        if key not in data_object:
            raise DataObjectError(f"DataObject missing required top-level key: {key}")

    if not isinstance(data_object["metadata"], dict):
        raise DataObjectError("DataObject['metadata'] must be a dict")
    if not isinstance(data_object["order_books"], dict):
        raise DataObjectError("DataObject['order_books'] must be a dict")
    if not isinstance(data_object["temporal_features"], dict):
        raise DataObjectError("DataObject['temporal_features'] must be a dict")
    if not isinstance(data_object["targets"], dict):
        raise DataObjectError("DataObject['targets'] must be a dict")
    if not isinstance(data_object["external_data"], dict):
        raise DataObjectError("DataObject['external_data'] must be a dict")


__all__ = ["DataObject", "DataObjectError", "build_data_object", "validate_data_object_structure"]
