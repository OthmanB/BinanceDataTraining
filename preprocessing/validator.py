"""Preprocessing validation utilities.

Phase 2 focuses on structural validation of DataObject instances.
"""

from typing import Any, Dict
import logging

from data.data_object import (
    DataObject,
    DataObjectError,
    validate_data_object_structure,
)


logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails during preprocessing."""


def validate_data_object(data_object: DataObject) -> None:
    """Validate the structural integrity of a DataObject.

    Wraps structural checks from data.data_object and raises DataValidationError
    on failure.
    """

    try:
        validate_data_object_structure(data_object)
    except DataObjectError as exc:
        logger.error("DataObject structural validation failed: %s", exc)
        raise DataValidationError(str(exc)) from exc


__all__ = ["DataValidationError", "validate_data_object"]
