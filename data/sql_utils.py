"""SQL sanitization utilities for GreptimeDB queries.

Provides defense-in-depth against SQL injection by validating
and escaping query parameters. Uses strict validation with 
clear error messages for fail-fast behavior.

All parameters are validated before interpolation into SQL strings.
Invalid inputs raise SQLValidationError with actionable messages.
"""

from __future__ import annotations

import re
from typing import List

import logging


logger = logging.getLogger(__name__)


# Valid datetime: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?$"
)

# Valid identifier: starts with letter/underscore, contains only alphanumerics/underscores
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SQLValidationError(ValueError):
    """Raised when SQL parameter validation fails.
    
    This exception indicates that a parameter intended for SQL interpolation
    contains invalid or potentially malicious content. The operation should
    be aborted and the input source investigated.
    """

    pass


def validate_datetime_literal(value: str, field_name: str = "datetime") -> str:
    """Validate a datetime literal for SQL interpolation.
    
    Parameters
    ----------
    value:
        Datetime string to validate. Expected formats:
        - 'YYYY-MM-DD' (date only)
        - 'YYYY-MM-DD HH:MM:SS' (date and time)
    field_name:
        Name of the field for error messages.
        
    Returns
    -------
    str:
        The validated and trimmed datetime string.
        
    Raises
    ------
    SQLValidationError:
        If the value doesn't match expected datetime format.
    """
    if not isinstance(value, str):
        raise SQLValidationError(
            f"{field_name} must be a string; got {type(value).__name__}"
        )
    
    value = value.strip()
    if not _DATETIME_PATTERN.match(value):
        raise SQLValidationError(
            f"Invalid {field_name} format: {value!r}. "
            "Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
        )
    
    return value


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate a SQL identifier (table or column name).
    
    Identifiers must:
    - Start with a letter (a-z, A-Z) or underscore (_)
    - Contain only letters, digits (0-9), or underscores
    
    Parameters
    ----------
    value:
        Table or column name to validate.
    field_name:
        Name of the field for error messages.
        
    Returns
    -------
    str:
        The validated identifier.
        
    Raises
    ------
    SQLValidationError:
        If the identifier contains invalid characters.
    """
    if not isinstance(value, str):
        raise SQLValidationError(
            f"{field_name} must be a string; got {type(value).__name__}"
        )
    
    if not value:
        raise SQLValidationError(
            f"{field_name} cannot be empty."
        )
    
    if not _IDENTIFIER_PATTERN.match(value):
        raise SQLValidationError(
            f"Invalid SQL {field_name}: {value!r}. "
            "Must start with letter/underscore and contain only alphanumerics/underscores."
        )
    
    return value


def build_order_book_query(
    table_name: str,
    columns: List[str],
    timestamp_column: str,
    start_datetime: str,
    end_datetime: str,
    end_inclusive: bool,
    bid_price_column: str,
    ask_price_column: str,
) -> str:
    """Build a validated SELECT query for order book data.
    
    Constructs a SQL query with all parameters validated before interpolation.
    The query filters for valid bid/ask prices (> 0) and orders by timestamp.
    
    Parameters
    ----------
    table_name:
        Name of the table to query.
    columns:
        List of column names to select.
    timestamp_column:
        Name of the timestamp column for filtering.
    start_datetime:
        Start of the time range (inclusive).
    end_datetime:
        End of the time range.
    end_inclusive:
        If True, use <= for end boundary; if False, use <.
    bid_price_column:
        Name of the bid price column for filtering.
    ask_price_column:
        Name of the ask price column for filtering.
        
    Returns
    -------
    str:
        A validated SQL SELECT query string.
        
    Raises
    ------
    SQLValidationError:
        If any parameter fails validation.
    """
    # Validate all identifiers
    table_name = validate_identifier(table_name, "table_name")
    validated_columns = [
        validate_identifier(c, f"column[{i}]") 
        for i, c in enumerate(columns)
    ]
    timestamp_column = validate_identifier(timestamp_column, "timestamp_column")
    bid_price_column = validate_identifier(bid_price_column, "bid_price_column")
    ask_price_column = validate_identifier(ask_price_column, "ask_price_column")
    
    # Validate datetime literals
    start_datetime = validate_datetime_literal(start_datetime, "start_datetime")
    end_datetime = validate_datetime_literal(end_datetime, "end_datetime")
    
    # Build query
    end_op = "<=" if end_inclusive else "<"
    col_str = ", ".join(validated_columns)
    
    query = (
        f"SELECT {col_str} "
        f"FROM {table_name} "
        f"WHERE {timestamp_column} >= '{start_datetime}' "
        f"AND {timestamp_column} {end_op} '{end_datetime}' "
        f"AND {bid_price_column} > 0 AND {ask_price_column} > 0 "
        f"ORDER BY {timestamp_column} ASC"
    )
    
    return query


def build_connectivity_check_query(table_name: str) -> str:
    """Build a simple connectivity check query.
    
    Parameters
    ----------
    table_name:
        Name of the table to check.
        
    Returns
    -------
    str:
        A simple SELECT 1 query for connectivity testing.
        
    Raises
    ------
    SQLValidationError:
        If table_name fails validation.
    """
    table_name = validate_identifier(table_name, "table_name")
    return f"SELECT 1 FROM {table_name} LIMIT 1"


__all__ = [
    "SQLValidationError",
    "validate_datetime_literal",
    "validate_identifier",
    "build_order_book_query",
    "build_connectivity_check_query",
]
