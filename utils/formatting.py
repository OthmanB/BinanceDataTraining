"""Shared formatting helpers used across training and observability modules."""

from __future__ import annotations

from typing import Any, Union


def format_bytes(value: Union[int, float]) -> str:
    size = float(max(value, 0.0))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f}{units[idx]}"


def _format_bytes(value: Union[int, float]) -> str:
    return format_bytes(value)


def _fmt_compact(value: float) -> str:
    """Format float using compact notation (e.g., for price boundaries).
    
    Uses the 'g' format specifier for general representation, returning the
    string representation if the g format produces an empty string.
    
    Args:
        value: Float value to format
        
    Returns:
        Formatted string representation
    """
    s = f"{value:g}"
    return s if s else str(value)


def _fmt_metric(value: Any) -> str:
    """Format value for metrics display with 3 decimal places and 'n/a' fallback.
    
    Attempts to format as a float with 3 decimal places. If the value cannot be
    converted to float, returns 'n/a'.
    
    Args:
        value: Value to format (may be None, numeric, or non-numeric)
        
    Returns:
        Formatted string ('X.XXX' format) or 'n/a'
    """
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


__all__ = ["format_bytes", "_format_bytes", "_fmt_compact", "_fmt_metric"]
