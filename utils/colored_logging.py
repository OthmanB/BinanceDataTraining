"""Logging utilities for colorized and structured JSON output.

Configures Python logging based on YAML settings. Supports two formats:
- 'colored': Human-readable colorized console output (default)
- 'json': Structured JSON lines for production log aggregation
"""

import json as _json
import logging
import time
from typing import Any, Dict

from termcolor import colored


class StructuredJsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Each line contains: timestamp, level, logger name, function, line number,
    and the formatted message. Suitable for ingestion by log aggregators
    (e.g., Loki, Datadog, ELK).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return _json.dumps(payload, default=str)


class ColoredFormatter(logging.Formatter):
    def __init__(self, colors: Dict[str, str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.colors = colors

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname.lower()
        level_color = self.colors.get(level, "white")
        func_color = self.colors.get("function_names", "cyan")

        # Colorize level name and function name for this formatter only.
        # Avoid mutating the LogRecord shared across handlers (e.g., file logs).
        original_levelname = record.levelname
        original_func_name = record.funcName
        try:
            record.levelname = colored(original_levelname, level_color)
            record.funcName = colored(original_func_name, func_color)
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.funcName = original_func_name


def setup_colored_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up root logger according to config.
    
    Requires logging.level and logging.colors to be present in config.
    These are validated by the config schema.

    Supports an optional logging.format key:
    - 'colored' (default): Human-readable colorized console output
    - 'json': Structured JSON lines for production log aggregation
    """

    logging_cfg = config["logging"]  # Required section, fail fast if missing
    level_name = logging_cfg["level"].upper()  # Required by schema
    level = getattr(logging, level_name, None)
    if level is None:
        raise ValueError(f"Invalid logging level: {level_name}")

    log_format = str(logging_cfg.get("format", "colored")).lower()
    if log_format not in ("colored", "json"):
        raise ValueError(
            f"logging.format must be 'colored' or 'json'; got {log_format!r}"
        )

    colors = logging_cfg["colors"]  # Required by schema

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs if reconfigured
    logger.handlers.clear()

    handler = logging.StreamHandler()

    if log_format == "json":
        formatter: logging.Formatter = StructuredJsonFormatter(
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = ColoredFormatter(
            colors=colors,
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


__all__ = ["setup_colored_logging", "StructuredJsonFormatter"]
