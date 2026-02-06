"""Logging utilities for colorized output.

Configures Python logging based on YAML settings and adds colored output with
function/module context.
"""

import logging
from typing import Any, Dict

from termcolor import colored


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
    """Set up root logger with colored output according to config.
    
    Requires logging.level and logging.colors to be present in config.
    These are validated by the config schema.
    """

    logging_cfg = config["logging"]  # Required section, fail fast if missing
    level_name = logging_cfg["level"].upper()  # Required by schema
    level = getattr(logging, level_name, None)
    if level is None:
        raise ValueError(f"Invalid logging level: {level_name}")

    colors = logging_cfg["colors"]  # Required by schema

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs if reconfigured
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        colors=colors,
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


__all__ = ["setup_colored_logging"]
