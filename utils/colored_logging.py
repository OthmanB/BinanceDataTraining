"""Colored logging setup for Binance ML Training Platform.

Phase 1 responsibilities:
- Configure Python logging based on YAML config
- Add colored output and function/module context
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

        # Colorize level name and function name
        record.levelname = colored(record.levelname, level_color)
        record.funcName = colored(record.funcName, func_color)

        return super().format(record)


def setup_colored_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up root logger with colored output according to config."""

    logging_cfg = config.get("logging", {})
    level_name = logging_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    colors = logging_cfg.get("colors", {})

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
