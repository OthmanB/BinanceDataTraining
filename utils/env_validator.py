"""Environment variable validation utilities.

Phase 1 responsibilities:
- Read required environment variables from configuration
- Enforce presence at startup when requested
"""

import os
from typing import Any, Dict, List

from .config_loader import ConfigError


def _get_required_env_vars(config: Dict[str, Any]) -> List[str]:
    security = config.get("security", {})
    env_list = security.get("environment_variables", [])
    if not isinstance(env_list, list):
        raise ConfigError("security.environment_variables must be a list")
    return [str(v) for v in env_list]


def validate_environment(config: Dict[str, Any]) -> None:
    """Validate that required environment variables are present.

    Behavior is controlled by:
    - security.validation.check_env_vars_at_startup
    - security.validation.fail_if_missing
    """

    security = config.get("security", {})
    validation_cfg = security.get("validation", {})

    check_at_start = bool(validation_cfg.get("check_env_vars_at_startup", True))
    fail_if_missing = bool(validation_cfg.get("fail_if_missing", True))

    if not check_at_start:
        return

    required_vars = _get_required_env_vars(config)
    missing = [var for var in required_vars if var not in os.environ]

    if missing and fail_if_missing:
        raise ConfigError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )


__all__ = ["validate_environment"]
