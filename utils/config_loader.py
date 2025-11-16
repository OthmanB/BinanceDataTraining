"""Configuration loading and validation utilities for Binance ML Training Platform.

Phase 1 responsibilities:
- Load training_config.yaml
- Resolve environment variables in ${VAR} placeholders
- Validate against validation_schema.yaml
- Fail fast on missing or invalid parameters
"""

import os
import copy
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when the configuration or schema is invalid."""


def _load_yaml_file(path: str) -> Any:
    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc


def _resolve_env_placeholders(obj: Any) -> Any:
    """Recursively resolve ${VAR} placeholders in strings using environment variables."""

    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        # Simple ${VAR} substitution; if VAR is missing, raise.
        if "${" in obj:
            result = obj
            start = result.find("${")
            while start != -1:
                end = result.find("}", start)
                if end == -1:
                    raise ConfigError(f"Unclosed environment placeholder in value: {obj}")
                var_name = result[start + 2 : end]
                if var_name not in os.environ:
                    raise ConfigError(
                        f"Environment variable '{var_name}' required by configuration is not set"
                    )
                value = os.environ[var_name]
                result = result[:start] + value + result[end + 1 :]
                start = result.find("${", start + len(value))
            return result
        return obj
    return obj


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "list": list,
    "dict": dict,
    "any": (str, int, float, bool, dict, list, type(None)),
}


def _get_nested(config: Dict[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: '{dotted_key}'")
        current = current[part]
    return current


def _validate_section(config: Dict[str, Any], section_name: str, section_schema: Dict[str, Any]) -> None:
    if section_name not in config:
        raise ConfigError(f"Missing required section in configuration: '{section_name}'")
    section_value = config[section_name]
    expected_type = _TYPE_MAP.get(section_schema.get("type", "dict"))
    if expected_type is not None and not isinstance(section_value, expected_type):
        raise ConfigError(
            f"Section '{section_name}' must be of type {section_schema.get('type')} "
            f"but got {type(section_value).__name__}"
        )

    required_keys = section_schema.get("required_keys", {})
    for dotted_key, key_schema in required_keys.items():
        full_key = f"{section_name}.{dotted_key}" if dotted_key else section_name
        value = _get_nested(config, full_key)
        expected_type_name = key_schema.get("type", "any")
        expected_py_type = _TYPE_MAP.get(expected_type_name)
        if expected_py_type is not None and not isinstance(value, expected_py_type):
            raise ConfigError(
                f"Configuration key '{full_key}' must be of type {expected_type_name} "
                f"but got {type(value).__name__}"
            )


def _validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    required_sections = schema.get("required_sections", [])
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required top-level section: '{section}'")

    sections_schema = schema.get("sections", {})
    for section_name, section_schema in sections_schema.items():
        # Only validate sections that are present or required
        if section_name in config or section_name in required_sections:
            _validate_section(config, section_name, section_schema)


def load_config(
    config_path: str = "config/training_config.yaml",
    schema_path: str = "config/validation_schema.yaml",
) -> Dict[str, Any]:
    """Load, resolve, and validate the training configuration.

    Returns a deep-copied, immutable-friendly dictionary.
    """

    raw_config = _load_yaml_file(config_path)
    schema = _load_yaml_file(schema_path)

    # Resolve environment placeholders before type checking
    resolved_config = _resolve_env_placeholders(copy.deepcopy(raw_config))

    _validate_config_schema(resolved_config, schema)

    # Return a deep copy so callers cannot accidentally mutate internal state
    return copy.deepcopy(resolved_config)


__all__ = ["ConfigError", "load_config"]
