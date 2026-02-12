"""Configuration loading and validation utilities.

Loads YAML configuration, resolves environment placeholders, validates against
the schema, and fails fast on invalid or missing parameters.
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


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


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


def _validate_class_balancing_config(config: Dict[str, Any]) -> None:
    """Validate preprocessing.class_balancing semantic constraints.

    The YAML schema validates types but cannot express conditional requirements.
    This function enforces those rules so misconfigured balancing fails fast.
    """

    preprocessing_cfg = config.get("preprocessing")
    if not isinstance(preprocessing_cfg, dict):
        raise ConfigError("preprocessing must be a dict")

    cb_cfg = preprocessing_cfg.get("class_balancing")
    if not isinstance(cb_cfg, dict):
        raise ConfigError("preprocessing.class_balancing must be a dict")

    method = str(cb_cfg.get("method") or "")
    if method != "undersampling":
        raise ConfigError(
            "preprocessing.class_balancing.method must be 'undersampling' (legacy methods removed)"
        )

    enabled = bool(cb_cfg.get("enabled", False))
    if not enabled:
        return

    undersampling_cfg = cb_cfg.get("undersampling")
    if not isinstance(undersampling_cfg, dict):
        raise ConfigError(
            "preprocessing.class_balancing.undersampling must be provided when class_balancing.enabled is true"
        )

    criteria = str(undersampling_cfg.get("labeling_criteria") or "")
    if criteria not in {"max_intensity", "up_intensity", "down_intensity"}:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.labeling_criteria must be one of: "
            "'max_intensity', 'up_intensity', 'down_intensity'"
        )

    selection_policy = str(undersampling_cfg.get("selection_policy") or "")
    if selection_policy not in {"uniform_time", "kmeans", "random"}:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.selection_policy must be one of: "
            "'uniform_time', 'kmeans', 'random'"
        )

    target_dist_raw = undersampling_cfg.get("target_distribution")
    if not isinstance(target_dist_raw, list) or not target_dist_raw:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.target_distribution must be a non-empty list"
        )

    try:
        target_dist = [float(v) for v in target_dist_raw]
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.target_distribution must contain only numbers"
        ) from exc

    if any(v < 0.0 for v in target_dist):
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.target_distribution values must be >= 0"
        )

    total_weight = float(sum(target_dist))
    if total_weight <= 0.0:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.target_distribution must have a positive sum"
        )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ConfigError("model must be a dict")
    output_cfg = model_cfg.get("output")
    if not isinstance(output_cfg, dict):
        raise ConfigError("model.output must be a dict")

    try:
        num_classes_raw = output_cfg["num_classes"]
    except KeyError as exc:
        raise ConfigError("model.output.num_classes is required") from exc
    try:
        num_classes = int(num_classes_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("model.output.num_classes must be an integer") from exc
    if num_classes < 2:
        raise ConfigError("model.output.num_classes must be >= 2")

    if len(target_dist) != num_classes:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.target_distribution length must equal model.output.num_classes: "
            f"len(target_distribution)={len(target_dist)}, num_classes={num_classes}"
        )

    try:
        min_samples_raw = undersampling_cfg["min_samples_after_balance"]
    except KeyError as exc:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_samples_after_balance is required when enabled"
        ) from exc
    try:
        min_samples = int(min_samples_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_samples_after_balance must be an integer"
        ) from exc
    if min_samples <= 0:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_samples_after_balance must be positive"
        )

    try:
        min_fraction_raw = undersampling_cfg["min_fraction_after_balance"]
    except KeyError as exc:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_fraction_after_balance is required when enabled"
        ) from exc
    try:
        min_fraction = float(min_fraction_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_fraction_after_balance must be a number"
        ) from exc
    if not (0.0 < min_fraction <= 1.0):
        raise ConfigError(
            "preprocessing.class_balancing.undersampling.min_fraction_after_balance must be in (0, 1]"
        )

    random_seed = undersampling_cfg.get("random_seed")
    if random_seed is not None:
        try:
            seed_int = int(random_seed)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.random_seed must be an integer"
            ) from exc
        if seed_int < 0:
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.random_seed must be >= 0"
            )


def _validate_two_head_intensity_num_classes(config: Dict[str, Any]) -> None:
    """Validate that output.num_classes matches targets.price_classes boundaries.

    This invariant is enforced in multiple runtime components. Validating it at
    config-load time provides clearer, earlier error messages (especially before
    launching parallel HPO workers).
    """

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return
    output_cfg = model_cfg.get("output")
    if not isinstance(output_cfg, dict):
        return

    output_type = str(output_cfg.get("type") or "")
    if output_type != "two_head_intensity":
        return

    try:
        num_classes = int(output_cfg["num_classes"])
    except Exception as exc:  # noqa: BLE001
        raise ConfigError("model.output.num_classes must be an integer") from exc

    targets_cfg = config.get("targets")
    if not isinstance(targets_cfg, dict):
        raise ConfigError("targets must be a dict")
    price_classes_cfg = targets_cfg.get("price_classes")
    if not isinstance(price_classes_cfg, dict):
        raise ConfigError("targets.price_classes must be a dict")

    boundaries = price_classes_cfg.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ConfigError("targets.price_classes.boundaries must be a non-empty list")

    expected = len(boundaries) + 1
    if num_classes != expected:
        raise ConfigError(
            "model.output.num_classes must equal len(targets.price_classes.boundaries) + 1: "
            f"num_classes={num_classes}, boundaries_len={len(boundaries)}, expected={expected}"
        )


def _get_nested(config: Dict[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: '{dotted_key}'")
        current = current[part]
    return current


def _has_nested(config: Dict[str, Any], dotted_key: str) -> bool:
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _validate_section(
    config: Dict[str, Any],
    section_name: str,
    section_schema: Dict[str, Any],
    *,
    strict_unknown: bool = False,
) -> None:
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

    optional_keys = section_schema.get("optional_keys", {})
    for dotted_key, key_schema in optional_keys.items():
        full_key = f"{section_name}.{dotted_key}" if dotted_key else section_name
        if not _has_nested(config, full_key):
            continue
        value = _get_nested(config, full_key)
        expected_type_name = key_schema.get("type", "any")
        expected_py_type = _TYPE_MAP.get(expected_type_name)
        if expected_py_type is not None and not isinstance(value, expected_py_type):
            raise ConfigError(
                f"Configuration key '{full_key}' must be of type {expected_type_name} "
                f"but got {type(value).__name__}"
            )

    if strict_unknown and isinstance(section_value, dict):
        for key in section_value.keys():
            key_str = str(key)
            allowed = False
            for dotted_key in list(required_keys.keys()) + list(optional_keys.keys()):
                if not dotted_key:
                    continue
                head = dotted_key.split(".", 1)[0]
                if head == key_str:
                    allowed = True
                    break
            if not allowed:
                raise ConfigError(
                    f"Unknown configuration key '{section_name}.{key_str}'. "
                    "Check config/validation_schema.yaml for supported keys."
                )


def _validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    required_sections = schema.get("required_sections", [])
    strict_unknown = bool(schema.get("strict_unknown_keys", False))
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required top-level section: '{section}'")

    sections_schema = schema.get("sections", {})
    for section_name, section_schema in sections_schema.items():
        # Only validate sections that are present or required
        if section_name in config or section_name in required_sections:
            _validate_section(config, section_name, section_schema, strict_unknown=strict_unknown)


def load_config(
    config_path: str = "config/training_config.yaml",
    schema_path: str = "config/validation_schema.yaml",
) -> Dict[str, Any]:
    """Load, resolve, and validate the training configuration.

    Returns a deep-copied, immutable-friendly dictionary.
    """

    def _load_with_base(path: str) -> Dict[str, Any]:
        seen_paths = set()
        current_path = path
        config = _load_yaml_file(current_path)
        while "base_config" in config:
            base_config_path = config["base_config"]
            if not isinstance(base_config_path, str) or not base_config_path:
                raise ConfigError("base_config must be a non-empty string path")
            if not os.path.isabs(base_config_path):
                base_config_path = os.path.join(
                    os.path.dirname(os.path.abspath(current_path)),
                    base_config_path,
                )
            if base_config_path in seen_paths:
                raise ConfigError("Detected recursive base_config references")
            seen_paths.add(base_config_path)
            base_config = _load_yaml_file(base_config_path)
            override_config = {k: v for k, v in config.items() if k != "base_config"}
            config = _deep_merge(base_config, override_config)
            current_path = base_config_path
        return config

    raw_config = _load_with_base(config_path)
    schema = _load_yaml_file(schema_path)

    # Resolve environment placeholders before type checking
    resolved_config = _resolve_env_placeholders(copy.deepcopy(raw_config))

    _validate_config_schema(resolved_config, schema)

    _validate_class_balancing_config(resolved_config)
    _validate_two_head_intensity_num_classes(resolved_config)

    # Return a deep copy so callers cannot accidentally mutate internal state
    return copy.deepcopy(resolved_config)


__all__ = ["ConfigError", "load_config"]
