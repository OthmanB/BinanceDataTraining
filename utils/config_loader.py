"""Configuration loading and validation utilities.

Loads YAML configuration, resolves environment placeholders, validates against
the schema, and fails fast on invalid or missing parameters.
"""

import os
import copy
from typing import Any, Dict, Set

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


def _validate_normalization_method(config: Dict[str, Any]) -> None:
    """Validate preprocessing.normalization.method at config load time.

    Reject 'robust' normalization as it is not production-ready.
    """
    preprocessing_cfg = config.get("preprocessing")
    if not isinstance(preprocessing_cfg, dict):
        raise ConfigError("preprocessing must be a dict")

    normalization_cfg = preprocessing_cfg.get("normalization")
    if not isinstance(normalization_cfg, dict):
        raise ConfigError("preprocessing.normalization must be a dict")

    method = str(normalization_cfg.get("method") or "")
    if method == "robust":
        raise ConfigError(
            "preprocessing.normalization.method='robust' is not supported "
            "(streaming quantile computation not implemented). "
            "Use 'min_max' or 'standard' instead."
        )
    if method not in {"min_max", "standard"}:
        raise ConfigError(
            f"preprocessing.normalization.method must be 'min_max' or 'standard', got {method!r}"
        )


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
    if isinstance(target_dist_raw, str):
        if target_dist_raw.strip().lower() != "auto":
            raise ConfigError(
                "preprocessing.class_balancing.undersampling.target_distribution must be a list or 'auto'"
            )
        target_dist = None
    else:
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

    if target_dist is not None:
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

    num_classes = _resolve_output_num_classes(config)

    if target_dist is not None and len(target_dist) != num_classes:
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

    num_classes = _resolve_output_num_classes(config)

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


def _resolve_price_class_boundaries(config: Dict[str, Any]) -> None:
    """Resolve targets.price_classes.boundaries.

    Supports:
    - list[float]: explicit boundaries in percent
    - "auto": fit boundaries from snapshot series mid-price moves

    This function mutates config in-place so downstream validation and snapshot
    hashing see an explicit numeric list.
    """

    targets_cfg = config.get("targets")
    if not isinstance(targets_cfg, dict):
        raise ConfigError("targets must be a dict")
    price_classes_cfg = targets_cfg.get("price_classes")
    if not isinstance(price_classes_cfg, dict):
        raise ConfigError("targets.price_classes must be a dict")

    boundaries_raw = price_classes_cfg.get("boundaries")
    if isinstance(boundaries_raw, list):
        if not boundaries_raw:
            raise ConfigError("targets.price_classes.boundaries must be a non-empty list")
        try:
            boundaries = [float(b) for b in boundaries_raw]
        except (TypeError, ValueError) as exc:
            raise ConfigError("targets.price_classes.boundaries must be numeric") from exc
        if any(v <= 0.0 for v in boundaries):
            raise ConfigError("targets.price_classes.boundaries must be > 0")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ConfigError("targets.price_classes.boundaries must be strictly increasing")
        price_classes_cfg["boundaries"] = boundaries
        return

    if isinstance(boundaries_raw, str) and boundaries_raw.strip().lower() == "auto":
        auto_cfg = price_classes_cfg.get("auto")
        if not isinstance(auto_cfg, dict):
            raise ConfigError(
                "targets.price_classes.auto must be a dict when targets.price_classes.boundaries='auto'"
            )

        method = str(auto_cfg.get("method") or "quantile").strip().lower()
        if method not in {"quantile"}:
            raise ConfigError("targets.price_classes.auto.method must be 'quantile'")

        per_window = bool(auto_cfg.get("per_window", True))
        if per_window is not True:
            # The pipeline can still fit once globally, but the current
            # implementation only fits for the active window config.
            raise ConfigError("targets.price_classes.auto.per_window must be true (global fit not implemented yet)")

        fit_on = str(auto_cfg.get("fit_on") or "train").strip().lower()
        if fit_on not in {"train", "full"}:
            raise ConfigError("targets.price_classes.auto.fit_on must be 'train' or 'full'")

        labeling_criteria = str(auto_cfg.get("labeling_criteria") or "max_intensity").strip().lower()
        if labeling_criteria not in {"max_intensity", "up_intensity", "down_intensity"}:
            raise ConfigError(
                "targets.price_classes.auto.labeling_criteria must be one of: max_intensity, up_intensity, down_intensity"
            )

        max_samples_raw = auto_cfg.get("max_samples", 500_000)
        try:
            max_samples = int(max_samples_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("targets.price_classes.auto.max_samples must be an integer") from exc
        if max_samples <= 0:
            raise ConfigError("targets.price_classes.auto.max_samples must be positive")

        random_seed_raw = auto_cfg.get("random_seed", 0)
        try:
            random_seed = int(random_seed_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("targets.price_classes.auto.random_seed must be an integer") from exc
        if random_seed < 0:
            raise ConfigError("targets.price_classes.auto.random_seed must be >= 0")

        # Determine num_classes (may be 'auto' and will be resolved after we set boundaries).
        model_cfg = config.get("model")
        output_cfg = model_cfg.get("output") if isinstance(model_cfg, dict) else None
        if not isinstance(output_cfg, dict):
            raise ConfigError("model.output must be a dict")
        output_type = str(output_cfg.get("type") or "")
        if output_type != "two_head_intensity":
            raise ConfigError(
                "targets.price_classes.boundaries='auto' is only supported for model.output.type='two_head_intensity'"
            )

        num_classes_raw = output_cfg.get("num_classes")
        if num_classes_raw is None:
            raise ConfigError("model.output.num_classes is required for auto boundaries")
        if isinstance(num_classes_raw, str) and num_classes_raw.strip().lower() == "auto":
            raise ConfigError(
                "model.output.num_classes cannot be 'auto' when targets.price_classes.boundaries='auto'. "
                "Set model.output.num_classes to an integer."
            )
        try:
            num_classes = int(num_classes_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError("model.output.num_classes must be an integer for auto boundaries") from exc
        if num_classes < 2:
            raise ConfigError("model.output.num_classes must be >= 2")

        # Fit boundaries from series-only cache (requires snapshot.enabled for directory settings).
        snapshot_cfg = config.get("snapshot")
        if not isinstance(snapshot_cfg, dict) or not bool(snapshot_cfg.get("enabled")):
            raise ConfigError(
                "targets.price_classes.boundaries='auto' requires snapshot.enabled=true (series-cache fitting)"
            )

        try:
            from training.auto_boundaries import fit_price_class_boundaries_from_series_cache
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to import auto boundary fitter: {exc}") from exc

        boundaries = fit_price_class_boundaries_from_series_cache(
            config,
            num_classes=num_classes,
            fit_on=fit_on,
            labeling_criteria=labeling_criteria,
            max_samples=max_samples,
            random_seed=random_seed,
        )
        price_classes_cfg["boundaries"] = boundaries
        return

    raise ConfigError("targets.price_classes.boundaries must be a non-empty list or the string 'auto'")


def _resolve_output_num_classes(config: Dict[str, Any]) -> int:
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

    output_type = str(output_cfg.get("type") or "")
    if isinstance(num_classes_raw, str):
        if num_classes_raw.strip().lower() != "auto":
            raise ConfigError("model.output.num_classes must be an integer or 'auto'")
        if output_type != "two_head_intensity":
            raise ConfigError(
                "model.output.num_classes='auto' is only supported for model.output.type='two_head_intensity'"
            )
        targets_cfg = config.get("targets")
        if not isinstance(targets_cfg, dict):
            raise ConfigError("targets must be a dict")
        price_classes_cfg = targets_cfg.get("price_classes")
        if not isinstance(price_classes_cfg, dict):
            raise ConfigError("targets.price_classes must be a dict")
        boundaries = price_classes_cfg.get("boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            raise ConfigError("targets.price_classes.boundaries must be a non-empty list")
        num_classes = int(len(boundaries) + 1)
        output_cfg["num_classes"] = num_classes
        return num_classes

    try:
        num_classes = int(num_classes_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("model.output.num_classes must be an integer") from exc
    if num_classes < 2:
        raise ConfigError("model.output.num_classes must be >= 2")
    return num_classes


def _validate_model_layer_configs(config: Dict[str, Any]) -> None:
    """Validate model layer dictionaries for unknown keys and basic types.

    The YAML schema validates that layer lists exist, but it does not validate
    per-layer dict keys. This function enforces a strict, fail-fast contract so
    typos like ``normalisation`` do not silently change model behavior.
    """

    def _as_int(value: Any, *, path: str) -> int:
        if not isinstance(value, int):
            raise ConfigError(f"Configuration key '{path}' must be an integer")
        return int(value)

    def _as_number(value: Any, *, path: str) -> float:
        if not isinstance(value, (int, float)):
            raise ConfigError(f"Configuration key '{path}' must be a number")
        return float(value)

    def _validate_dropout(value: Any, *, path: str) -> None:
        dr = _as_number(value, path=path)
        if not (0.0 <= dr <= 1.0):
            raise ConfigError(f"Configuration key '{path}' must be in [0, 1]")

    def _validate_norm(value: Any, *, path: str) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ConfigError(f"Configuration key '{path}' must be a string or null")
        norm = value.lower()
        if norm not in {"batch", "group", "layer"}:
            raise ConfigError(
                f"Configuration key '{path}' must be one of: null, 'batch', 'group', 'layer'"
            )

    def _validate_2d_size(value: Any, *, path: str) -> None:
        if not isinstance(value, list) or len(value) != 2:
            raise ConfigError(f"Configuration key '{path}' must be a list of two integers")
        for j, item in enumerate(value):
            if not isinstance(item, int):
                raise ConfigError(f"Configuration key '{path}[{j}]' must be an integer")

    def _validate_layer_dict(
        layer_cfg: Any,
        *,
        path: str,
        allowed_keys: Set[str],
        required_keys: Set[str],
    ) -> Dict[str, Any]:
        if not isinstance(layer_cfg, dict):
            raise ConfigError(f"Configuration key '{path}' must be a dict")
        for key in layer_cfg.keys():
            key_str = str(key)
            if key_str not in allowed_keys:
                raise ConfigError(
                    f"Unknown configuration key '{path}.{key_str}'. "
                    "Check config/validation_schema.yaml for supported keys."
                )
        for req in required_keys:
            if req not in layer_cfg:
                raise ConfigError(f"Missing required configuration key: '{path}.{req}'")
        return layer_cfg

    def _validate_layer_list(
        layers_cfg: Any,
        *,
        path: str,
        allowed_keys: Set[str],
        required_keys: Set[str],
        kind: str,
    ) -> None:
        if not isinstance(layers_cfg, list):
            raise ConfigError(f"Configuration key '{path}' must be a list")
        for i, layer in enumerate(layers_cfg):
            layer_path = f"{path}[{i}]"
            cfg = _validate_layer_dict(
                layer,
                path=layer_path,
                allowed_keys=allowed_keys,
                required_keys=required_keys,
            )

            if kind == "cnn2d":
                f = _as_int(cfg["filters"], path=f"{layer_path}.filters")
                if f <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}.filters' must be positive")
                _validate_2d_size(cfg["kernel_size"], path=f"{layer_path}.kernel_size")
                if "pool_size" in cfg and cfg["pool_size"] is not None:
                    _validate_2d_size(cfg["pool_size"], path=f"{layer_path}.pool_size")
                if "normalization" in cfg:
                    _validate_norm(cfg.get("normalization"), path=f"{layer_path}.normalization")
                if "dropout" in cfg and cfg["dropout"] is not None:
                    _validate_dropout(cfg["dropout"], path=f"{layer_path}.dropout")

            elif kind == "lstm":
                units = _as_int(cfg["units"], path=f"{layer_path}.units")
                if units <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}.units' must be positive")
                for k in ("dropout", "recurrent_dropout", "post_dropout"):
                    if k in cfg and cfg[k] is not None:
                        _validate_dropout(cfg[k], path=f"{layer_path}.{k}")

            elif kind == "dense":
                units = _as_int(cfg["units"], path=f"{layer_path}.units")
                if units <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}.units' must be positive")
                if "dropout" in cfg and cfg["dropout"] is not None:
                    _validate_dropout(cfg["dropout"], path=f"{layer_path}.dropout")

            elif kind == "conv1d":
                f = _as_int(cfg["filters"], path=f"{layer_path}.filters")
                if f <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}.filters' must be positive")
                if "kernel_size" in cfg and cfg["kernel_size"] is not None:
                    k = _as_int(cfg["kernel_size"], path=f"{layer_path}.kernel_size")
                    if k <= 0:
                        raise ConfigError(f"Configuration key '{layer_path}.kernel_size' must be positive")
                if "pool_size" in cfg and cfg["pool_size"] is not None:
                    p = _as_int(cfg["pool_size"], path=f"{layer_path}.pool_size")
                    if p <= 0:
                        raise ConfigError(f"Configuration key '{layer_path}.pool_size' must be positive")
                if "normalization" in cfg:
                    _validate_norm(cfg.get("normalization"), path=f"{layer_path}.normalization")
                if "dropout" in cfg and cfg["dropout"] is not None:
                    _validate_dropout(cfg["dropout"], path=f"{layer_path}.dropout")

            else:
                raise ConfigError(f"Unsupported model layer kind: {kind}")

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return

    cnn_cfg = model_cfg.get("cnn")
    if isinstance(cnn_cfg, dict) and "layers" in cnn_cfg:
        _validate_layer_list(
            cnn_cfg.get("layers"),
            path="model.cnn.layers",
            allowed_keys={"filters", "kernel_size", "pool_size", "normalization", "dropout"},
            required_keys={"filters", "kernel_size"},
            kind="cnn2d",
        )

    lstm_cfg = model_cfg.get("lstm")
    if isinstance(lstm_cfg, dict) and "layers" in lstm_cfg:
        _validate_layer_list(
            lstm_cfg.get("layers"),
            path="model.lstm.layers",
            allowed_keys={"units", "dropout", "recurrent_dropout", "post_dropout"},
            required_keys={"units"},
            kind="lstm",
        )

    dense_cfg = model_cfg.get("dense")
    if isinstance(dense_cfg, dict) and "layers" in dense_cfg:
        _validate_layer_list(
            dense_cfg.get("layers"),
            path="model.dense.layers",
            allowed_keys={"units", "dropout"},
            required_keys={"units"},
            kind="dense",
        )

    long_term_cfg = model_cfg.get("long_term")
    if not isinstance(long_term_cfg, dict):
        return

    arch_cfg = long_term_cfg.get("architecture")
    if isinstance(arch_cfg, dict):
        conv1d_cfg = arch_cfg.get("conv1d")
        if isinstance(conv1d_cfg, dict) and "layers" in conv1d_cfg:
            _validate_layer_list(
                conv1d_cfg.get("layers"),
                path="model.long_term.architecture.conv1d.layers",
                allowed_keys={"filters", "kernel_size", "pool_size", "normalization", "dropout"},
                required_keys={"filters"},
                kind="conv1d",
            )

        lt_dense_cfg = arch_cfg.get("dense")
        if isinstance(lt_dense_cfg, dict) and "layers" in lt_dense_cfg:
            layers_cfg = lt_dense_cfg.get("layers")
            if not isinstance(layers_cfg, list):
                raise ConfigError("Configuration key 'model.long_term.architecture.dense.layers' must be a list")
            for i, layer in enumerate(layers_cfg):
                layer_path = f"model.long_term.architecture.dense.layers[{i}]"
                if isinstance(layer, int):
                    if int(layer) <= 0:
                        raise ConfigError(f"Configuration key '{layer_path}' must be positive")
                    continue
                cfg = _validate_layer_dict(
                    layer,
                    path=layer_path,
                    allowed_keys={"units", "dropout"},
                    required_keys={"units"},
                )
                units = _as_int(cfg["units"], path=f"{layer_path}.units")
                if units <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}.units' must be positive")
                if "dropout" in cfg and cfg["dropout"] is not None:
                    _validate_dropout(cfg["dropout"], path=f"{layer_path}.dropout")

    # Legacy long-term dense-only format (supported by model builder)
    legacy_dense_cfg = long_term_cfg.get("dense")
    if isinstance(legacy_dense_cfg, dict) and "layers" in legacy_dense_cfg:
        layers_cfg = legacy_dense_cfg.get("layers")
        if not isinstance(layers_cfg, list):
            raise ConfigError("Configuration key 'model.long_term.dense.layers' must be a list")
        for i, layer in enumerate(layers_cfg):
            layer_path = f"model.long_term.dense.layers[{i}]"
            if isinstance(layer, int):
                if int(layer) <= 0:
                    raise ConfigError(f"Configuration key '{layer_path}' must be positive")
                continue
            cfg = _validate_layer_dict(
                layer,
                path=layer_path,
                allowed_keys={"units", "dropout"},
                required_keys={"units"},
            )
            units = _as_int(cfg["units"], path=f"{layer_path}.units")
            if units <= 0:
                raise ConfigError(f"Configuration key '{layer_path}.units' must be positive")
            if "dropout" in cfg and cfg["dropout"] is not None:
                _validate_dropout(cfg["dropout"], path=f"{layer_path}.dropout")


def _validate_market_session_sessions(config: Dict[str, Any]) -> None:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return
    temporal_cfg = data_cfg.get("temporal_features")
    if not isinstance(temporal_cfg, dict):
        return
    market_cfg = temporal_cfg.get("market_session")
    if not isinstance(market_cfg, dict):
        return

    sessions = market_cfg.get("sessions")
    if not isinstance(sessions, list):
        raise ConfigError("Configuration key 'data.temporal_features.market_session.sessions' must be a list")

    allowed = {"name", "start_hour", "end_hour"}
    required = allowed
    seen_names = set()

    for i, session in enumerate(sessions):
        path = f"data.temporal_features.market_session.sessions[{i}]"
        if not isinstance(session, dict):
            raise ConfigError(f"Configuration key '{path}' must be a dict")
        for key in session.keys():
            key_str = str(key)
            if key_str not in allowed:
                raise ConfigError(
                    f"Unknown configuration key '{path}.{key_str}'. "
                    "Check config/validation_schema.yaml for supported keys."
                )
        for req in required:
            if req not in session:
                raise ConfigError(f"Missing required configuration key: '{path}.{req}'")

        name = session.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"Configuration key '{path}.name' must be a non-empty string")
        if name in seen_names:
            raise ConfigError(f"Duplicate market session name in config: {name!r}")
        seen_names.add(name)

        start = session.get("start_hour")
        end = session.get("end_hour")
        if not isinstance(start, int):
            raise ConfigError(f"Configuration key '{path}.start_hour' must be an integer")
        if not isinstance(end, int):
            raise ConfigError(f"Configuration key '{path}.end_hour' must be an integer")
        if not (0 <= start < end <= 24):
            raise ConfigError(
                "Invalid market session hours at '{path}': start_hour={start} end_hour={end} (expected 0 <= start < end <= 24)".format(
                    path=path,
                    start=int(start),
                    end=int(end),
                )
            )


def _validate_multi_database_connections(config: Dict[str, Any]) -> None:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return
    multi_cfg = data_cfg.get("multi_database")
    if not isinstance(multi_cfg, dict):
        return

    enabled = bool(multi_cfg.get("enabled", False))

    connections = multi_cfg.get("connections")
    if not isinstance(connections, list):
        raise ConfigError("Configuration key 'data.multi_database.connections' must be a list")

    allowed = {
        "name",
        "database_uri",
        "table_prefix",
        "request_timeout_seconds",
        "connect_timeout_seconds",
        "max_retries",
        "retry_backoff_factor",
        "time_range",
    }
    required = {"name", "database_uri", "table_prefix", "time_range"}
    required_when_enabled = {
        "request_timeout_seconds",
        "connect_timeout_seconds",
        "max_retries",
        "retry_backoff_factor",
    }

    for i, conn in enumerate(connections):
        path = f"data.multi_database.connections[{i}]"
        if not isinstance(conn, dict):
            raise ConfigError(f"Configuration key '{path}' must be a dict")
        for key in conn.keys():
            key_str = str(key)
            if key_str not in allowed:
                raise ConfigError(
                    f"Unknown configuration key '{path}.{key_str}'. "
                    "Check config/validation_schema.yaml for supported keys."
                )
        for req in required:
            if req not in conn:
                raise ConfigError(f"Missing required configuration key: '{path}.{req}'")

        if enabled:
            for req in required_when_enabled:
                if req not in conn:
                    raise ConfigError(
                        f"Missing required configuration key: '{path}.{req}' (required when data.multi_database.enabled is true)"
                    )

        for k in ("name", "database_uri", "table_prefix"):
            v = conn.get(k)
            if not isinstance(v, str) or not v:
                raise ConfigError(f"Configuration key '{path}.{k}' must be a non-empty string")

        # Validate connection-level timeouts if present (and required when enabled).
        for k in ("request_timeout_seconds", "connect_timeout_seconds", "max_retries"):
            if k in conn and conn.get(k) is not None:
                v = conn.get(k)
                if not isinstance(v, int):
                    raise ConfigError(f"Configuration key '{path}.{k}' must be an integer")
                if k in ("request_timeout_seconds", "connect_timeout_seconds") and int(v) <= 0:
                    raise ConfigError(f"Configuration key '{path}.{k}' must be positive")
                if k == "max_retries" and int(v) < 0:
                    raise ConfigError(f"Configuration key '{path}.{k}' must be >= 0")

        if "retry_backoff_factor" in conn and conn.get("retry_backoff_factor") is not None:
            v = conn.get("retry_backoff_factor")
            if not isinstance(v, (int, float)):
                raise ConfigError(f"Configuration key '{path}.retry_backoff_factor' must be a number")
            if float(v) < 0.0:
                raise ConfigError(f"Configuration key '{path}.retry_backoff_factor' must be >= 0")

        tr = conn.get("time_range")
        tr_path = f"{path}.time_range"
        if not isinstance(tr, dict):
            raise ConfigError(f"Configuration key '{tr_path}' must be a dict")

        tr_allowed = {"start_date", "end_date"}
        for key in tr.keys():
            key_str = str(key)
            if key_str not in tr_allowed:
                raise ConfigError(
                    f"Unknown configuration key '{tr_path}.{key_str}'. "
                    "Check config/validation_schema.yaml for supported keys."
                )
        for req in tr_allowed:
            if req not in tr:
                raise ConfigError(f"Missing required configuration key: '{tr_path}.{req}'")
            v = tr.get(req)
            if not isinstance(v, str) or not v:
                raise ConfigError(f"Configuration key '{tr_path}.{req}' must be a non-empty string")


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


def _get_nested_local(obj: Dict[str, Any], dotted_key: str, *, base_path: str) -> Any:
    parts = dotted_key.split(".")
    current: Any = obj
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: '{base_path}.{dotted_key}'")
        current = current[part]
    return current


def _has_nested_local(obj: Dict[str, Any], dotted_key: str) -> bool:
    parts = dotted_key.split(".")
    current: Any = obj
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _build_allowed_key_tree(dotted_keys: list[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for dotted in dotted_keys:
        dotted_str = str(dotted)
        if not dotted_str:
            continue
        parts = dotted_str.split(".")
        node: Dict[str, Any] = tree
        for part in parts:
            if part not in node or not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
    return tree


def _validate_unknown_keys_recursive(value: Any, allowed_tree: Dict[str, Any], *, path: str) -> None:
    if not isinstance(value, dict):
        return
    if not allowed_tree:
        # Leaf dict: schema does not enumerate children, so treat as open.
        return

    for key, child in value.items():
        key_str = str(key)
        if key_str not in allowed_tree:
            raise ConfigError(
                f"Unknown configuration key '{path}.{key_str}'. "
                "Check config/validation_schema.yaml for supported keys."
            )
        subtree = allowed_tree.get(key_str)
        if isinstance(subtree, dict) and subtree and isinstance(child, dict):
            _validate_unknown_keys_recursive(child, subtree, path=f"{path}.{key_str}")


def _validate_inline_dict_schema(
    value: Any,
    key_schema: Dict[str, Any],
    *,
    full_key: str,
    strict_unknown: bool,
) -> None:
    if not isinstance(value, dict):
        return

    required_keys = key_schema.get("required_keys")
    optional_keys = key_schema.get("optional_keys")
    if not isinstance(required_keys, dict) and not isinstance(optional_keys, dict):
        return

    required_keys = required_keys if isinstance(required_keys, dict) else {}
    optional_keys = optional_keys if isinstance(optional_keys, dict) else {}

    for dotted_key, child_schema in required_keys.items():
        child_schema_dict = child_schema if isinstance(child_schema, dict) else {}
        child_value = _get_nested_local(value, str(dotted_key), base_path=full_key)
        expected_type_name = child_schema_dict.get("type", "any")
        expected_py_type = _TYPE_MAP.get(expected_type_name)
        if expected_py_type is not None and not isinstance(child_value, expected_py_type):
            raise ConfigError(
                f"Configuration key '{full_key}.{dotted_key}' must be of type {expected_type_name} "
                f"but got {type(child_value).__name__}"
            )

        _validate_inline_dict_schema(
            child_value,
            child_schema_dict,
            full_key=f"{full_key}.{dotted_key}",
            strict_unknown=strict_unknown,
        )

    for dotted_key, child_schema in optional_keys.items():
        dotted_key_str = str(dotted_key)
        if not _has_nested_local(value, dotted_key_str):
            continue
        child_schema_dict = child_schema if isinstance(child_schema, dict) else {}
        child_value = _get_nested_local(value, dotted_key_str, base_path=full_key)
        expected_type_name = child_schema_dict.get("type", "any")
        expected_py_type = _TYPE_MAP.get(expected_type_name)
        if expected_py_type is not None and not isinstance(child_value, expected_py_type):
            raise ConfigError(
                f"Configuration key '{full_key}.{dotted_key_str}' must be of type {expected_type_name} "
                f"but got {type(child_value).__name__}"
            )

        _validate_inline_dict_schema(
            child_value,
            child_schema_dict,
            full_key=f"{full_key}.{dotted_key_str}",
            strict_unknown=strict_unknown,
        )

    if strict_unknown:
        allowed_heads = set()
        for dotted_key in list(required_keys.keys()) + list(optional_keys.keys()):
            dotted_key_str = str(dotted_key)
            if not dotted_key_str:
                continue
            head = dotted_key_str.split(".", 1)[0]
            allowed_heads.add(head)

        if allowed_heads:
            for key in value.keys():
                key_str = str(key)
                if key_str not in allowed_heads:
                    raise ConfigError(
                        f"Unknown configuration key '{full_key}.{key_str}'. "
                        "Check config/validation_schema.yaml for supported keys."
                    )


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
        if isinstance(key_schema, dict):
            _validate_inline_dict_schema(
                value,
                key_schema,
                full_key=full_key,
                strict_unknown=strict_unknown,
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

        if isinstance(key_schema, dict):
            _validate_inline_dict_schema(
                value,
                key_schema,
                full_key=full_key,
                strict_unknown=strict_unknown,
            )

    if strict_unknown and isinstance(section_value, dict):
        allowed_tree = _build_allowed_key_tree(list(required_keys.keys()) + list(optional_keys.keys()))
        _validate_unknown_keys_recursive(section_value, allowed_tree, path=section_name)


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

    # Resolve dynamic target boundaries before enforcing num_classes invariants.
    _resolve_price_class_boundaries(resolved_config)

    _validate_normalization_method(resolved_config)
    _validate_two_head_intensity_num_classes(resolved_config)
    _validate_class_balancing_config(resolved_config)
    _validate_model_layer_configs(resolved_config)
    _validate_market_session_sessions(resolved_config)
    _validate_multi_database_connections(resolved_config)

    # Return a deep copy so callers cannot accidentally mutate internal state
    return copy.deepcopy(resolved_config)


__all__ = ["ConfigError", "load_config"]
