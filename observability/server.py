"""Minimal observability server with HTMX dashboard and Prometheus metrics.

Configuration:
- Non-secret settings (host/port/paths) can come from `config/observability.yaml`.
- Secrets (Basic Auth user/password) must come from environment variables.

Environment variable overrides (take precedence over YAML):
- OBSERVABILITY_HOST, OBSERVABILITY_PORT
- RUN_STATE_PATH, RUN_LOG_PATH
- OBSERVABILITY_ALLOW_RUN_CONTROL
- OBSERVABILITY_ALLOWED_CONFIGS_GLOB
- OBSERVABILITY_STATIC_DIR
- OBSERVABILITY_TAIL_MAX_LINES

Required environment variables:
- OBSERVABILITY_USER, OBSERVABILITY_PASSWORD
"""

from __future__ import annotations

import base64
import argparse
from dataclasses import dataclass
from datetime import datetime
import html
import json
import logging
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .run_state import RunStateWriter, load_run_state


logger = logging.getLogger(__name__)


class ServerConfig:
    def __init__(
        self,
        *,
        user: str,
        password: str,
        run_state_path: str,
        run_log_path: str,
        host: str,
        port: int,
        allow_run_control: bool,
        allowed_configs_glob: str,
        static_dir: str,
        tail_max_lines: int,
        config_path: str,
        file_config: Dict[str, Any],
    ) -> None:
        self.user = user
        self.password = password
        self.run_state_path = run_state_path
        self.run_log_path = run_log_path
        self.host = host
        self.port = port
        self.allow_run_control = allow_run_control
        self.allowed_configs_glob = allowed_configs_glob
        self.static_dir = static_dir
        self.tail_max_lines = tail_max_lines
        self.config_path = config_path
        self.file_config = dict(file_config)

    @staticmethod
    def _parse_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        raise ValueError(f"Invalid boolean value: {value!r}")

    @staticmethod
    def _parse_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return int(value)
        raise ValueError(f"Invalid integer value: {value!r}")

    @staticmethod
    def _resolve_env_placeholders(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ServerConfig._resolve_env_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ServerConfig._resolve_env_placeholders(v) for v in obj]
        if isinstance(obj, str) and "${" in obj:
            result = obj
            start = result.find("${")
            while start != -1:
                end = result.find("}", start)
                if end == -1:
                    raise RuntimeError(f"Unclosed environment placeholder in value: {obj}")
                var_name = result[start + 2 : end]
                if var_name not in os.environ:
                    raise RuntimeError(
                        f"Environment variable '{var_name}' required by observability config is not set"
                    )
                value = os.environ[var_name]
                result = result[:start] + value + result[end + 1 :]
                start = result.find("${", start + len(value))
            return result
        return obj

    @staticmethod
    def _load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
        if not config_path:
            return {}
        path_obj = Path(config_path)
        if not path_obj.exists():
            return {}
        try:
            import yaml  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyYAML is required to load observability config") from exc
        data = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Observability config must be a mapping, got: {type(data).__name__}")
        return ServerConfig._resolve_env_placeholders(data)

    @classmethod
    def from_sources(cls, *, config_path: Optional[str]) -> "ServerConfig":
        resolved_path = str(config_path) if config_path else ""
        file_cfg = cls._load_yaml_config(resolved_path)

        # Secrets must be provided by environment variables.
        user = os.environ.get("OBSERVABILITY_USER")
        password = os.environ.get("OBSERVABILITY_PASSWORD")
        if not user or not password:
            raise RuntimeError("OBSERVABILITY_USER and OBSERVABILITY_PASSWORD must be set")

        host = os.environ.get("OBSERVABILITY_HOST") or str(file_cfg.get("host") or "127.0.0.1")
        port = cls._parse_int(os.environ.get("OBSERVABILITY_PORT") or file_cfg.get("port"), default=8008)

        run_state_path = os.environ.get("RUN_STATE_PATH") or file_cfg.get("run_state_path")
        run_log_path = os.environ.get("RUN_LOG_PATH") or file_cfg.get("run_log_path")
        if not run_state_path:
            raise RuntimeError("RUN_STATE_PATH must be set (env or config file)")
        if not run_log_path:
            raise RuntimeError("RUN_LOG_PATH must be set (env or config file)")

        allow_run_control = cls._parse_bool(
            os.environ.get("OBSERVABILITY_ALLOW_RUN_CONTROL")
            if os.environ.get("OBSERVABILITY_ALLOW_RUN_CONTROL") is not None
            else file_cfg.get("allow_run_control"),
            default=False,
        )

        allowed_configs_glob = (
            os.environ.get("OBSERVABILITY_ALLOWED_CONFIGS_GLOB")
            or str(file_cfg.get("allowed_configs_glob") or "config/e2e_trial_*.yaml")
        )
        static_dir = os.environ.get("OBSERVABILITY_STATIC_DIR") or str(file_cfg.get("static_dir") or "static")
        tail_max_lines = cls._parse_int(
            os.environ.get("OBSERVABILITY_TAIL_MAX_LINES") or file_cfg.get("tail_max_lines"),
            default=200,
        )

        return cls(
            user=str(user),
            password=str(password),
            run_state_path=str(run_state_path),
            run_log_path=str(run_log_path),
            host=str(host),
            port=int(port),
            allow_run_control=bool(allow_run_control),
            allowed_configs_glob=str(allowed_configs_glob),
            static_dir=str(static_dir),
            tail_max_lines=int(tail_max_lines),
            config_path=resolved_path,
            file_config=file_cfg,
        )


class ServerState:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.process_lock = threading.Lock()
        self.selected_config_path: Optional[str] = None
        self.selection_lock = threading.Lock()
        self.config_mode = "extended"
        self.mode_lock = threading.Lock()

    def is_running(self) -> bool:
        with self.process_lock:
            return self.process is not None and self.process.poll() is None

    def set_selected_config_path(self, path: Optional[str]) -> None:
        with self.selection_lock:
            self.selected_config_path = path

    def get_selected_config_path(self) -> Optional[str]:
        with self.selection_lock:
            return self.selected_config_path

    def set_config_mode(self, mode: str) -> None:
        if mode not in {"simple", "extended"}:
            mode = "extended"
        with self.mode_lock:
            self.config_mode = mode

    def get_config_mode(self) -> str:
        with self.mode_lock:
            return self.config_mode

    def start_run(self, config_path: str) -> Tuple[bool, str]:
        if not self.config.allow_run_control:
            return False, "Run control is disabled"
        with self.process_lock:
            if self.process is not None and self.process.poll() is None:
                return False, "A run is already in progress"

            cmd = [sys.executable, "main.py", "--config", config_path]
            env = os.environ.copy()
            env["RUN_STATE_PATH"] = self.config.run_state_path
            env["RUN_LOG_PATH"] = self.config.run_log_path
            self.process = subprocess.Popen(cmd, env=env)
            return True, "Run started"

    def stop_run(self) -> Tuple[bool, str]:
        if not self.config.allow_run_control:
            return False, "Run control is disabled"
        with self.process_lock:
            if self.process is None or self.process.poll() is not None:
                return False, "No active run to stop"
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

        writer = RunStateWriter(self.config.run_state_path)
        writer.set_error("Run stopped by operator", traceback_text=None)
        return True, "Run stopped"


def _check_auth(headers: Any, user: str, password: str) -> bool:
    auth_header = headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    try:
        encoded = auth_header.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode("utf-8")
    except Exception:  # noqa: BLE001
        return False
    return decoded == f"{user}:{password}"


def _tail_log(path: str, *, max_lines: int) -> List[str]:
    path_obj = Path(path)
    if not path_obj.exists():
        return ["Log file not found"]
    lines = path_obj.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _allowed_configs(glob_pattern: str) -> List[str]:
    return sorted(str(p) for p in Path().glob(glob_pattern))


def _safe_static_file_path(*, static_dir: str, request_path: str) -> Optional[Path]:
    if not request_path.startswith("/static/"):
        return None
    rel = request_path[len("/static/") :]
    if not rel or rel.startswith("/"):
        return None
    base = Path(static_dir).resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate

_CONFIG_ROOT = Path("config").resolve()
_SCHEMA_FIELDS: Optional[List["SchemaField"]] = None
_HINT_MAP: Optional[Dict[str, str]] = None
_DEFAULT_SIMPLE_CONFIG: Optional[Dict[str, Any]] = None
_MISSING = object()

_SELECT_OPTIONS: Dict[str, List[str]] = {
    "run_mode.mode": ["production", "trial"],
    "data.source_type": ["database", "file"],
    "data.asset_pairs.alignment.method": ["interpolate", "bucket"],
    "data.asset_pairs.alignment.missing_policy": ["forward_fill", "skip", "error"],
    "data.order_book.representation": ["full", "aggregated", "hybrid"],
    "data.order_book.hybrid.bin_strategy": ["equal_width", "log_spaced"],
    "preprocessing.normalization.method": ["min_max", "standard", "robust"],
    "preprocessing.feature_engineering.volume_proxy_method": ["top_of_book", "total_depth"],
    "preprocessing.feature_engineering.edge_decay.method": ["linear", "exponential"],
    "preprocessing.train_test_split.method": ["chronological"],
    "preprocessing.class_balancing.method": ["class_weights", "oversampling", "undersampling"],
    "model.framework": ["keras"],
    "model.backend": ["tensorflow"],
    "model.architecture": ["CNN_LSTM_MultiClass"],
    "model.input_representation.strategy": ["stacked_channels"],
    "model.input_representation.temporal_features.integration_mode": ["concat_channels"],
    "training.missing_snapshot_strategy": ["fail", "skip", "synthetic"],
    "training.sample_weighting.method": ["exponential_decay"],
    "training.sample_weighting.apply_to": ["loss_function"],
    "training.fine_tuning.freeze_layers": ["none", "cnn", "cnn_lstm", "all_but_output"],
    "evaluation.post_hoc_calibration.method": ["temperature_scaling", "isotonic"],
    "evaluation.backtesting.signal_strategy": ["net_intensity", "threshold"],
    "evaluation.backtesting.position_sizing": ["equal", "confidence"],
    "evaluation.missing_snapshot_strategy": ["fail", "skip", "synthetic"],
}


@dataclass(frozen=True)
class SchemaField:
    section: str
    dotted_key: str
    full_key: str
    field_type: str


def _escape_text(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to parse configuration files") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a mapping, got {type(data).__name__}")
    return data


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


def _normalize_selected_path(path: str) -> str:
    cleaned = path.strip().lstrip("./")
    if not cleaned:
        return ""
    if not cleaned.startswith("config/"):
        cleaned = f"config/{cleaned}"
    return cleaned


def _resolve_selected_path(path: str) -> Optional[Path]:
    normalized = _normalize_selected_path(path)
    if not normalized:
        return None
    rel = normalized[len("config/") :]
    candidate = (_CONFIG_ROOT / rel).resolve()
    try:
        candidate.relative_to(_CONFIG_ROOT)
    except ValueError:
        return None
    return candidate


def _list_config_entries(browse_path: str) -> Tuple[str, Optional[str], List[str], List[str]]:
    if not _CONFIG_ROOT.exists():
        return "", None, [], []
    rel = browse_path.strip().lstrip("/")
    current = (_CONFIG_ROOT / rel).resolve()
    try:
        current.relative_to(_CONFIG_ROOT)
    except ValueError:
        current = _CONFIG_ROOT
        rel = ""
    if not current.exists() or not current.is_dir():
        current = _CONFIG_ROOT
        rel = ""
    rel_path = current.relative_to(_CONFIG_ROOT).as_posix()
    if rel_path == ".":
        rel_path = ""
    parent_rel = None
    if current != _CONFIG_ROOT:
        parent_rel = current.parent.relative_to(_CONFIG_ROOT).as_posix()
        if parent_rel == ".":
            parent_rel = ""

    dirs = sorted([p.name for p in current.iterdir() if p.is_dir()])
    files = sorted(
        [
            p.name
            for p in current.iterdir()
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        ]
    )
    return rel_path, parent_rel, dirs, files


def _join_rel(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p]
    return "/".join([p for p in cleaned if p])


def _load_training_config(config_path: Path) -> Dict[str, Any]:
    raw_config = _load_yaml_file(config_path)
    base_config_path = raw_config.get("base_config")
    if base_config_path:
        base_path = Path(base_config_path)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        base_config = _load_yaml_file(base_path)
        override_config = {k: v for k, v in raw_config.items() if k != "base_config"}
        return _deep_merge(base_config, override_config)
    return raw_config


def _load_validation_schema() -> Dict[str, Any]:
    schema_path = _CONFIG_ROOT / "validation_schema.yaml"
    return _load_yaml_file(schema_path)


def _get_schema_fields() -> List[SchemaField]:
    global _SCHEMA_FIELDS
    if _SCHEMA_FIELDS is not None:
        return _SCHEMA_FIELDS
    schema = _load_validation_schema()
    required_sections = schema.get("required_sections", [])
    sections_schema = schema.get("sections", {})
    fields: List[SchemaField] = []
    for section in required_sections:
        section_schema = sections_schema.get(section, {})
        required_keys = section_schema.get("required_keys", {})
        for dotted_key, key_schema in required_keys.items():
            field_type = str(key_schema.get("type", "any"))
            full_key = f"{section}.{dotted_key}" if dotted_key else section
            fields.append(SchemaField(section=section, dotted_key=dotted_key, full_key=full_key, field_type=field_type))
    _SCHEMA_FIELDS = fields
    return fields


def _is_hidden_in_simple(full_key: str) -> bool:
    if full_key.startswith("security."):
        return True
    if full_key.startswith("data.connection."):
        return True
    if full_key == "data.source_type":
        return True
    if full_key == "data.multi_database.strategy":
        return True
    if full_key.startswith("data.multi_database.connections"):
        return True
    if full_key.startswith("data.validation."):
        return True
    if full_key.startswith("data.order_book."):
        return True
    if full_key.startswith("snapshot."):
        return True
    if full_key.startswith("mlflow."):
        return True
    if full_key.startswith("logging."):
        return True
    if full_key.startswith("diagnostics."):
        return True
    if full_key in {"model.framework", "model.backend", "model.architecture"}:
        return True
    if full_key.startswith("model.input_representation."):
        return True
    if full_key.startswith("training.dataset_cache."):
        return True
    return False


def _get_schema_fields_for_mode(mode: str) -> List[SchemaField]:
    fields = _get_schema_fields()
    if mode != "simple":
        return fields
    return [field for field in fields if not _is_hidden_in_simple(field.full_key)]


def _build_hint_map() -> Dict[str, str]:
    config_path = _CONFIG_ROOT / "training_config.yaml"
    if not config_path.exists():
        return {}
    hints: Dict[str, str] = {}
    stack: List[Tuple[int, str]] = []
    comment_block: List[str] = []

    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            comment_block = []
            continue
        if stripped.startswith("#"):
            comment_block.append(stripped.lstrip("#").strip())
            continue
        if ":" not in stripped:
            comment_block = []
            continue
        key_part = stripped.split(":", 1)[0].strip()
        if key_part.startswith("-"):
            comment_block = []
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack.append((indent, key_part))
        full_key = ".".join([item[1] for item in stack])

        inline_comment = ""
        if "#" in line:
            before_hash, after_hash = line.split("#", 1)
            if ":" in before_hash:
                inline_comment = after_hash.strip()
        if inline_comment:
            hints[full_key] = inline_comment
        elif comment_block:
            hints[full_key] = " ".join(comment_block)
        comment_block = []

    return hints


def _get_hint_map() -> Dict[str, str]:
    global _HINT_MAP
    if _HINT_MAP is None:
        _HINT_MAP = _build_hint_map()
        _HINT_MAP.setdefault(
            "training.runtime.device",
            "Compute device target for training. Use 'cpu' or 'gpu'.",
        )
        _HINT_MAP.setdefault(
            "training.runtime.gpu_visible_devices",
            "Optional CUDA device list (e.g. '0' or '0,1'); use null for default visibility.",
        )
    return _HINT_MAP


def _load_default_simple_config() -> Dict[str, Any]:
    global _DEFAULT_SIMPLE_CONFIG
    if _DEFAULT_SIMPLE_CONFIG is not None:
        return _DEFAULT_SIMPLE_CONFIG
    default_path = _CONFIG_ROOT / "training_config_default.yaml"
    _DEFAULT_SIMPLE_CONFIG = _load_yaml_file(default_path)
    return _DEFAULT_SIMPLE_CONFIG


def _get_nested_value(config: Dict[str, Any], full_key: str) -> Any:
    current: Any = config
    for part in full_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_nested_value(config: Dict[str, Any], full_key: str, value: Any) -> None:
    parts = full_key.split(".")
    current = config
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _missing_fields(config: Dict[str, Any], schema_fields: List[SchemaField]) -> List[str]:
    missing: List[str] = []
    for field in schema_fields:
        value = _get_nested_value(config, field.full_key)
        if value is _MISSING:
            missing.append(field.full_key)
            continue
        if value is None and field.field_type != "any":
            missing.append(field.full_key)
            continue
        if isinstance(value, str) and value.strip() == "":
            missing.append(field.full_key)
    return missing


def _render_value(value: Any, field_type: str) -> Any:
    if field_type == "boolean":
        try:
            return ServerConfig._parse_bool(value, default=False)
        except ValueError:
            return False
    if value is None:
        if field_type == "any":
            return "null"
        return ""
    if field_type in {"list", "dict", "any"}:
        try:
            import yaml  # type: ignore[import]
        except Exception:  # noqa: BLE001
            return str(value)
        return yaml.safe_dump(value, sort_keys=False).strip()
    return str(value)


def _build_render_values_from_config(
    config: Dict[str, Any],
    schema_fields: List[SchemaField],
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for field in schema_fields:
        value = _get_nested_value(config, field.full_key)
        if value is _MISSING:
            continue
        values[field.full_key] = _render_value(value, field.field_type)
    return values


def _build_render_values_from_params(
    params: Dict[str, List[str]],
    schema_fields: List[SchemaField],
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for field in schema_fields:
        if field.field_type == "boolean":
            values[field.full_key] = field.full_key in params
        else:
            values[field.full_key] = params.get(field.full_key, [""])[0]
    return values


def _parse_training_form(
    params: Dict[str, List[str]],
    schema_fields: List[SchemaField],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    config: Dict[str, Any] = {}
    missing: List[str] = []
    errors: List[str] = []
    for field in schema_fields:
        if field.field_type == "boolean":
            value = field.full_key in params
            _set_nested_value(config, field.full_key, value)
            continue
        raw = params.get(field.full_key, [""])[0].strip()
        if raw == "":
            missing.append(field.full_key)
            continue
        try:
            if field.field_type == "integer":
                value = int(raw)
            elif field.field_type == "number":
                value = float(raw)
            elif field.field_type == "string":
                value = raw
            else:
                try:
                    import yaml  # type: ignore[import]
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("PyYAML is required to parse list/dict fields") from exc
                value = yaml.safe_load(raw)
                if field.field_type == "list" and not isinstance(value, list):
                    raise ValueError("must be a YAML list")
                if field.field_type == "dict" and not isinstance(value, dict):
                    raise ValueError("must be a YAML mapping")
            _set_nested_value(config, field.full_key, value)
        except ValueError as exc:
            errors.append(f"{field.full_key}: {exc}")
    return config, missing, errors


def _write_training_config(path: Path, config: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to write configuration files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _render_status(message: Optional[str], level: str, missing: List[str], errors: List[str]) -> str:
    sections: List[str] = []
    if message:
        sections.append(f"<div class=\"status {html.escape(level)}\">{_escape_text(message)}</div>")
    if missing:
        missing_items = "".join(f"<li>{_escape_text(item)}</li>" for item in missing)
        sections.append(
            f"<div class=\"status warning\">Missing required fields:<ul>{missing_items}</ul></div>"
        )
    if errors:
        error_items = "".join(f"<li>{_escape_text(item)}</li>" for item in errors)
        sections.append(
            f"<div class=\"status error\">Validation errors:<ul>{error_items}</ul></div>"
        )
    return "".join(sections)


def _render_config_fields(
    values: Dict[str, Any],
    missing: List[str],
    schema_fields: List[SchemaField],
) -> str:
    hint_map = _get_hint_map()
    fields_html: List[str] = []
    current_section = None
    current_subsection = None
    current_subsub = None
    for field in schema_fields:
        if field.section != current_section:
            current_section = field.section
            current_subsection = None
            current_subsub = None
            fields_html.append(
                f"<div class=\"section-title\">{_escape_text(current_section)}</div>"
            )
        parts = field.dotted_key.split(".") if field.dotted_key else []
        subsection = parts[0] if len(parts) >= 1 else None
        subsub = parts[1] if len(parts) >= 2 else None
        if subsection and subsection != current_subsection:
            current_subsection = subsection
            current_subsub = None
            fields_html.append(
                f"<div class=\"subsection-title\">{_escape_text(subsection)}</div>"
            )
        if subsub and subsub != current_subsub:
            current_subsub = subsub
            fields_html.append(
                f"<div class=\"subsub-title\">{_escape_text(subsub)}</div>"
            )
        label = parts[-1] if parts else field.section
        hint = hint_map.get(field.full_key) or f"Required. Type: {field.field_type}."
        is_missing = field.full_key in missing
        field_class = "field missing" if is_missing else "field"
        value = values.get(field.full_key, "")

        if field.field_type == "boolean":
            checked = "checked" if value else ""
            input_html = (
                f"<input type=\"checkbox\" name=\"{_escape_text(field.full_key)}\" value=\"true\" {checked} />"
            )
        elif field.field_type in {"list", "dict", "any"}:
            input_html = (
                f"<textarea name=\"{_escape_text(field.full_key)}\" rows=\"3\">"
                f"{_escape_text(value)}</textarea>"
            )
        elif field.field_type == "string" and field.full_key in _SELECT_OPTIONS:
            options = list(_SELECT_OPTIONS[field.full_key])
            if value and str(value) not in options:
                options = [str(value)] + options
            option_html = ""
            if not value:
                option_html += "<option value=\"\">Select...</option>"
            for option in options:
                selected = "selected" if str(value) == option else ""
                option_html += (
                    f"<option value=\"{_escape_text(option)}\" {selected}>{_escape_text(option)}</option>"
                )
            input_html = (
                f"<select name=\"{_escape_text(field.full_key)}\">{option_html}</select>"
            )
        else:
            step = "1" if field.field_type == "integer" else "any"
            input_type = "number" if field.field_type in {"integer", "number"} else "text"
            input_html = (
                f"<input type=\"{input_type}\" name=\"{_escape_text(field.full_key)}\" "
                f"step=\"{step}\" value=\"{_escape_text(value)}\" />"
            )

        fields_html.append(
            """
            <div class="{field_class}">
              <label>{label} <span class="hint" data-hint="{hint}">i</span></label>
              {input_html}
            </div>
            """.format(
                field_class=field_class,
                label=_escape_text(label),
                hint=_escape_text(hint),
                input_html=input_html,
            )
        )
    return "".join(fields_html)


def _render_training_config_panel(
    *,
    browse_path: str,
    selected_path: str,
    values: Dict[str, Any],
    missing: List[str],
    errors: List[str],
    config_mode: str,
    schema_fields: List[SchemaField],
    message: Optional[str] = None,
    message_level: str = "info",
) -> str:
    rel_path, parent_rel, dirs, files = _list_config_entries(browse_path)
    selected_display = selected_path or "None"

    entries: List[str] = []
    if parent_rel is not None:
        entries.append(
            f"<button type=\"submit\" name=\"browse_to\" value=\"{_escape_text(parent_rel)}\" "
            f"class=\"file-entry dir\">..</button>"
        )
    for dir_name in dirs:
        dir_rel = _join_rel(rel_path, dir_name)
        entries.append(
            f"<button type=\"submit\" name=\"browse_to\" value=\"{_escape_text(dir_rel)}\" "
            f"class=\"file-entry dir\">{_escape_text(dir_name)}/</button>"
        )
    for file_name in files:
        file_rel = _join_rel(rel_path, file_name)
        file_path = _normalize_selected_path(_join_rel("config", file_rel))
        selected_class = "selected" if file_path == _normalize_selected_path(selected_path) else ""
        entries.append(
            f"<button type=\"submit\" name=\"select_path\" value=\"{_escape_text(file_path)}\" "
            f"class=\"file-entry file {selected_class}\">{_escape_text(file_name)}</button>"
        )

    status_html = _render_status(message, message_level, missing, errors)
    fields_html = _render_config_fields(values, missing, schema_fields)
    browse_display = rel_path or "config"
    mode_label = "Simple" if config_mode == "simple" else "Extended"
    other_mode = "extended" if config_mode == "simple" else "simple"
    other_label = "Extended" if config_mode == "simple" else "Simple"

    return f"""
    <div id=\"config-panel\" class=\"panel config-panel\">
      <div class=\"panel-header\">
        <h3>Configuration</h3>
        <div class=\"mode-toggle\">
          <span class=\"muted\">Mode: {mode_label}</span>
          <form hx-post=\"/ui/config/mode\" hx-target=\"#config-panel\" hx-swap=\"outerHTML\">
            <input type=\"hidden\" name=\"browse_path\" value=\"{_escape_text(rel_path)}\" />
            <input type=\"hidden\" name=\"selected_path\" value=\"{_escape_text(selected_path)}\" />
            <input type=\"hidden\" name=\"mode\" value=\"{other_mode}\" />
            <button type=\"submit\" class=\"ghost\">Switch to {other_label}</button>
          </form>
        </div>
      </div>
      <p class=\"muted\">Load a training config, edit fields, and save. All fields are required.</p>
      {status_html}
      <div class=\"config-browser\">
        <div class=\"browser-header\">
          <div><strong>Browse:</strong> {_escape_text(browse_display)}</div>
          <div><strong>Selected:</strong> {_escape_text(selected_display)}</div>
        </div>
        <form hx-post=\"/ui/config/browse\" hx-target=\"#config-panel\" hx-swap=\"outerHTML\">
          <input type=\"hidden\" name=\"browse_path\" value=\"{_escape_text(rel_path)}\" />
          <input type=\"hidden\" name=\"selected_path\" value=\"{_escape_text(selected_path)}\" />
          <div class=\"file-list\">{''.join(entries)}</div>
        </form>
        <form hx-post=\"/ui/config/load\" hx-target=\"#config-panel\" hx-swap=\"outerHTML\" class=\"load-form\">
          <input type=\"hidden\" name=\"browse_path\" value=\"{_escape_text(rel_path)}\" />
          <input type=\"hidden\" name=\"config_path\" value=\"{_escape_text(selected_path)}\" />
          <button type=\"submit\" class=\"primary\">Load</button>
        </form>
      </div>
      <form hx-post=\"/ui/config/save\" hx-target=\"#config-panel\" hx-swap=\"outerHTML\" class=\"config-form\">
        <input type=\"hidden\" name=\"browse_path\" value=\"{_escape_text(rel_path)}\" />
        <input type=\"hidden\" name=\"config_path\" value=\"{_escape_text(selected_path)}\" />
        <button type=\"submit\" class=\"primary\">Save</button>
        <div class=\"config-fields\">{fields_html}</div>
        <button type=\"submit\" class=\"primary\">Save</button>
      </form>
    </div>
    """


def _render_run_control_panel(config: ServerConfig, selected_path: Optional[str]) -> str:
    allow_run_control = config.allow_run_control
    allowed_configs = set(_allowed_configs(config.allowed_configs_glob))
    selected_normalized = _normalize_selected_path(selected_path or "")
    selected_display = selected_normalized or "No config selected"
    allowlisted = bool(selected_normalized and selected_normalized in allowed_configs)

    status_label = "Disabled" if not allow_run_control else "Enabled"
    status_class = "chip muted" if not allow_run_control else "chip ok"
    hint_text = (
        "Run control is disabled. Set allow_run_control=true in observability config."
        if not allow_run_control
        else "Start/stop uses the selected config from the Configuration panel."
    )
    warning_html = ""
    if selected_normalized and not allowlisted:
        warning_html = (
            "<div class=\"status warning\">Selected config is not in the allowlist.</div>"
        )

    start_disabled = "disabled" if (not allow_run_control or not allowlisted) else ""
    stop_disabled = "disabled" if not allow_run_control else ""

    return f"""
    <div class=\"panel\">
      <div class=\"panel-header\">
        <h3>Run Control</h3>
        <span class=\"{status_class}\">{status_label}</span>
      </div>
      <p class=\"muted\">{_escape_text(hint_text)}</p>
      <div class=\"muted\"><strong>Selected:</strong> {_escape_text(selected_display)}</div>
      {warning_html}
      <div class=\"run-actions\">
        <form hx-post=\"/ui/start\" hx-target=\"#run-status\">
          <input type=\"hidden\" name=\"config\" value=\"{_escape_text(selected_normalized)}\" />
          <button type=\"submit\" class=\"primary\" {start_disabled}>Start Run</button>
        </form>
        <form hx-post=\"/ui/stop\" hx-target=\"#run-status\" hx-confirm=\"Stop the running job?\">
          <input type=\"hidden\" name=\"confirm\" value=\"yes\" />
          <button type=\"submit\" class=\"danger\" {stop_disabled}>Stop Run</button>
        </form>
      </div>
    </div>
    """


def _render_ui_page(_config: ServerConfig) -> str:
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Observability Dashboard</title>
    <script src=\"/static/htmx.min.js\"></script>
    <style>
      :root {{
        --bg: #f4efe6;
        --panel: #fffaf1;
        --ink: #262626;
        --muted: #6d6a61;
        --border: #e1d7c7;
        --accent: #1f6f8b;
        --accent-2: #e1a95f;
        --danger: #b23a48;
        --shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
      }}
      body {{
        font-family: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
        margin: 0;
        padding: 24px;
        background: radial-gradient(circle at top left, #fdf7ec 0%, #f4efe6 45%, #efe7d8 100%);
        color: var(--ink);
      }}
      h2 {{ margin: 0 0 16px 0; letter-spacing: 0.5px; }}
      h3 {{ margin: 0; }}
      .layout {{ max-width: 1200px; margin: 0 auto; }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        box-shadow: var(--shadow);
      }}
      .panel + .panel {{ margin-top: 16px; }}
      .panel-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
      .mode-toggle {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
      .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
      .stats-grid {{ margin-top: 16px; }}
      .muted {{ color: var(--muted); font-size: 0.9rem; }}
      .chip {{
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: #e9dfcf;
        color: var(--muted);
      }}
      .chip.ok {{ background: #d5e6e0; color: #24544a; }}
      .chip.muted {{ background: #efe7d8; color: var(--muted); }}
      .config-panel {{ margin-top: 16px; }}
      .config-panel .field label,
      .config-panel input,
      .config-panel select,
      .config-panel textarea,
      .config-panel .file-entry,
      .config-panel .browser-header {{
        font-size: 0.7rem;
        line-height: 1.2;
      }}
      .config-browser {{ margin-top: 12px; border: 1px dashed var(--border); border-radius: 10px; padding: 12px; background: #fffdf7; }}
      .browser-header {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 0.9rem; }}
      .file-list {{ max-height: 220px; overflow: auto; display: grid; gap: 6px; padding: 8px; margin-top: 10px; border: 1px solid var(--border); border-radius: 8px; background: #fff; }}
      .file-entry {{ text-align: left; padding: 6px 8px; border-radius: 6px; border: 1px solid transparent; background: #f7f2e9; cursor: pointer; }}
      .file-entry.dir {{ font-weight: 600; color: var(--ink); }}
      .file-entry.file.selected {{ border-color: var(--accent); background: #e7f1f5; }}
      .load-form {{ margin-top: 10px; }}
      .config-form {{ margin-top: 16px; display: grid; gap: 12px; }}
      .config-fields {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
      .section-title {{ grid-column: 1 / -1; font-weight: 700; margin-top: 12px; }}
      .subsection-title {{ grid-column: 1 / -1; font-weight: 600; margin-top: 8px; color: var(--muted); }}
      .subsub-title {{ grid-column: 1 / -1; font-weight: 600; margin-top: 6px; color: #8b846f; font-size: 0.75rem; }}
      .field {{ display: grid; gap: 6px; }}
      .field.missing input, .field.missing textarea, .field.missing select {{ border-color: #b23a48; }}
      .status {{ margin: 8px 0; padding: 8px 10px; border-radius: 6px; font-size: 0.85rem; }}
      .status.info {{ background: #ede4d5; color: var(--muted); }}
      .status.ok {{ background: #d5e6e0; color: #24544a; }}
      .status.error {{ background: #f8d7da; color: #7a2832; }}
      .status.warning {{ background: #fff3cd; color: #7a5a00; }}
      .status ul {{ margin: 6px 0 0 18px; }}
      progress {{ width: 100%; height: 12px; }}
      form {{ display: grid; gap: 10px; margin-top: 10px; }}
      .run-actions {{ display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }}
      label {{ font-weight: 600; }}
      input, select {{
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: #fff;
        font-size: 0.95rem;
      }}
      input[type="checkbox"] {{
        width: 12px;
        height: 12px;
        margin: 0;
        vertical-align: middle;
      }}
      button {{
        padding: 8px 12px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-weight: 600;
      }}
      button.ghost {{
        background: transparent;
        border: 1px solid var(--border);
        color: var(--muted);
      }}
      button.primary {{ background: var(--accent); color: #fff; }}
      button.danger {{ background: var(--danger); color: #fff; }}
      button[disabled], select[disabled] {{ opacity: 0.55; cursor: not-allowed; }}
      .hint {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        margin-left: 6px;
        border-radius: 50%;
        background: #e8dfd0;
        color: var(--muted);
        font-size: 0.7rem;
        position: relative;
        cursor: help;
      }}
      .hint::after {{
        content: attr(data-hint);
        position: absolute;
        bottom: 150%;
        left: 50%;
        transform: translateX(-50%);
        background: #222;
        color: #f8f6f0;
        padding: 6px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
      }}
      .hint:hover::after, .hint:focus::after {{ opacity: 1; }}
      .logs pre {{
        background: #0d0f12;
        color: #9be564;
        padding: 10px;
        border-radius: 8px;
        height: 320px;
        overflow: auto;
        font-family: "Courier New", monospace;
      }}
      @media (max-width: 720px) {{
        body {{ padding: 16px; }}
        .panel {{ padding: 14px; }}
      }}
    </style>
  </head>
  <body>
    <div class=\"layout\">
      <h2>Run Dashboard</h2>
      <div id=\"run-status\" class=\"panel\" hx-get=\"/ui/status\" hx-trigger=\"load, every 5s\"></div>
      <div id=\"config-panel\" class=\"panel config-panel\" hx-get=\"/ui/config\" hx-trigger=\"load\"></div>
      <div id=\"run-control\" class=\"panel\" hx-get=\"/ui/run-control\" hx-trigger=\"load, every 5s\" style=\"margin-top: 16px;\"></div>
      <div class=\"grid stats-grid\">
        <div class=\"panel\" hx-get=\"/ui/progress\" hx-trigger=\"load, every 5s\"></div>
        <div class=\"panel\" hx-get=\"/ui/metrics\" hx-trigger=\"load, every 5s\"></div>
        <div class=\"panel\" hx-get=\"/ui/stats\" hx-trigger=\"load, every 5s\"></div>
      </div>
      <div class=\"panel logs\" style=\"margin-top: 16px;\">
        <div class=\"panel-header\">
          <h3>Logs</h3>
        </div>
        <div id=\"logs-panel\" hx-get=\"/ui/logs\" hx-trigger=\"load, every 2s\"
             hx-on::afterSwap=\"const el = this.querySelector('pre'); if (el) {{ el.scrollTop = el.scrollHeight; }}\"></div>
      </div>
    </div>
  </body>
</html>
"""


class ObservabilityHandler(BaseHTTPRequestHandler):
    server_state: ServerState

    def _unauthorized(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", "Basic realm=\"Observability\"")
        self.end_headers()

    def _send_html(self, body: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _require_auth(self) -> bool:
        config = self.server_state.config
        if _check_auth(self.headers, config.user, config.password):
            return True
        self._unauthorized()
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_auth():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/ui":
            cfg = self.server_state.config
            body = _render_ui_page(cfg)
            self._send_html(body)
            return

        if path == "/ui/config":
            selected_path = self.server_state.get_selected_config_path() or ""
            mode = self.server_state.get_config_mode()
            schema_fields = _get_schema_fields_for_mode(mode)
            message = None if selected_path else "No config loaded yet. Select a file and click Load."
            body = _render_training_config_panel(
                browse_path="",
                selected_path=selected_path,
                values={},
                missing=[],
                errors=[],
                config_mode=mode,
                schema_fields=schema_fields,
                message=message,
                message_level="info",
            )
            self._send_html(body)
            return

        if path == "/ui/run-control":
            selected_path = self.server_state.get_selected_config_path()
            body = _render_run_control_panel(self.server_state.config, selected_path)
            self._send_html(body)
            return

        if path == "/ui/status":
            state = load_run_state(self.server_state.config.run_state_path)
            if not state:
                self._send_html("<strong>No active run.</strong>")
                return
            body = (
                f"<strong>Status:</strong> {state.get('status')}<br/>"
                f"<strong>Stage:</strong> {state.get('stage')}<br/>"
                f"<strong>Run ID:</strong> {state.get('run_id', '-')}")
            self._send_html(body)
            return

        if path == "/ui/progress":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            progress = float(state.get("progress", 0.0)) * 100.0
            eta = state.get("eta_seconds")
            eta_display = f"{int(eta)}s" if eta is not None else "n/a"
            body = f"""
            <h3>Progress</h3>
            <div>Progress: {progress:.2f}%</div>
            <div>ETA: {eta_display}</div>
            <progress value=\"{progress}\" max=\"100\"></progress>
            """
            self._send_html(body)
            return

        if path == "/ui/metrics":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            def _fmt(value: Any) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{float(value):.3f}"
                except (TypeError, ValueError):
                    return "n/a"

            body = "<h3>Duty Cycle</h3>"
            body += f"<div>min: {_fmt(state.get('duty_cycle_min'))}</div>"
            body += f"<div>median: {_fmt(state.get('duty_cycle_median'))}</div>"
            body += f"<div>p95: {_fmt(state.get('duty_cycle_p95'))}</div>"
            self._send_html(body)
            return

        if path == "/ui/stats":
            state = load_run_state(self.server_state.config.run_state_path) or {}

            def _fmt_time(ts: Any) -> str:
                if ts is None:
                    return "n/a"
                try:
                    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError):
                    return "n/a"

            def _fmt_duration(seconds: Any) -> str:
                if seconds is None:
                    return "n/a"
                try:
                    total = int(float(seconds))
                except (TypeError, ValueError):
                    return "n/a"
                hours = total // 3600
                minutes = (total % 3600) // 60
                secs = total % 60
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            start_ts = state.get("start_time")
            updated_ts = state.get("updated_time")
            runtime = None
            if start_ts is not None:
                try:
                    runtime = float(datetime.now().timestamp()) - float(start_ts)
                except (TypeError, ValueError):
                    runtime = None

            body = "<h3>Run Stats</h3>"
            body += f"<div>Status: {state.get('status', 'idle')}</div>"
            body += f"<div>Stage: {state.get('stage', 'idle')}</div>"
            body += f"<div>Start: {_fmt_time(start_ts)}</div>"
            body += f"<div>Last update: {_fmt_time(updated_ts)}</div>"
            body += f"<div>Runtime: {_fmt_duration(runtime)}</div>"
            body += (
                f"<div>Snapshot chunks: {state.get('snapshot_chunks_processed', 0)} / "
                f"{state.get('snapshot_chunks_total', 0)}</div>"
            )
            body += (
                f"<div>Training epochs: {state.get('training_epochs_done', 0)} / "
                f"{state.get('training_epochs_total', 0)}</div>"
            )
            body += (
                f"<div>Training batches: {state.get('training_batches_done', 0)} / "
                f"{state.get('training_batches_total', 0)}</div>"
            )
            body += (
                f"<div>Eval batches: {state.get('eval_batches_done', 0)} / "
                f"{state.get('eval_batches_total', 0)}</div>"
            )
            if state.get("last_error"):
                body += f"<div>Last error: {_escape_text(state.get('last_error'))}</div>"
            self._send_html(body)
            return

        if path == "/ui/logs":
            cfg = self.server_state.config
            lines = _tail_log(cfg.run_log_path, max_lines=cfg.tail_max_lines)
            body = "<pre>" + "\n".join(lines) + "</pre>"
            self._send_html(body)
            return

        if path == "/metrics":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            registry = CollectorRegistry()
            Gauge("run_progress", "Run progress", registry=registry).set(float(state.get("progress", 0.0)))
            Gauge("run_eta_seconds", "Run ETA seconds", registry=registry).set(float(state.get("eta_seconds") or 0.0))
            Gauge("snapshot_chunks_processed", "Snapshot chunks processed", registry=registry).set(
                float(state.get("snapshot_chunks_processed", 0))
            )
            Gauge("snapshot_chunks_total", "Snapshot chunks total", registry=registry).set(
                float(state.get("snapshot_chunks_total", 0))
            )
            Gauge("training_batches_processed", "Training batches processed", registry=registry).set(
                float(state.get("training_batches_done", 0))
            )
            Gauge("training_batches_total", "Training batches total", registry=registry).set(
                float(state.get("training_batches_total", 0))
            )
            Gauge("eval_batches_processed", "Eval batches processed", registry=registry).set(
                float(state.get("eval_batches_done", 0))
            )
            Gauge("eval_batches_total", "Eval batches total", registry=registry).set(
                float(state.get("eval_batches_total", 0))
            )
            if state.get("duty_cycle_min") is not None:
                Gauge("duty_cycle_min", "Duty cycle min", registry=registry).set(
                    float(state["duty_cycle_min"])
                )
                Gauge("duty_cycle_median", "Duty cycle median", registry=registry).set(
                    float(state.get("duty_cycle_median", 0.0))
                )
                Gauge("duty_cycle_p95", "Duty cycle p95", registry=registry).set(
                    float(state.get("duty_cycle_p95", 0.0))
                )

            output = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
            return

        if path == "/api/run":
            state = load_run_state(self.server_state.config.run_state_path) or {"status": "idle"}
            self._send_json(state)
            return

        if path == "/api/logs":
            cfg = self.server_state.config
            lines = _tail_log(cfg.run_log_path, max_lines=cfg.tail_max_lines)
            self._send_json({"lines": lines})
            return

        if path.startswith("/static/"):
            cfg = self.server_state.config
            static_path = _safe_static_file_path(static_dir=cfg.static_dir, request_path=path)
            if static_path is None or not static_path.exists() or not static_path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            content = static_path.read_bytes()
            self.send_response(200)
            mime_type, _encoding = mimetypes.guess_type(str(static_path))
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return

        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        params = parse_qs(body.decode("utf-8"))

        if path == "/ui/config/browse":
            browse_path = params.get("browse_path", [""])[0]
            selected_path = params.get("selected_path", [""])[0]
            browse_to = params.get("browse_to", [""])[0]
            select_path = params.get("select_path", [""])[0]
            if browse_to:
                browse_path = browse_to
            if select_path:
                selected_path = select_path
                self.server_state.set_selected_config_path(selected_path)
            mode = self.server_state.get_config_mode()
            schema_fields = _get_schema_fields_for_mode(mode)
            body_html = _render_training_config_panel(
                browse_path=browse_path,
                selected_path=selected_path,
                values={},
                missing=[],
                errors=[],
                config_mode=mode,
                schema_fields=schema_fields,
                message=None,
                message_level="info",
            )
            self._send_html(body_html)
            return

        if path == "/ui/config/load":
            browse_path = params.get("browse_path", [""])[0]
            selected_path = params.get("config_path", [""])[0]
            mode = self.server_state.get_config_mode()
            schema_fields = _get_schema_fields_for_mode(mode)
            if not selected_path:
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path="",
                    values={},
                    missing=[],
                    errors=[],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Select a config file before loading.",
                    message_level="error",
                )
                self._send_html(body_html)
                return
            resolved = _resolve_selected_path(selected_path)
            if resolved is None or not resolved.exists():
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=selected_path,
                    values={},
                    missing=[],
                    errors=[],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Selected config file was not found.",
                    message_level="error",
                )
                self._send_html(body_html)
                return
            try:
                config_data = _load_training_config(resolved)
                merged_config = config_data
                if mode == "simple":
                    defaults = _load_default_simple_config()
                    merged_config = _deep_merge(defaults, config_data)
                values = _build_render_values_from_config(merged_config, schema_fields)
                missing = _missing_fields(merged_config, schema_fields)
                for field_key in missing:
                    values[field_key] = ""
                message = "Config loaded."
                level = "ok"
                if missing:
                    message = "Config loaded with missing required fields."
                    level = "warning"
                self.server_state.set_selected_config_path(_normalize_selected_path(selected_path))
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=_normalize_selected_path(selected_path),
                    values=values,
                    missing=missing,
                    errors=[],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message=message,
                    message_level=level,
                )
            except Exception as exc:  # noqa: BLE001
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=selected_path,
                    values={},
                    missing=[],
                    errors=[str(exc)],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Failed to load config file.",
                    message_level="error",
                )
            self._send_html(body_html)
            return

        if path == "/ui/config/save":
            browse_path = params.get("browse_path", [""])[0]
            selected_path = params.get("config_path", [""])[0]
            mode = self.server_state.get_config_mode()
            schema_fields = _get_schema_fields_for_mode(mode)
            if not selected_path:
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path="",
                    values=_build_render_values_from_params(params, schema_fields),
                    missing=[],
                    errors=[],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Select a config file before saving.",
                    message_level="error",
                )
                self._send_html(body_html)
                return
            config_data, missing, errors = _parse_training_form(params, schema_fields)
            values = _build_render_values_from_params(params, schema_fields)
            if missing or errors:
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=_normalize_selected_path(selected_path),
                    values=values,
                    missing=missing,
                    errors=errors,
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Cannot save until all required fields are filled.",
                    message_level="error",
                )
                self._send_html(body_html)
                return
            resolved = _resolve_selected_path(selected_path)
            if resolved is None:
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=selected_path,
                    values=values,
                    missing=[],
                    errors=["Invalid config path."],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Failed to save config.",
                    message_level="error",
                )
                self._send_html(body_html)
                return
            try:
                save_config = config_data
                if mode == "simple":
                    defaults = _load_default_simple_config()
                    save_config = _deep_merge(defaults, config_data)
                full_missing = _missing_fields(save_config, _get_schema_fields())
                if full_missing:
                    body_html = _render_training_config_panel(
                        browse_path=browse_path,
                        selected_path=_normalize_selected_path(selected_path),
                        values=values,
                        missing=missing,
                        errors=["Missing required fields in defaults: " + ", ".join(full_missing)],
                        config_mode=mode,
                        schema_fields=schema_fields,
                        message="Cannot save: defaults missing required fields.",
                        message_level="error",
                    )
                    self._send_html(body_html)
                    return
                _write_training_config(resolved, save_config)
                self.server_state.set_selected_config_path(_normalize_selected_path(selected_path))
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=_normalize_selected_path(selected_path),
                    values=_build_render_values_from_config(save_config, schema_fields),
                    missing=[],
                    errors=[],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message=f"Saved to {_normalize_selected_path(selected_path)}.",
                    message_level="ok",
                )
            except Exception as exc:  # noqa: BLE001
                body_html = _render_training_config_panel(
                    browse_path=browse_path,
                    selected_path=selected_path,
                    values=values,
                    missing=[],
                    errors=[str(exc)],
                    config_mode=mode,
                    schema_fields=schema_fields,
                    message="Failed to save config.",
                    message_level="error",
                )
            self._send_html(body_html)
            return

        if path == "/ui/config/mode":
            browse_path = params.get("browse_path", [""])[0]
            selected_path = params.get("selected_path", [""])[0]
            mode = params.get("mode", ["extended"])[0]
            self.server_state.set_config_mode(mode)
            mode = self.server_state.get_config_mode()
            schema_fields = _get_schema_fields_for_mode(mode)
            values: Dict[str, Any] = {}
            missing: List[str] = []
            errors: List[str] = []
            message = f"Switched to {mode} mode."
            level = "info"
            if selected_path:
                resolved = _resolve_selected_path(selected_path)
                if resolved is None or not resolved.exists():
                    errors = ["Selected config file was not found."]
                    level = "error"
                else:
                    try:
                        config_data = _load_training_config(resolved)
                        merged_config = config_data
                        if mode == "simple":
                            defaults = _load_default_simple_config()
                            merged_config = _deep_merge(defaults, config_data)
                        values = _build_render_values_from_config(merged_config, schema_fields)
                        missing = _missing_fields(merged_config, schema_fields)
                        for field_key in missing:
                            values[field_key] = ""
                    except Exception as exc:  # noqa: BLE001
                        errors = [str(exc)]
                        level = "error"
            body_html = _render_training_config_panel(
                browse_path=browse_path,
                selected_path=selected_path,
                values=values,
                missing=missing,
                errors=errors,
                config_mode=mode,
                schema_fields=schema_fields,
                message=message,
                message_level=level,
            )
            self._send_html(body_html)
            return

        if path in {"/api/runs", "/ui/start"}:
            config_path = params.get("config", [None])[0] or self.server_state.get_selected_config_path()
            if not config_path:
                self._send_json({"error": "config is required"}, status=400)
                return
            config_path = _normalize_selected_path(config_path)
            cfg = self.server_state.config
            if config_path not in _allowed_configs(cfg.allowed_configs_glob):
                self._send_json({"error": "config is not in allow-list"}, status=400)
                return
            self.server_state.set_selected_config_path(config_path)
            ok, message = self.server_state.start_run(config_path)
            if path == "/ui/start":
                status = 200 if ok else 409
                self._send_html(f"<strong>{message}</strong>", status=status)
            else:
                status = 200 if ok else 409
                self._send_json({"message": message}, status=status)
            return

        if path in {"/api/runs/stop", "/ui/stop"}:
            confirm = params.get("confirm", [""])[0]
            if confirm.lower() != "yes":
                self._send_json({"error": "confirm=yes required"}, status=400)
                return
            ok, message = self.server_state.stop_run()
            if path == "/ui/stop":
                status = 200 if ok else 409
                self._send_html(f"<strong>{message}</strong>", status=status)
            else:
                status = 200 if ok else 409
                self._send_json({"message": message}, status=status)
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Observability server")
    parser.add_argument(
        "--config",
        default="config/observability.yaml",
        help="Path to observability YAML config (non-secrets)",
    )
    args = parser.parse_args()

    config = ServerConfig.from_sources(config_path=str(args.config) if args.config else None)
    server_state = ServerState(config)

    handler = ObservabilityHandler
    handler.server_state = server_state
    server = ThreadingHTTPServer((config.host, config.port), handler)
    logger.info("Observability server listening on %s:%s", config.host, config.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
