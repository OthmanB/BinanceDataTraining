"""Minimal observability server with HTMX dashboard and Prometheus metrics.

Configuration:
- Non-secret settings (host/port/paths) can come from `config/observability.yaml`.
- Secrets (Basic Auth user/password) must come from environment variables.

Environment variable overrides (take precedence over YAML):
- OBSERVABILITY_HOST, OBSERVABILITY_PORT
- RUN_STATE_PATH (sqlite URI), RUN_LOG_PATH
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
import time
from typing import Any, Dict, List, Optional, Tuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .run_state import RunStateWriter, _resolve_sqlite_path, load_run_state, load_run_history


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
        try:
            _resolve_sqlite_path(str(run_state_path))
        except ValueError as exc:
            raise RuntimeError(f"RUN_STATE_PATH must be a sqlite URI (sqlite:///...): {exc}") from exc

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
        self.system_metrics_lock = threading.Lock()
        self.system_metrics_cached_at = 0.0
        self.system_metrics_cache: Dict[str, Any] = {}
        self.system_metrics_ttl_seconds = 3.0

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

    def get_system_metrics(self) -> Dict[str, Any]:
        with self.system_metrics_lock:
            now = time.time()
            if self.system_metrics_cache and (now - self.system_metrics_cached_at) < self.system_metrics_ttl_seconds:
                return dict(self.system_metrics_cache)

            cpu_count = os.cpu_count() or 1
            load_1m: Optional[float] = None
            load_5m: Optional[float] = None
            load_15m: Optional[float] = None
            load_norm_1m: Optional[float] = None
            try:
                load_1m, load_5m, load_15m = os.getloadavg()
                load_norm_1m = float(load_1m) / float(cpu_count)
            except Exception:  # noqa: BLE001
                load_1m = None
                load_5m = None
                load_15m = None
                load_norm_1m = None

            mem_stats = _read_linux_memory_stats()

            run_pid_fallback: Optional[int] = None
            with self.process_lock:
                if self.process is not None and self.process.poll() is None:
                    run_pid_fallback = int(self.process.pid)

            state = load_run_state(self.config.run_state_path) or {}
            run_pid = _parse_positive_int(state.get("run_process_pid"))
            if run_pid is None:
                run_pid = run_pid_fallback

            run_pid_rss_bytes = _read_linux_process_rss_bytes(run_pid or -1)

            payload: Dict[str, Any] = {
                "cpu_count": int(cpu_count),
                "load_1m": load_1m,
                "load_5m": load_5m,
                "load_15m": load_15m,
                "load_norm_1m": load_norm_1m,
                "run_process_pid": run_pid,
                "run_process_rss_bytes": run_pid_rss_bytes,
                "collected_time": now,
                "gpus": _read_gpu_stats(),
            }
            payload.update(mem_stats)

            self.system_metrics_cached_at = now
            self.system_metrics_cache = dict(payload)
            return payload

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


def _read_linux_memory_stats() -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {
        "mem_total_bytes": None,
        "mem_available_bytes": None,
        "mem_used_bytes": None,
        "mem_used_ratio": None,
    }
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return stats

    values_kb: Dict[str, int] = {}
    try:
        for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                values_kb[key] = int(parts[0])
            except ValueError:
                continue
    except Exception:  # noqa: BLE001
        return stats

    total_kb = values_kb.get("MemTotal")
    available_kb = values_kb.get("MemAvailable")
    if total_kb is None or available_kb is None or total_kb <= 0:
        return stats

    total_bytes = float(total_kb) * 1024.0
    available_bytes = float(available_kb) * 1024.0
    used_bytes = max(total_bytes - available_bytes, 0.0)
    used_ratio = used_bytes / total_bytes if total_bytes > 0 else None

    stats["mem_total_bytes"] = total_bytes
    stats["mem_available_bytes"] = available_bytes
    stats["mem_used_bytes"] = used_bytes
    stats["mem_used_ratio"] = used_ratio
    return stats


def _read_linux_process_rss_bytes(pid: int) -> Optional[float]:
    if pid <= 0:
        return None
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("VmRSS:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            return float(int(parts[1]) * 1024)
    except Exception:  # noqa: BLE001
        return None
    return None


def _read_gpu_stats() -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:  # noqa: BLE001
        return stats

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue

        gpu: Dict[str, Any] = {
            "index": None,
            "name": None,
            "utilization_ratio": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "memory_used_ratio": None,
            "power_draw_watts": None,
            "power_limit_watts": None,
        }

        try:
            gpu["index"] = float(parts[0])
        except ValueError:
            continue
        gpu["name"] = parts[1]

        try:
            gpu["utilization_ratio"] = float(parts[2]) / 100.0
        except ValueError:
            gpu["utilization_ratio"] = None
        try:
            used_bytes = float(parts[3]) * 1024.0 * 1024.0
            gpu["memory_used_bytes"] = used_bytes
        except ValueError:
            used_bytes = None
        try:
            total_bytes = float(parts[4]) * 1024.0 * 1024.0
            gpu["memory_total_bytes"] = total_bytes
        except ValueError:
            total_bytes = None
        if used_bytes is not None and total_bytes is not None and total_bytes > 0:
            gpu["memory_used_ratio"] = used_bytes / total_bytes

        try:
            gpu["power_draw_watts"] = float(parts[5])
        except ValueError:
            gpu["power_draw_watts"] = None
        try:
            gpu["power_limit_watts"] = float(parts[6])
        except ValueError:
            gpu["power_limit_watts"] = None

        stats.append(gpu)

    return stats


def _parse_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_sqlite_uri(path: Any) -> Optional[str]:
    if path is None:
        return None
    candidate = str(path).strip()
    if not candidate:
        return None
    try:
        resolved = _resolve_sqlite_path(candidate)
    except ValueError:
        return None
    return f"sqlite:///{resolved.as_posix()}"


def _normalize_file_path(path: Any) -> Optional[str]:
    if path is None:
        return None
    candidate = str(path).strip()
    if not candidate:
        return None
    return str(Path(candidate).expanduser().resolve())


def _run_state_path_warnings(state: Dict[str, Any], config: ServerConfig) -> List[str]:
    warnings: List[str] = []
    writer_state_path = _normalize_sqlite_uri(state.get("run_state_path"))
    server_state_path = _normalize_sqlite_uri(config.run_state_path)
    if writer_state_path and server_state_path and writer_state_path != server_state_path:
        warnings.append(
            "Run-state path mismatch (server vs writer). "
            f"server={server_state_path} writer={writer_state_path}"
        )

    writer_log_path = _normalize_file_path(state.get("run_log_path"))
    server_log_path = _normalize_file_path(config.run_log_path)
    if writer_log_path and server_log_path and writer_log_path != server_log_path:
        warnings.append(
            "Run-log path mismatch (server vs writer). "
            f"server={server_log_path} writer={writer_log_path}"
        )
    return warnings


def _heartbeat_age_seconds(heartbeat_ts: Any) -> Optional[float]:
    if heartbeat_ts is None:
        return None
    try:
        return max(float(datetime.now().timestamp()) - float(heartbeat_ts), 0.0)
    except (TypeError, ValueError):
        return None


def _is_run_state_stale(state: Dict[str, Any], *, threshold_seconds: float = 30.0) -> bool:
    age = _heartbeat_age_seconds(state.get("heartbeat_time"))
    if age is None:
        return False
    return age > threshold_seconds


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




_PIPELINE_STAGES = [
    ("initializing", "Init"),
    ("snapshot_build", "Snapshot"),
    ("training", "Training"),
    ("evaluation", "Eval"),
]


def _render_pipeline_stepper(
    current_stage: str,
    status: str,
    stage_timestamps: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a horizontal pipeline stepper showing stage progression."""
    stage_ids = [s[0] for s in _PIPELINE_STAGES]
    ts = stage_timestamps or {}

    def _stage_duration(stage_id: str) -> str:
        start = ts.get(stage_id)
        if start is None:
            return ""
        idx = stage_ids.index(stage_id) if stage_id in stage_ids else -1
        end = None
        for next_idx in range(idx + 1, len(stage_ids)):
            end = ts.get(stage_ids[next_idx])
            if end is not None:
                break
        if end is None and status == "completed":
            end = ts.get("completed")
        if end is None and stage_id == current_stage:
            end = time.time()
        if end is None:
            return ""
        try:
            secs = max(float(end) - float(start), 0)
        except (TypeError, ValueError):
            return ""
        if secs < 60:
            return f" ({int(secs)}s)"
        minutes = int(secs) // 60
        remainder = int(secs) % 60
        return f" ({minutes}m{remainder:02d}s)"

    found = False
    parts: list = []
    for i, (stage_id, label) in enumerate(_PIPELINE_STAGES):
        if i > 0:
            parts.append('<span class="step-arrow">\u2192</span>')
        dur = _stage_duration(stage_id)
        if status == "completed":
            parts.append(f'<span class="step done">\u2713 {_escape_text(label)}{dur}</span>')
        elif status == "failed" and stage_id == current_stage:
            parts.append(f'<span class="step error">\u2717 {_escape_text(label)}{dur}</span>')
            found = True
        elif found or (stage_id != current_stage and current_stage in stage_ids and stage_ids.index(stage_id) > stage_ids.index(current_stage)):
            parts.append(f'<span class="step pending">\u25CB {_escape_text(label)}</span>')
        elif stage_id == current_stage:
            parts.append(f'<span class="step active">\u25CF {_escape_text(label)}{dur}</span>')
            found = True
        elif not found:
            parts.append(f'<span class="step done">\u2713 {_escape_text(label)}{dur}</span>')
        else:
            parts.append(f'<span class="step pending">\u25CB {_escape_text(label)}</span>')
    if status == "completed":
        parts.append('<span class="step-arrow">\u2192</span>')
        parts.append('<span class="step done">\u2713 Done</span>')
    if not found and status not in ("completed", "failed") and current_stage not in stage_ids:
        parts = []
        for i, (_, label) in enumerate(_PIPELINE_STAGES):
            if i > 0:
                parts.append('<span class="step-arrow">\u2192</span>')
            parts.append(f'<span class="step pending">\u25CB {_escape_text(label)}</span>')
    return '<div class="stepper">' + ''.join(parts) + '</div>'


def _render_ui_page(_config: ServerConfig) -> str:
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Observability Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x1F4CA;</text></svg>" />
    <script src="/static/htmx.min.js"></script>
    <script src="/static/chart.min.js"></script>
    <style>
      :root {{
        --bg: #f4f6f9; --bg-alt: #e8ecf1; --panel: #ffffff;
        --ink: #1a1a2e; --ink-2: #5a6072; --ink-3: #8b90a0;
        --border: #dce0e8;
        --accent: #3b82f6; --accent-hover: #2563eb; --accent-soft: #dbeafe;
        --danger: #ef4444; --danger-soft: #fce4e4;
        --success: #22c55e; --success-soft: #dcfce7;
        --warning: #f59e0b; --warning-soft: #fef3c7;
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
        --shadow-lg: 0 4px 12px rgba(0,0,0,0.08);
        --radius: 10px;
        --font: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        --mono: "SF Mono", "Cascadia Code", "JetBrains Mono", "Fira Code", ui-monospace, monospace;
      }}
      [data-theme="dark"] {{
        --bg: #0c0e14; --bg-alt: #151822; --panel: #1a1e2e;
        --ink: #e2e4eb; --ink-2: #9499ad; --ink-3: #5d6377;
        --border: #282d3e;
        --accent: #60a5fa; --accent-hover: #93c5fd; --accent-soft: #1e3a5f;
        --danger: #f87171; --danger-soft: #3b1c1c;
        --success: #4ade80; --success-soft: #14352a;
        --warning: #fbbf24; --warning-soft: #3b2f10;
        --shadow: 0 1px 3px rgba(0,0,0,0.25), 0 2px 8px rgba(0,0,0,0.2);
        --shadow-lg: 0 4px 12px rgba(0,0,0,0.35);
      }}
      *, *::before, *::after {{ box-sizing: border-box; }}
      body {{
        font-family: var(--font); margin: 0; padding: 0;
        background: var(--bg); color: var(--ink);
        font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased;
      }}
      .shell {{ max-width: 1320px; margin: 0 auto; padding: 16px 20px; }}
      /* Header */
      .header {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 0 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px;
      }}
      .header h1 {{ font-size: 1.25rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }}
      .header-controls {{ display: flex; align-items: center; gap: 12px; }}
      .theme-toggle {{
        background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px;
        padding: 6px 10px; cursor: pointer; font-size: 1rem; line-height: 1; color: var(--ink);
      }}
      .theme-toggle:hover {{ background: var(--border); }}
      /* Tabs */
      .tab-bar {{
        display: flex; gap: 2px; background: var(--bg-alt); border-radius: 10px;
        padding: 3px; margin-bottom: 16px; width: fit-content;
      }}
      .tab-btn {{
        padding: 7px 18px; border: none; border-radius: 8px; cursor: pointer;
        font-size: 0.85rem; font-weight: 600; color: var(--ink-2); background: transparent;
        transition: all 0.15s ease; font-family: var(--font);
      }}
      .tab-btn:hover {{ color: var(--ink); }}
      .tab-btn.active {{ background: var(--panel); color: var(--ink); box-shadow: var(--shadow); }}
      .tab-panel {{ display: none; }}
      .tab-panel.active {{ display: block; }}
      /* Cards */
      .card, .panel {{
        background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 16px; box-shadow: var(--shadow);
      }}
      .card + .card {{ margin-top: 16px; }}
      .card-header, .panel-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
      }}
      .card-header h3, .panel-header h3 {{ margin: 0; font-size: 0.95rem; font-weight: 700; letter-spacing: -0.01em; }}
      .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 16px; }}
      /* Badges */
      .badge {{
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
      }}
      .badge.idle {{ background: var(--bg-alt); color: var(--ink-3); }}
      .badge.running {{ background: var(--accent-soft); color: var(--accent); }}
      .badge.completed {{ background: var(--success-soft); color: var(--success); }}
      .badge.failed {{ background: var(--danger-soft); color: var(--danger); }}
      .badge.fresh {{ background: var(--success-soft); color: var(--success); }}
      .badge.stale {{ background: var(--warning-soft); color: var(--warning); }}
      .badge.unknown {{ background: var(--bg-alt); color: var(--ink-3); }}
      .badge.warn {{ background: var(--warning-soft); color: var(--warning); }}
      .badge.ok {{ background: var(--success-soft); color: var(--success); }}
      /* KV rows */
      .kv {{ display: flex; justify-content: space-between; padding: 5px 0; font-size: 0.85rem; }}
      .kv .k {{ color: var(--ink-2); }}
      .kv .v {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
      .kv + .kv {{ border-top: 1px solid var(--border); }}
      /* Pipeline stepper */
      .stepper {{ display: flex; align-items: center; gap: 0; margin: 12px 0; flex-wrap: wrap; }}
      .step {{
        display: flex; align-items: center; gap: 4px; padding: 5px 12px;
        font-size: 0.78rem; font-weight: 600; border-radius: 6px; white-space: nowrap;
      }}
      .step.done {{ color: var(--success); }}
      .step.active {{ background: var(--accent-soft); color: var(--accent); }}
      .step.pending {{ color: var(--ink-3); }}
      .step.error {{ background: var(--danger-soft); color: var(--danger); }}
      .step-arrow {{ color: var(--ink-3); font-size: 0.7rem; margin: 0 2px; }}
      /* Progress bar */
      .progress-wrap {{ margin: 8px 0; }}
      .progress-bar-outer {{
        width: 100%; height: 20px; background: var(--bg-alt); border-radius: 10px;
        overflow: hidden; position: relative;
      }}
      .progress-bar-inner {{
        height: 100%; border-radius: 10px; transition: width 0.4s ease;
        background: linear-gradient(90deg, var(--accent), #818cf8);
        position: relative; min-width: 0;
      }}
      .progress-bar-inner.training {{ background: linear-gradient(90deg, #22c55e, #4ade80); }}
      .progress-bar-inner.snapshot_build {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
      .progress-bar-inner.evaluation {{ background: linear-gradient(90deg, #a855f7, #c084fc); }}
      .progress-bar-inner.active {{
        background-image: linear-gradient(
          -45deg, rgba(255,255,255,0.15) 25%, transparent 25%,
          transparent 50%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.15) 75%, transparent 75%
        );
        background-size: 30px 30px; animation: barberpole 1s linear infinite;
      }}
      @keyframes barberpole {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: 30px 0; }} }}
      .progress-label {{
        font-size: 0.8rem; font-weight: 600; color: var(--ink-2); margin-top: 4px;
        display: flex; justify-content: space-between;
      }}
      /* Log viewer */
      .log-viewer {{
        background: #0d1117; color: #c9d1d9; padding: 12px 14px; border-radius: var(--radius);
        height: 420px; overflow: auto; font-family: var(--mono); font-size: 0.78rem;
        line-height: 1.65; white-space: pre-wrap; word-break: break-all; border: 1px solid #21262d;
      }}
      [data-theme="dark"] .log-viewer {{ background: #010409; border-color: #21262d; }}
      .log-line {{ display: block; }}
      .log-line-error {{ color: #f87171; }}
      .log-line-warning {{ color: #fbbf24; }}
      .log-line-debug {{ color: #6b7280; }}
      .log-controls {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
      .log-controls input {{
        flex: 1; padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border);
        background: var(--panel); color: var(--ink); font-family: var(--font); font-size: 0.8rem;
      }}
      .log-controls button {{
        padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
        background: var(--bg-alt); color: var(--ink-2); cursor: pointer; font-size: 0.8rem;
        font-family: var(--font); font-weight: 500;
      }}
      /* Error detail panel */
      .error-detail {{
        background: var(--danger-soft); border: 1px solid var(--danger); border-radius: var(--radius);
        padding: 12px; margin-top: 10px;
      }}
      .error-detail summary {{ cursor: pointer; font-weight: 600; color: var(--danger); font-size: 0.85rem; }}
      .error-detail pre {{
        margin: 8px 0 0 0; font-family: var(--mono); font-size: 0.75rem;
        white-space: pre-wrap; color: var(--ink); max-height: 300px; overflow: auto;
      }}
      /* Config panel */
      .config-panel {{ margin-top: 0; }}
      .config-panel .field label,
      .config-panel input:not([type="checkbox"]),
      .config-panel select,
      .config-panel textarea,
      .config-panel .file-entry,
      .config-panel .browser-header {{ font-size: 0.8rem; line-height: 1.3; }}
      .config-browser {{
        margin-top: 12px; border: 1px dashed var(--border); border-radius: var(--radius);
        padding: 12px; background: var(--bg-alt);
      }}
      .browser-header {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 0.85rem; }}
      .file-list {{
        max-height: 220px; overflow: auto; display: grid; gap: 4px; padding: 8px;
        margin-top: 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--panel);
      }}
      .file-entry {{
        text-align: left; padding: 5px 8px; border-radius: 6px;
        border: 1px solid transparent; background: var(--bg-alt); cursor: pointer;
        font-family: var(--mono); font-size: 0.78rem; color: var(--ink);
      }}
      .file-entry:hover {{ background: var(--border); }}
      .file-entry.dir {{ font-weight: 600; }}
      .file-entry.file.selected {{ border-color: var(--accent); background: var(--accent-soft); }}
      .load-form {{ margin-top: 10px; }}
      .config-form {{ margin-top: 16px; display: grid; gap: 12px; }}
      .config-fields {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }}
      .section-title {{
        grid-column: 1 / -1; font-weight: 700; margin-top: 14px; padding-bottom: 4px;
        border-bottom: 2px solid var(--border); font-size: 0.9rem; cursor: pointer;
        display: flex; align-items: center; gap: 6px;
      }}
      .section-title::before {{ content: "\\25BC"; font-size: 0.65rem; color: var(--ink-3); transition: transform 0.15s; }}
      .section-title.collapsed::before {{ transform: rotate(-90deg); }}
      .subsection-title {{ grid-column: 1 / -1; font-weight: 600; margin-top: 8px; color: var(--ink-2); font-size: 0.82rem; }}
      .subsub-title {{ grid-column: 1 / -1; font-weight: 600; margin-top: 4px; color: var(--ink-3); font-size: 0.78rem; }}
      .field {{ display: grid; gap: 4px; }}
      .field.missing input, .field.missing textarea, .field.missing select {{ border-color: var(--danger); }}
      .status {{ margin: 8px 0; padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; }}
      .status.info {{ background: var(--bg-alt); color: var(--ink-2); }}
      .status.ok {{ background: var(--success-soft); color: var(--success); }}
      .status.error {{ background: var(--danger-soft); color: var(--danger); }}
      .status.warning {{ background: var(--warning-soft); color: var(--warning); }}
      .status ul {{ margin: 6px 0 0 18px; }}
      form {{ display: grid; gap: 8px; margin-top: 8px; }}
      .run-actions {{ display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }}
      .mode-toggle {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
      .muted {{ color: var(--ink-2); font-size: 0.85rem; }}
      label {{ font-weight: 600; font-size: 0.82rem; }}
      input, select {{
        padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border);
        background: var(--panel); color: var(--ink); font-family: var(--font); font-size: 0.85rem;
      }}
      textarea {{
        padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border);
        background: var(--panel); color: var(--ink); font-family: var(--mono); font-size: 0.8rem; resize: vertical;
      }}
      input:focus, select:focus, textarea:focus {{ outline: 2px solid var(--accent); outline-offset: -1px; border-color: var(--accent); }}
      input[type="checkbox"] {{ width: 15px; height: 15px; margin: 0; accent-color: var(--accent); }}
      button {{
        padding: 7px 14px; border-radius: 8px; border: none; cursor: pointer;
        font-weight: 600; font-family: var(--font); font-size: 0.82rem; transition: all 0.15s;
      }}
      button.ghost {{ background: transparent; border: 1px solid var(--border); color: var(--ink-2); }}
      button.ghost:hover {{ background: var(--bg-alt); }}
      button.primary {{ background: var(--accent); color: #fff; }}
      button.primary:hover {{ background: var(--accent-hover); }}
      button.danger {{ background: var(--danger); color: #fff; }}
      button[disabled] {{ opacity: 0.45; cursor: not-allowed; }}
      .chip {{ padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
      .chip.ok {{ background: var(--success-soft); color: var(--success); }}
      .chip.muted {{ background: var(--bg-alt); color: var(--ink-3); }}
      .hint {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 16px; height: 16px; margin-left: 4px; border-radius: 50%;
        background: var(--bg-alt); color: var(--ink-3); font-size: 0.65rem;
        cursor: help; position: relative;
      }}
      .hint::after {{
        content: attr(data-hint); position: absolute; bottom: 150%; left: 50%;
        transform: translateX(-50%); background: var(--ink); color: var(--panel);
        padding: 6px 10px; border-radius: 6px; font-size: 0.72rem; white-space: nowrap;
        max-width: 300px; opacity: 0; pointer-events: none; transition: opacity 0.15s; z-index: 10;
      }}
      .hint:hover::after {{ opacity: 1; }}
      @media (max-width: 720px) {{
        .shell {{ padding: 12px; }}
        .card-grid {{ grid-template-columns: 1fr; }}
        .tab-bar {{ width: 100%; }}
        .tab-btn {{ flex: 1; text-align: center; }}
      }}
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="header">
        <h1>Observability Dashboard</h1>
        <div class="header-controls">
          <select id="refresh-rate" onchange="setRefreshRate(this.value)" style="padding:5px 8px;border-radius:6px;border:1px solid var(--border);background:var(--bg-alt);color:var(--ink);font-size:0.78rem;font-family:var(--font)">
            <option value="2">2s</option>
            <option value="5" selected>5s</option>
            <option value="10">10s</option>
            <option value="30">30s</option>
          </select>
          <button class="theme-toggle" onclick="exportRunState()" title="Export run state as JSON" aria-label="Export">&#x2B07;</button>
          <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode" aria-label="Toggle theme">
            <span id="theme-icon"></span>
          </button>
        </div>
      </div>
      <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab(this,'dashboard')">Dashboard</button>
        <button class="tab-btn" onclick="switchTab(this,'config')">Config</button>
        <button class="tab-btn" onclick="switchTab(this,'logs')">Logs</button>
        <button class="tab-btn" onclick="switchTab(this,'history')">History</button>
      </div>
      <div id="tab-dashboard" class="tab-panel active">
        <div id="run-status" class="card" hx-get="/ui/status" hx-trigger="load, every 5s"></div>
        <div class="card-grid" style="margin-top:16px">
          <div class="card" hx-get="/ui/progress" hx-trigger="load, every 5s"></div>
          <div class="card" hx-get="/ui/metrics" hx-trigger="load, every 5s"></div>
        </div>
        <div class="card-grid">
          <div class="card" hx-get="/ui/stats" hx-trigger="load, every 5s"></div>
          <div class="card" hx-get="/ui/system" hx-trigger="load, every 5s"></div>
        </div>
        <div class="card" id="training-chart-card" hx-get="/ui/training-chart" hx-trigger="load, every 5s"></div>
        <div class="card" id="hpo-card" hx-get="/ui/hpo" hx-trigger="load, every 5s"></div>
      </div>
      <div id="tab-config" class="tab-panel">
        <div id="config-panel" hx-get="/ui/config" hx-trigger="load" hx-swap="outerHTML"></div>
        <div id="run-control" hx-get="/ui/run-control" hx-trigger="load, every 5s" style="margin-top:16px"></div>
      </div>
      <div id="tab-logs" class="tab-panel">
        <div class="card">
          <div class="card-header"><h3>Logs</h3></div>
          <div class="log-controls">
            <input type="text" id="log-filter" placeholder="Filter logs..." oninput="filterLogs()" />
            <button onclick="toggleAutoScroll()" id="autoscroll-btn">Auto-scroll: ON</button>
          </div>
          <div id="logs-panel" hx-get="/ui/logs" hx-trigger="load, every 2s"
               hx-on::afterSwap="if(window._autoScroll!==false){{const el=this.querySelector('.log-viewer');if(el)el.scrollTop=el.scrollHeight;}}"></div>
        </div>
      </div>
      <div id="tab-history" class="tab-panel">
        <div class="card" hx-get="/ui/history" hx-trigger="load, every 15s"></div>
      </div>
    </div>
    <script>
      function getPreferredTheme(){{var s=localStorage.getItem('obs-theme');if(s)return s;return window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}}
      function applyTheme(t){{document.documentElement.setAttribute('data-theme',t);document.getElementById('theme-icon').textContent=t==='dark'?'\u2600\uFE0F':'\uD83C\uDF19';localStorage.setItem('obs-theme',t);}}
      function toggleTheme(){{var c=document.documentElement.getAttribute('data-theme')||'light';applyTheme(c==='dark'?'light':'dark');}}
      applyTheme(getPreferredTheme());
      function switchTab(btn,name){{document.querySelectorAll('.tab-panel').forEach(function(p){{p.classList.remove('active')}});document.querySelectorAll('.tab-btn').forEach(function(t){{t.classList.remove('active')}});document.getElementById('tab-'+name).classList.add('active');btn.classList.add('active');}}
      function filterLogs(){{var q=document.getElementById('log-filter').value.toLowerCase();document.querySelectorAll('#logs-panel .log-line').forEach(function(l){{l.style.display=(!q||l.textContent.toLowerCase().indexOf(q)!==-1)?'':'none'}});}}
      window._autoScroll=true;
      function toggleAutoScroll(){{window._autoScroll=!window._autoScroll;document.getElementById('autoscroll-btn').textContent='Auto-scroll: '+(window._autoScroll?'ON':'OFF');}}
      document.addEventListener('click',function(e){{if(e.target.classList.contains('section-title')){{e.target.classList.toggle('collapsed');var el=e.target.nextElementSibling;while(el&&!el.classList.contains('section-title')){{el.style.display=e.target.classList.contains('collapsed')?'none':'';el=el.nextElementSibling;}}}}}});
      function exportRunState(){{fetch('/api/run-state').then(function(r){{return r.json()}}).then(function(d){{var blob=new Blob([JSON.stringify(d,null,2)],{{type:'application/json'}});var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='run_state_'+new Date().toISOString().slice(0,19).replace(/:/g,'-')+'.json';a.click();}}).catch(function(e){{alert('Export failed: '+e);}});}}
      function setRefreshRate(sec){{var val=parseInt(sec,10)||5;document.querySelectorAll('[hx-trigger*="every"]').forEach(function(el){{var t=el.getAttribute('hx-trigger');if(t){{var newT=t.replace(/every \\d+s/g,'every '+val+'s');el.setAttribute('hx-trigger',newT);if(window.htmx)htmx.process(el);}}}});}}
    </script>
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
        parsed = urlparse(self.path)
        path = parsed.path

        # Health check endpoint — no auth required (for load balancers / probes)
        if path == "/healthz":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            run_status = str(state.get("status", "idle"))
            is_stale = _is_run_state_stale(state)
            self._send_json({
                "status": "ok",
                "run_status": run_status,
                "run_state_stale": is_stale,
            })
            return

        if not self._require_auth():
            return

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
                self._send_html(
                    '<div class="card-header"><h3>Run Status</h3>'
                    '<span class="badge idle">No Active Run</span></div>'
                    f'<div class="kv"><span class="k">State Source</span>'
                    f'<span class="v">{_escape_text(self.server_state.config.run_state_path)}</span></div>'
                )
                return
            run_status = str(state.get("status", "idle"))
            run_stage = str(state.get("stage", "idle"))
            heartbeat_age_seconds = _heartbeat_age_seconds(state.get("heartbeat_time"))
            heartbeat_age = "n/a"
            if heartbeat_age_seconds is not None:
                heartbeat_age = f"{int(heartbeat_age_seconds)}s"

            is_stale = _is_run_state_stale(state)
            stale_badge = '<span class="badge unknown">unknown</span>'
            if heartbeat_age_seconds is not None:
                stale_badge = '<span class="badge stale">stale</span>' if is_stale else '<span class="badge fresh">fresh</span>'

            status_class = run_status if run_status in ("running", "completed", "failed", "idle") else "idle"
            warnings = _run_state_path_warnings(state, self.server_state.config)
            stage_ts = state.get("stage_timestamps")
            stepper_html = _render_pipeline_stepper(run_stage, run_status, stage_timestamps=stage_ts)

            body = (
                f'<div class="card-header"><h3>Run Status</h3>'
                f'<span class="badge {status_class}">{_escape_text(run_status)}</span></div>'
                f'{stepper_html}'
                f'<div class="kv"><span class="k">Stage</span><span class="v">{_escape_text(run_stage)}</span></div>'
                f'<div class="kv"><span class="k">Run ID</span><span class="v">{_escape_text(state.get("run_id", "-"))}</span></div>'
                f'<div class="kv"><span class="k">Run PID</span><span class="v">{_escape_text(state.get("run_process_pid"))}</span></div>'
                f'<div class="kv"><span class="k">Heartbeat Age</span><span class="v">{heartbeat_age} {stale_badge}</span></div>'
            )
            if warnings:
                for msg in warnings:
                    body += f'<div class="status warning">{_escape_text(msg)}</div>'
            last_error = state.get("last_error")
            last_tb = state.get("last_traceback")
            if last_error:
                body += '<details class="error-detail" open>'
                body += f'<summary>Error: {_escape_text(last_error)}</summary>'
                if last_tb:
                    body += f'<pre>{_escape_text(last_tb)}</pre>'
                body += '</details>'
            self._send_html(body)
            return

        if path == "/ui/progress":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            progress = float(state.get("progress", 0.0)) * 100.0
            eta = state.get("eta_seconds")
            run_status = str(state.get("status", "idle"))
            stage = str(state.get("stage", "idle"))
            eta_display = f"{int(eta)}s" if eta is not None else "n/a"
            active_class = " active" if run_status == "running" else ""
            stage_class = stage if stage in ("training", "snapshot_build", "evaluation") else ""
            body = (
                '<div class="card-header"><h3>Progress</h3></div>'
                '<div class="progress-wrap">'
                '<div class="progress-bar-outer">'
                f'<div class="progress-bar-inner {stage_class}{active_class}" style="width:{progress:.1f}%"></div>'
                '</div>'
                f'<div class="progress-label"><span>{progress:.1f}%</span><span>ETA: {eta_display}</span></div>'
                '</div>'
            )
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

            body = '<div class="card-header"><h3>Duty Cycle</h3></div>'
            body += f'<div class="kv"><span class="k">Min</span><span class="v">{_fmt(state.get("duty_cycle_min"))}</span></div>'
            body += f'<div class="kv"><span class="k">Median</span><span class="v">{_fmt(state.get("duty_cycle_median"))}</span></div>'
            body += f'<div class="kv"><span class="k">P95</span><span class="v">{_fmt(state.get("duty_cycle_p95"))}</span></div>'
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
            heartbeat_ts = state.get("heartbeat_time")
            runtime = None
            if start_ts is not None:
                try:
                    runtime = float(datetime.now().timestamp()) - float(start_ts)
                except (TypeError, ValueError):
                    runtime = None
            heartbeat_age = None
            if heartbeat_ts is not None:
                try:
                    heartbeat_age = max(float(datetime.now().timestamp()) - float(heartbeat_ts), 0.0)
                except (TypeError, ValueError):
                    heartbeat_age = None

            body = '<div class="card-header"><h3>Run Stats</h3></div>'
            body += f'<div class="kv"><span class="k">Start</span><span class="v">{_fmt_time(start_ts)}</span></div>'
            body += f'<div class="kv"><span class="k">Runtime</span><span class="v">{_fmt_duration(runtime)}</span></div>'
            body += f'<div class="kv"><span class="k">Last Update</span><span class="v">{_fmt_time(updated_ts)}</span></div>'
            body += (
                f'<div class="kv"><span class="k">Snapshots</span>'
                f'<span class="v">{state.get("snapshot_chunks_processed", 0)} / {state.get("snapshot_chunks_total", 0)}</span></div>'
            )
            body += (
                f'<div class="kv"><span class="k">Epochs</span>'
                f'<span class="v">{state.get("training_epochs_done", 0)} / {state.get("training_epochs_total", 0)}</span></div>'
            )
            body += (
                f'<div class="kv"><span class="k">Batches</span>'
                f'<span class="v">{state.get("training_batches_done", 0)} / {state.get("training_batches_total", 0)}</span></div>'
            )
            body += (
                f'<div class="kv"><span class="k">Eval</span>'
                f'<span class="v">{state.get("eval_batches_done", 0)} / {state.get("eval_batches_total", 0)}</span></div>'
            )
            body += (
                f'<div class="kv"><span class="k">HPO Trials</span>'
                f'<span class="v">{state.get("hpo_trials_completed", 0)} / {state.get("hpo_trials_total", 0)}'
                f' (pruned={state.get("hpo_trials_pruned", 0)}, failed={state.get("hpo_trials_failed", 0)})</span></div>'
            )
            self._send_html(body)
            return

        if path == "/ui/system":
            metrics = self.server_state.get_system_metrics()
            state = load_run_state(self.server_state.config.run_state_path) or {}

            def _fmt_bytes(value: Any) -> str:
                if value is None:
                    return "n/a"
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    return "n/a"
                units = ["B", "KiB", "MiB", "GiB", "TiB"]
                idx = 0
                while num >= 1024.0 and idx < len(units) - 1:
                    num /= 1024.0
                    idx += 1
                return f"{num:.2f} {units[idx]}"

            def _fmt_ratio(value: Any) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{float(value) * 100.0:.1f}%"
                except (TypeError, ValueError):
                    return "n/a"

            def _fmt_number(value: Any) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{float(value):.2f}"
                except (TypeError, ValueError):
                    return "n/a"

            body = '<div class="card-header"><h3>System</h3></div>'
            body += (
                f'<div class="kv"><span class="k">CPU Load (1/5/15m)</span>'
                f'<span class="v">{_fmt_number(metrics.get("load_1m"))} / '
                f'{_fmt_number(metrics.get("load_5m"))} / {_fmt_number(metrics.get("load_15m"))}</span></div>'
            )
            body += f'<div class="kv"><span class="k">CPU Normalized (1m)</span><span class="v">{_fmt_ratio(metrics.get("load_norm_1m"))}</span></div>'
            body += (
                f'<div class="kv"><span class="k">RAM Used</span>'
                f'<span class="v">{_fmt_bytes(metrics.get("mem_used_bytes"))} / '
                f'{_fmt_bytes(metrics.get("mem_total_bytes"))} ({_fmt_ratio(metrics.get("mem_used_ratio"))})</span></div>'
            )
            body += f'<div class="kv"><span class="k">Run PID</span><span class="v">{_escape_text(metrics.get("run_process_pid"))}</span></div>'
            body += f'<div class="kv"><span class="k">Run RSS</span><span class="v">{_fmt_bytes(metrics.get("run_process_rss_bytes"))}</span></div>'

            hpo_worker_count = state.get("hpo_wave_worker_count")
            if hpo_worker_count:
                body += f'<div class="kv"><span class="k">HPO Workers</span><span class="v">{_escape_text(hpo_worker_count)}</span></div>'
                body += (
                    f'<div class="kv"><span class="k">HPO RSS (cur/max)</span>'
                    f'<span class="v">{_fmt_bytes(state.get("hpo_wave_worker_rss_current_bytes"))} / '
                    f'{_fmt_bytes(state.get("hpo_wave_worker_rss_max_bytes"))}</span></div>'
                )
                watchdog_count = state.get("hpo_rss_watchdog_trigger_count", 0)
                if watchdog_count:
                    body += (
                        f'<div class="kv"><span class="k">RSS Watchdog</span>'
                        f'<span class="v">{_escape_text(watchdog_count)} triggers</span></div>'
                    )

                top_workers = state.get("hpo_wave_worker_rss_top")
                if isinstance(top_workers, list) and top_workers:
                    for worker in top_workers:
                        if not isinstance(worker, dict):
                            continue
                        worker_pid_raw = _parse_positive_int(worker.get("pid"))
                        worker_pid = _escape_text(worker_pid_raw if worker_pid_raw is not None else worker.get("pid"))
                        body += f'<div class="kv"><span class="k">Worker {worker_pid}</span><span class="v">{_fmt_bytes(worker.get("rss_bytes"))}</span></div>'

            gpus = metrics.get("gpus")
            if isinstance(gpus, list) and gpus:
                for gpu in gpus:
                    if not isinstance(gpu, dict):
                        continue
                    body += (
                        f'<div class="kv"><span class="k">GPU {int(gpu.get("index", 0))} {_escape_text(gpu.get("name", ""))}</span>'
                        f'<span class="v">{_fmt_ratio(gpu.get("utilization_ratio"))} util, '
                        f'{_fmt_bytes(gpu.get("memory_used_bytes"))}/{_fmt_bytes(gpu.get("memory_total_bytes"))}, '
                        f'{_fmt_number(gpu.get("power_draw_watts"))}W</span></div>'
                    )
            else:
                body += '<div class="kv"><span class="k">GPU</span><span class="v">n/a</span></div>'

            self._send_html(body)
            return

        if path == "/ui/hpo":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            trials = state.get("hpo_trial_results")
            body = '<div class="card-header"><h3>HPO Trials</h3></div>'
            if not isinstance(trials, list) or not trials:
                body += '<div class="muted" style="padding:20px 0;text-align:center">No HPO trial data yet.</div>'
                self._send_html(body)
                return

            best_value: Optional[float] = None
            best_number: Optional[int] = None
            for t in trials:
                if t.get("status") == "completed" and t.get("value") is not None:
                    v = float(t["value"])
                    if best_value is None or v < best_value:
                        best_value = v
                        best_number = t.get("number")

            param_keys = sorted({k for t in trials for k in (t.get("params") or {})})
            body += '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
            body += '<thead><tr style="border-bottom:2px solid var(--border);text-align:left">'
            body += '<th style="padding:5px 8px">#</th><th style="padding:5px 8px">Status</th><th style="padding:5px 8px">Value</th><th style="padding:5px 8px">Duration</th>'
            for pk in param_keys:
                body += f'<th style="padding:5px 8px">{_escape_text(pk)}</th>'
            body += '</tr></thead><tbody>'

            for t in trials:
                number = t.get("number", "-")
                status = str(t.get("status", "-"))
                badge_class = status if status in ("completed", "pruned", "failed", "running") else "idle"
                value = t.get("value")
                value_str = f"{float(value):.5f}" if value is not None else "-"
                dur = t.get("duration")
                dur_str = f"{int(dur)}s" if dur is not None else "-"
                is_best = (number == best_number and status == "completed")
                row_style = "border-bottom:1px solid var(--border);"
                if is_best:
                    row_style += "background:var(--success-soft);"
                body += f'<tr style="{row_style}">'
                best_marker = " *" if is_best else ""
                body += f'<td style="padding:5px 8px;font-weight:600">{_escape_text(number)}{best_marker}</td>'
                body += f'<td style="padding:5px 8px"><span class="badge {badge_class}">{_escape_text(status)}</span></td>'
                body += f'<td style="padding:5px 8px;font-variant-numeric:tabular-nums">{value_str}</td>'
                body += f'<td style="padding:5px 8px">{dur_str}</td>'
                params = t.get("params") or {}
                for pk in param_keys:
                    pv = params.get(pk)
                    if isinstance(pv, float):
                        pv_str = f"{pv:.4g}"
                    elif pv is not None:
                        pv_str = str(pv)
                    else:
                        pv_str = "-"
                    body += f'<td style="padding:5px 8px;font-family:var(--mono);font-size:0.75rem">{_escape_text(pv_str)}</td>'
                body += '</tr>'
            body += '</tbody></table></div>'
            self._send_html(body)
            return

        if path == "/ui/history":
            history = load_run_history(self.server_state.config.run_state_path, limit=50)
            body = '<div class="card-header"><h3>Run History</h3></div>'
            if not history:
                body += '<div class="muted" style="padding:20px 0;text-align:center">No completed runs yet.</div>'
                self._send_html(body)
                return

            def _fmt_ts(ts: Any) -> str:
                if ts is None:
                    return "-"
                try:
                    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
                except (TypeError, ValueError):
                    return "-"

            def _fmt_dur(start: Any, end: Any) -> str:
                if start is None or end is None:
                    return "-"
                try:
                    secs = max(float(end) - float(start), 0)
                except (TypeError, ValueError):
                    return "-"
                if secs < 60:
                    return f"{int(secs)}s"
                minutes = int(secs) // 60
                remainder = int(secs) % 60
                return f"{minutes}m{remainder:02d}s"

            def _fmt_metric(v: Any) -> str:
                if v is None:
                    return "-"
                try:
                    return f"{float(v):.4f}"
                except (TypeError, ValueError):
                    return "-"

            body += (
                '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
                '<thead><tr style="border-bottom:2px solid var(--border);text-align:left">'
                '<th style="padding:6px 8px">Run ID</th>'
                '<th style="padding:6px 8px">Status</th>'
                '<th style="padding:6px 8px">Started</th>'
                '<th style="padding:6px 8px">Duration</th>'
                '<th style="padding:6px 8px">Epochs</th>'
                '<th style="padding:6px 8px">Loss</th>'
                '<th style="padding:6px 8px">Val Loss</th>'
                '</tr></thead><tbody>'
            )
            for run in history:
                status = str(run.get("status", "-"))
                badge_class = status if status in ("completed", "failed", "running") else "idle"
                body += (
                    f'<tr style="border-bottom:1px solid var(--border)">'
                    f'<td style="padding:6px 8px;font-family:var(--mono);font-size:0.75rem">{_escape_text(run.get("run_id") or "-")}</td>'
                    f'<td style="padding:6px 8px"><span class="badge {badge_class}">{_escape_text(status)}</span></td>'
                    f'<td style="padding:6px 8px">{_fmt_ts(run.get("start_time"))}</td>'
                    f'<td style="padding:6px 8px">{_fmt_dur(run.get("start_time"), run.get("end_time"))}</td>'
                    f'<td style="padding:6px 8px">{_escape_text(run.get("total_epochs") or "-")}</td>'
                    f'<td style="padding:6px 8px">{_fmt_metric(run.get("final_loss"))}</td>'
                    f'<td style="padding:6px 8px">{_fmt_metric(run.get("final_val_loss"))}</td>'
                    f'</tr>'
                )
            body += '</tbody></table></div>'
            self._send_html(body)
            return

        if path == "/ui/training-chart":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            epoch_metrics = state.get("training_epoch_metrics")
            if not isinstance(epoch_metrics, list) or not epoch_metrics:
                self._send_html(
                    '<div class="card-header"><h3>Training Curves</h3></div>'
                    '<div class="muted" style="padding:20px 0;text-align:center">No epoch data yet.</div>'
                )
                return
            epochs = [str(m.get("epoch", i + 1)) for i, m in enumerate(epoch_metrics)]
            metric_keys = sorted({k for m in epoch_metrics for k in m if k != "epoch"})
            palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]
            datasets_js_parts: List[str] = []
            for idx, key in enumerate(metric_keys):
                color = palette[idx % len(palette)]
                values = [str(m.get(key, "null")) for m in epoch_metrics]
                is_val = key.startswith("val_")
                dash = "borderDash:[5,3]," if is_val else ""
                datasets_js_parts.append(
                    f'{{label:"{_escape_text(key)}",data:[{",".join(values)}],'
                    f'borderColor:"{color}",backgroundColor:"{color}22",{dash}'
                    f'tension:0.3,pointRadius:2,borderWidth:2,fill:false}}'
                )
            datasets_js = ",".join(datasets_js_parts)
            chart_id = "trainChart"
            body = (
                '<div class="card-header"><h3>Training Curves</h3></div>'
                f'<canvas id="{chart_id}" style="width:100%;max-height:320px"></canvas>'
                '<script>'
                f'(function(){{'
                f'var ctx=document.getElementById("{chart_id}");'
                f'if(!ctx)return;'
                f'if(ctx._chartInstance){{ctx._chartInstance.destroy();}}'
                f'var isDark=document.documentElement.getAttribute("data-theme")==="dark";'
                f'var gridColor=isDark?"rgba(255,255,255,0.08)":"rgba(0,0,0,0.06)";'
                f'var tickColor=isDark?"#9499ad":"#5a6072";'
                f'ctx._chartInstance=new Chart(ctx,{{'
                f'type:"line",'
                f'data:{{labels:[{",".join(repr(e) for e in epochs)}],datasets:[{datasets_js}]}},'
                f'options:{{responsive:true,maintainAspectRatio:false,animation:false,'
                f'plugins:{{legend:{{position:"top",labels:{{color:tickColor,font:{{size:11}}}}}}}},'
                f'scales:{{x:{{grid:{{color:gridColor}},ticks:{{color:tickColor,font:{{size:10}}}},title:{{display:true,text:"Epoch",color:tickColor}}}},'
                f'y:{{grid:{{color:gridColor}},ticks:{{color:tickColor,font:{{size:10}}}}}}}}}}'
                f'}});'
                f'}})()'
                '</script>'
            )
            self._send_html(body)
            return

        if path == "/ui/logs":
            cfg = self.server_state.config
            lines = _tail_log(cfg.run_log_path, max_lines=cfg.tail_max_lines)
            colored_lines: List[str] = []
            for line in lines:
                css_class = "log-line"
                lower = line.lower()
                if "[error]" in lower or "error:" in lower or "traceback" in lower:
                    css_class += " log-line-error"
                elif "[warning]" in lower or "warning:" in lower:
                    css_class += " log-line-warning"
                elif "[debug]" in lower:
                    css_class += " log-line-debug"
                colored_lines.append(f'<span class="{css_class}">{_escape_text(line)}</span>')
            body = '<div class="log-viewer">' + "\n".join(colored_lines) + "</div>"
            self._send_html(body)
            return

        if path == "/metrics":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            system_metrics = self.server_state.get_system_metrics()
            registry = CollectorRegistry()
            Gauge("run_progress", "Run progress", registry=registry).set(float(state.get("progress", 0.0)))
            Gauge("run_eta_seconds", "Run ETA seconds", registry=registry).set(float(state.get("eta_seconds") or 0.0))
            heartbeat_ts = state.get("heartbeat_time")
            heartbeat_age = float(_heartbeat_age_seconds(heartbeat_ts) or 0.0)
            Gauge("run_heartbeat_age_seconds", "Run heartbeat age seconds", registry=registry).set(heartbeat_age)
            Gauge("run_heartbeat_unix_seconds", "Run heartbeat unix timestamp", registry=registry).set(
                float(heartbeat_ts or 0.0)
            )
            Gauge("run_state_stale", "Run state stale flag (1 stale, 0 fresh)", registry=registry).set(
                1.0 if _is_run_state_stale(state) else 0.0
            )
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
            Gauge("hpo_trials_total", "HPO trials total", registry=registry).set(
                float(state.get("hpo_trials_total", 0))
            )
            Gauge("hpo_trials_completed", "HPO trials completed", registry=registry).set(
                float(state.get("hpo_trials_completed", 0))
            )
            Gauge("hpo_trials_pruned", "HPO trials pruned", registry=registry).set(
                float(state.get("hpo_trials_pruned", 0))
            )
            Gauge("hpo_trials_failed", "HPO trials failed", registry=registry).set(
                float(state.get("hpo_trials_failed", 0))
            )

            Gauge("system_memory_total_bytes", "System memory total bytes", registry=registry).set(
                float(system_metrics.get("mem_total_bytes") or 0.0)
            )
            Gauge("system_memory_used_bytes", "System memory used bytes", registry=registry).set(
                float(system_metrics.get("mem_used_bytes") or 0.0)
            )
            Gauge("system_memory_used_ratio", "System memory used ratio", registry=registry).set(
                float(system_metrics.get("mem_used_ratio") or 0.0)
            )
            Gauge("system_loadavg_1m", "System load average 1 minute", registry=registry).set(
                float(system_metrics.get("load_1m") or 0.0)
            )
            Gauge("system_loadavg_5m", "System load average 5 minutes", registry=registry).set(
                float(system_metrics.get("load_5m") or 0.0)
            )
            Gauge("system_loadavg_15m", "System load average 15 minutes", registry=registry).set(
                float(system_metrics.get("load_15m") or 0.0)
            )
            Gauge("system_loadavg_normalized_1m", "System normalized load average 1 minute", registry=registry).set(
                float(system_metrics.get("load_norm_1m") or 0.0)
            )
            Gauge("run_process_rss_bytes", "Run process RSS bytes", registry=registry).set(
                float(system_metrics.get("run_process_rss_bytes") or 0.0)
            )
            Gauge("hpo_wave_worker_count", "HPO wave active worker count", registry=registry).set(
                float(state.get("hpo_wave_worker_count") or 0.0)
            )
            Gauge("hpo_wave_worker_rss_current_bytes", "HPO wave current hottest worker RSS bytes", registry=registry).set(
                float(state.get("hpo_wave_worker_rss_current_bytes") or 0.0)
            )
            Gauge("hpo_wave_worker_rss_max_bytes", "HPO wave maximum worker RSS bytes", registry=registry).set(
                float(state.get("hpo_wave_worker_rss_max_bytes") or 0.0)
            )
            Gauge("hpo_rss_watchdog_trigger_count", "HPO RSS watchdog trigger count", registry=registry).set(
                float(state.get("hpo_rss_watchdog_trigger_count") or 0.0)
            )
            Gauge("hpo_rss_watchdog_last_trigger_unix_seconds", "HPO RSS watchdog last trigger timestamp", registry=registry).set(
                float(state.get("hpo_rss_watchdog_last_trigger_time") or 0.0)
            )

            gpus = system_metrics.get("gpus")
            if isinstance(gpus, list):
                gpu_util = Gauge(
                    "gpu_utilization_ratio",
                    "GPU utilization ratio",
                    ["gpu", "name"],
                    registry=registry,
                )
                gpu_mem_used = Gauge(
                    "gpu_memory_used_bytes",
                    "GPU memory used bytes",
                    ["gpu", "name"],
                    registry=registry,
                )
                gpu_mem_total = Gauge(
                    "gpu_memory_total_bytes",
                    "GPU memory total bytes",
                    ["gpu", "name"],
                    registry=registry,
                )
                gpu_mem_ratio = Gauge(
                    "gpu_memory_used_ratio",
                    "GPU memory used ratio",
                    ["gpu", "name"],
                    registry=registry,
                )
                gpu_power_draw = Gauge(
                    "gpu_power_draw_watts",
                    "GPU power draw watts",
                    ["gpu", "name"],
                    registry=registry,
                )
                gpu_power_limit = Gauge(
                    "gpu_power_limit_watts",
                    "GPU power limit watts",
                    ["gpu", "name"],
                    registry=registry,
                )
                for gpu in gpus:
                    if not isinstance(gpu, dict):
                        continue
                    index = str(int(gpu.get("index", 0)))
                    name = str(gpu.get("name") or "unknown")
                    gpu_util.labels(gpu=index, name=name).set(float(gpu.get("utilization_ratio") or 0.0))
                    gpu_mem_used.labels(gpu=index, name=name).set(float(gpu.get("memory_used_bytes") or 0.0))
                    gpu_mem_total.labels(gpu=index, name=name).set(float(gpu.get("memory_total_bytes") or 0.0))
                    gpu_mem_ratio.labels(gpu=index, name=name).set(float(gpu.get("memory_used_ratio") or 0.0))
                    gpu_power_draw.labels(gpu=index, name=name).set(float(gpu.get("power_draw_watts") or 0.0))
                    gpu_power_limit.labels(gpu=index, name=name).set(float(gpu.get("power_limit_watts") or 0.0))

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

        if path == "/api/run-state":
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
