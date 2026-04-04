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
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import glob
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
from utils.config_loader import _resolve_env_placeholders, _deep_merge, ConfigError
from utils.formatting import _fmt_metric


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
        self.duty_cycle_chart_resolution_days = int(
            file_config.get("duty_cycle_chart_resolution_days", 1)
        )
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
        try:
            return _resolve_env_placeholders(data)
        except ConfigError as exc:
            raise RuntimeError(str(exc)) from exc

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
    """Return the last max_lines from a file using O(max_lines) memory.
    
    Uses a contiguous byte-buffer strategy: reads backward in fixed-size chunks,
    prepends each chunk to a growing buffer, and splits lines only when the full
    buffer is available. This ensures no line is split at chunk boundaries.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return ["Log file not found"]
    
    try:
        file_size = path_obj.stat().st_size
    except (OSError, ValueError):
        return ["Unable to read log file"]
    
    if file_size == 0:
        return []
    
    CHUNK_READ_SIZE = 64 * 1024
    MAX_BUFFER_SIZE = 10 * 1024 * 1024
    
    buffer = b""
    collected_lines: List[str] = []
    seek_pos = file_size
    
    try:
        with open(path_obj, "rb") as f:
            while len(collected_lines) < max_lines and seek_pos > 0:
                read_size = min(CHUNK_READ_SIZE, seek_pos)
                seek_pos -= read_size
                f.seek(seek_pos)
                chunk = f.read(read_size)
                
                buffer = chunk + buffer
                
                if len(buffer) > MAX_BUFFER_SIZE:
                    buffer = buffer[-MAX_BUFFER_SIZE:]
                
                try:
                    text = buffer.decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    text = buffer.decode("latin-1", errors="replace")
                
                lines = text.splitlines()
                collected_lines = lines[-max_lines:]
    
    except (OSError, ValueError):
        return ["Unable to read log file"]
    
    return collected_lines[-max_lines:] if collected_lines else []


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
    pattern = str(glob_pattern).strip()
    if not pattern:
        return []

    matches = [Path(match).expanduser().resolve() for match in glob.glob(pattern, recursive=True)]
    normalized: List[str] = []
    for match in matches:
        try:
            rel = match.relative_to(_CONFIG_ROOT).as_posix()
            normalized.append(f"config/{rel}")
        except ValueError:
            normalized.append(str(match))
    return sorted(set(normalized))


_MODULE_DIR = Path(__file__).resolve().parent  # observability/ directory


def _safe_static_file_path(*, static_dir: str, request_path: str) -> Optional[Path]:
    if not request_path.startswith("/static/"):
        return None
    rel = request_path[len("/static/") :]
    if not rel or rel.startswith("/"):
        return None
    static_path = Path(static_dir)
    if not static_path.is_absolute():
        static_path = _MODULE_DIR.parent / static_path
    base = static_path.resolve()
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
    # GreptimeDB multi-database mode currently supports time-splitting only.
    "data.multi_database.strategy": ["time_split"],
    "data.asset_pairs.alignment.method": ["interpolate", "bucket"],
    "data.asset_pairs.alignment.missing_policy": ["forward_fill", "skip", "error"],
    "data.order_book.representation": ["full", "aggregated", "hybrid"],
    "data.order_book.hybrid.bin_strategy": ["equal_width", "log_spaced"],
    # The current pipeline supports percentage-based bins and two-head intensity labels.
    "targets.price_classes.definition_type": ["percentage"],
    "targets.labeling.scheme": ["two_head_intensity"],
    "targets.labeling.handle_gaps": ["skip", "interpolate", "error"],
    "preprocessing.normalization.method": ["min_max", "standard"],
    "preprocessing.feature_engineering.volume_proxy_method": ["top_of_book", "total_depth"],
    "preprocessing.feature_engineering.edge_decay.method": ["linear", "exponential"],
    "preprocessing.train_test_split.method": ["chronological"],
    "preprocessing.class_balancing.method": ["undersampling"],
    "preprocessing.class_balancing.undersampling.labeling_criteria": [
        "max_intensity",
        "up_intensity",
        "down_intensity",
    ],
    "preprocessing.class_balancing.undersampling.selection_policy": [
        "uniform_time",
        "kmeans",
        "random",
    ],
    "model.framework": ["keras"],
    "model.backend": ["tensorflow"],
    "model.architecture": ["CNN_LSTM_MultiClass"],
    "model.input_representation.strategy": ["stacked_channels"],
    "model.input_representation.temporal_features.integration_mode": ["concat_channels"],
    "model.cnn.activation": ["relu", "elu", "selu", "leaky_relu", "swish", "tanh", "sigmoid"],
    "model.output.type": ["two_head_intensity"],
    "model.output.activation": ["softmax", "sigmoid", "linear"],
    "model.compilation.optimizer": ["adam", "sgd", "rmsprop", "adamw"],
    "model.compilation.loss": [
        "categorical_crossentropy", "sparse_categorical_crossentropy",
        "binary_crossentropy", "mse", "mae", "focal_loss",
    ],
    "model.long_term.summary_method": ["mean", "median", "ewma", "last"],
    "model.long_term.architecture.conv1d.activation": ["relu", "elu", "selu", "leaky_relu", "swish", "tanh"],
    "training.missing_snapshot_strategy": ["fail", "skip", "synthetic"],
    "training.runtime.device": ["cpu", "gpu"],
    "training.sample_weighting.method": ["exponential_decay"],
    "training.sample_weighting.apply_to": ["loss_function"],
    "training.fine_tuning.freeze_layers": ["none", "cnn", "cnn_lstm", "all_but_output"],
    "snapshot.on_config_mismatch": ["create_new", "error"],
    "hyperparameter_optimization.framework": ["optuna"],
    "hyperparameter_optimization.direction": ["minimize", "maximize"],
    "hyperparameter_optimization.metric": [
        "val_loss", "val_accuracy", "val_f1_score", "val_precision", "val_recall",
    ],
    "evaluation.post_hoc_calibration.method": ["temperature_scaling"],
    "evaluation.backtesting.signal_strategy": ["net_intensity", "threshold"],
    "evaluation.backtesting.position_sizing": ["equal", "confidence"],
    "evaluation.missing_snapshot_strategy": ["fail", "skip", "synthetic"],
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "diagnostics.sampling.method": ["uniform", "random"],
}

_COLOR_OPTIONS: List[str] = [
    "red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey",
    "light_red", "light_green", "light_yellow", "light_blue",
    "light_magenta", "light_cyan",
]

_COLOR_FIELDS: List[str] = [
    "logging.colors.function_names",
    "logging.colors.parameter_names",
    "logging.colors.parameter_values",
    "logging.colors.info",
    "logging.colors.warning",
    "logging.colors.error",
    "logging.colors.debug",
]

for _cf in _COLOR_FIELDS:
    _SELECT_OPTIONS[_cf] = _COLOR_OPTIONS

_HIDDEN_SECTIONS: set = {"security"}

_HIDDEN_FIELDS: set = {
    "logging.format",
    "model.long_term.input_dim",
}

_ASSET_SELECTOR_FIELDS: set = {
    "data.asset_pairs.target_asset",
}

_ASSET_LIST_FIELDS: set = {
    "data.asset_pairs.correlated_assets",
}

_DATE_FIELDS: set = {
    "data.time_range.start_date",
    "data.time_range.end_date",
}

_LIST_FIELD_OPTIONS: Dict[str, List[str]] = {
    "data.asset_pairs.correlated_assets": [],
    "data.temporal_features.local": [
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "minute_sin", "minute_cos", "second_sin", "second_cos",
    ],
    "data.temporal_features.global": [
        "day_of_month_sin", "day_of_month_cos",
        "month_sin", "month_cos", "week_of_year_sin", "week_of_year_cos",
    ],
    "preprocessing.feature_engineering.order_book_features": [
        "bid_ask_spread", "mid_price", "weighted_mid_price",
        "bid_depth", "ask_depth", "depth_imbalance", "volume_imbalance",
        "cumulative_depth_ratio", "price_gradient",
    ],
    "preprocessing.feature_engineering.derived_features": [
        "spread_ma", "mid_price_returns", "volatility",
        "momentum", "trade_intensity", "order_flow_imbalance",
        "depth_weighted_price", "vwap_proxy",
    ],
    "targets.price_classes.boundaries": [],
    "model.long_term.windows_days": [],
    "model.long_term.features": [
        "close", "volume", "spread", "volatility", "momentum",
        "depth_imbalance", "trade_intensity",
    ],
    "model.compilation.metrics": [
        "accuracy", "precision", "recall", "f1_score",
        "categorical_accuracy", "top_k_categorical_accuracy", "AUC",
    ],
    "evaluation.metrics": [
        "accuracy", "precision", "recall", "f1_score",
        "confusion_matrix", "classification_report", "roc_auc",
        "log_loss", "brier_score", "calibration_error",
    ],
    "diagnostics.visualization.heatmaps.types": [
        "spread_time", "depth_time", "imbalance_time",
        "price_level", "volume_profile",
    ],
}


_PRICE_CLASS_AUTO_UI_FIELDS: Dict[str, str] = {
    "targets.price_classes.auto.method": "string",
    "targets.price_classes.auto.per_window": "boolean",
    "targets.price_classes.auto.fit_on": "string",
    "targets.price_classes.auto.labeling_criteria": "string",
    "targets.price_classes.auto.max_samples": "integer",
    "targets.price_classes.auto.random_seed": "integer",
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


def _safe_yaml_load(value: Any) -> Any:
    """Best-effort YAML parse for form-rendered scalar/collection values."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, list, dict)):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if raw == "":
        return ""
    try:
        import yaml  # type: ignore[import]
    except Exception:  # noqa: BLE001
        return raw
    try:
        return yaml.safe_load(raw)
    except Exception:  # noqa: BLE001
        return raw


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
    """Load a training config, following the full base_config chain."""
    chain = _build_config_inheritance_chain(config_path)
    if len(chain) == 1:
        return {k: v for k, v in chain[0]["data"].items() if k != "base_config"}
    merged: Dict[str, Any] = {}
    for entry in reversed(chain):
        override = {k: v for k, v in entry["data"].items() if k != "base_config"}
        merged = _deep_merge(merged, override)
    return merged


def _build_config_inheritance_chain(config_path: Path) -> List[Dict[str, Any]]:
    """Return the inheritance chain as a list of dicts with 'path' and 'data'.

    Index 0 is the leaf (selected file), last element is the root base config.
    """
    chain: List[Dict[str, Any]] = []
    seen: set = set()
    current = config_path.resolve()
    while True:
        if str(current) in seen:
            raise RuntimeError(f"Detected recursive base_config chain at {current}")
        seen.add(str(current))
        raw = _load_yaml_file(current)
        chain.append({"path": str(current), "data": raw})
        base_ref = raw.get("base_config")
        if not base_ref or not isinstance(base_ref, str):
            break
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (current.parent / base_path).resolve()
        if not base_path.exists():
            raise RuntimeError(f"base_config file not found: {base_path} (referenced by {current})")
        current = base_path
    return chain


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

    # Optional UI fields (not part of required schema fields).
    for full_key, field_type in _PRICE_CLASS_AUTO_UI_FIELDS.items():
        value = _get_nested_value(config, full_key)
        if value is _MISSING:
            continue
        values[full_key] = _render_value(value, field_type)
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

    # Optional UI fields (not part of required schema fields).
    for full_key, field_type in _PRICE_CLASS_AUTO_UI_FIELDS.items():
        if field_type == "boolean":
            values[full_key] = full_key in params
        else:
            values[full_key] = params.get(full_key, [""])[0]
    return values


def _render_price_class_boundaries_widget(values: Dict[str, Any]) -> str:
    """Render targets.price_classes.boundaries as manual list or 'auto'."""

    full_key = "targets.price_classes.boundaries"
    uid = full_key.replace(".", "_")
    escaped_key = _escape_text(full_key)

    raw_value = values.get(full_key, "")
    parsed = _safe_yaml_load(raw_value)

    mode = "manual"
    if isinstance(parsed, str):
        first_line = parsed.strip().splitlines()[0].strip() if parsed.strip() else ""
        if first_line.strip("\"'") == "auto":
            mode = "auto"

    current_items: List[str] = []
    if mode == "manual" and isinstance(parsed, list):
        current_items = [str(v) for v in parsed]

    items_html = ""
    for idx, item in enumerate(current_items):
        items_html += (
            f'<div class="list-item" data-list="{uid}">'
            f'<span class="list-item-text">{_escape_text(item)}</span>'
            f'<button type="button" class="list-btn-sm danger" '
            f'onclick="removeListItem(this,\'{uid}\')">-</button>'
            f"</div>"
        )

    manual_style = "" if mode == "manual" else "display:none"
    auto_style = "" if mode == "auto" else "display:none"

    ta_val = "auto" if mode == "auto" else _list_to_yaml(current_items)

    auto_fields_html = ""
    for auto_key, auto_type in _PRICE_CLASS_AUTO_UI_FIELDS.items():
        label = auto_key.split(".")[-1]
        val = values.get(auto_key, "")
        if auto_type == "boolean":
            checked = "checked" if bool(val) else ""
            auto_fields_html += (
                f'<div class="pc-auto-field">'
                f'<label>{_escape_text(label)}</label>'
                f'<input type="checkbox" name="{_escape_text(auto_key)}" value="true" {checked} />'
                f"</div>"
            )
        else:
            step = "1" if auto_type == "integer" else "any"
            input_type = "number" if auto_type in {"integer", "number"} else "text"
            auto_fields_html += (
                f'<div class="pc-auto-field">'
                f'<label>{_escape_text(label)}</label>'
                f'<input type="{input_type}" name="{_escape_text(auto_key)}" step="{step}" '
                f'value="{_escape_text(val)}" />'
                f"</div>"
            )

    sel_manual = "selected" if mode == "manual" else ""
    sel_auto = "selected" if mode == "auto" else ""

    return (
        '<div class="pc-boundaries-wrap">'
        '<div class="pc-boundaries-head">'
        '<span class="muted" style="font-size:0.75rem">Mode</span>'
        f'<select id="pc_boundaries_mode" onchange="setPriceBoundariesMode(this.value)">'
        f'<option value="manual" {sel_manual}>Manual list</option>'
        f'<option value="auto" {sel_auto}>Auto</option>'
        '</select>'
        '</div>'
        f'<div id="pc_boundaries_manual" style="{manual_style}">'
        f'<div class="list-field-wrap" id="lf_{uid}">'
        f'<div class="list-items" id="items_{uid}">{items_html}</div>'
        f'<div class="list-add-row">'
        f'<input type="text" id="inp_{uid}" class="list-add-input" placeholder="value" />'
        f'<button type="button" class="list-btn-sm primary" onclick="addListItemFromInput(\'{uid}\')">+</button>'
        f'</div>'
        f"</div>"
        f"</div>"
        f'<div id="pc_boundaries_auto" style="{auto_style}">'
        '<div class="pc-auto-grid">'
        f"{auto_fields_html}"
        '</div>'
        '<div class="muted" style="font-size:0.72rem;line-height:1.3;margin-top:4px">'
        'Uses <code>targets.price_classes.auto.*</code> when boundaries is <code>auto</code>.'
        '</div>'
        '</div>'
        f'<textarea name="{escaped_key}" class="list-hidden-ta" id="ta_{uid}" rows="1">{_escape_text(ta_val)}</textarea>'
        '</div>'
    )


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


def _apply_price_class_auto_params(
    *,
    config: Dict[str, Any],
    params: Dict[str, List[str]],
    errors: List[str],
) -> None:
    """Apply optional targets.price_classes.auto.* params when boundaries == 'auto'."""

    boundaries = _get_nested_value(config, "targets.price_classes.boundaries")
    if boundaries is _MISSING:
        return
    if not isinstance(boundaries, str) or boundaries.strip().lower().strip('"\'') != "auto":
        return

    for full_key, field_type in _PRICE_CLASS_AUTO_UI_FIELDS.items():
        if field_type == "boolean":
            _set_nested_value(config, full_key, full_key in params)
            continue

        raw = params.get(full_key, [""])[0].strip()
        if raw == "":
            continue
        try:
            if field_type == "integer":
                value: Any = int(raw)
            elif field_type == "number":
                value = float(raw)
            else:
                value = raw
            _set_nested_value(config, full_key, value)
        except ValueError as exc:
            errors.append(f"{full_key}: {exc}")


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


def _render_config_chain_graph(chain: List[Dict[str, Any]]) -> str:
    """Render the config inheritance chain as a visual dependency graph."""
    if not chain:
        return ""
    parts: List[str] = []
    parts.append('<div class="chain-graph">')
    parts.append('<div class="chain-label">Inheritance chain:</div>')
    parts.append('<div class="chain-nodes">')
    for i, entry in enumerate(chain):
        path_str = entry.get("path", "")
        name = Path(path_str).name if path_str else "?"
        css = "chain-node chain-leaf" if i == 0 else "chain-node chain-base"
        tag = "leaf" if i == 0 else "base"
        parts.append(f'<span class="{css}" title="{_escape_text(path_str)}">{_escape_text(name)}'
                     f'<span class="chain-tag">{tag}</span></span>')
        if i < len(chain) - 1:
            parts.append('<span class="chain-arrow">\u2192</span>')
    parts.append('</div></div>')
    return "".join(parts)


def _render_yaml_block(data: Dict[str, Any], *, title: str, css_id: str) -> str:
    """Render a dict as a read-only syntax-highlighted YAML block."""
    try:
        import yaml  # type: ignore[import]
    except Exception:  # noqa: BLE001
        text = str(data)
    else:
        filtered = {k: v for k, v in data.items() if k != "base_config"}
        text = yaml.safe_dump(filtered, sort_keys=False, default_flow_style=False)
    lines: List[str] = []
    for line in text.splitlines():
        escaped = _escape_text(line)
        if ":" in line:
            colon_idx = escaped.index(":")
            key_part = escaped[:colon_idx]
            rest = escaped[colon_idx:]
            lines.append(f'<span class="yaml-key">{key_part}</span>{rest}')
        elif escaped.strip().startswith("- "):
            lines.append(f'<span class="yaml-list">{escaped}</span>')
        else:
            lines.append(escaped)
    return (
        f'<details class="yaml-block" id="{_escape_text(css_id)}">'
        f'<summary>{_escape_text(title)}</summary>'
        f'<pre class="yaml-pre">' + "\n".join(lines) + '</pre>'
        f'</details>'
    )


_NN_LAYER_PARAMS: Dict[str, List[Dict[str, str]]] = {
    "cnn": [
        {"name": "filters", "type": "number", "default": "32"},
        {"name": "kernel_size", "type": "text", "default": "[3,3]"},
        {"name": "pool_size", "type": "text", "default": "[2,2]"},
        {"name": "normalization", "type": "select", "default": "null",
         "options": "null,batch,group,layer"},
        {"name": "dropout", "type": "number", "default": "0.0"},
    ],
    "lstm": [
        {"name": "units", "type": "number", "default": "64"},
        {"name": "dropout", "type": "number", "default": "0.0"},
        {"name": "recurrent_dropout", "type": "number", "default": "0.0"},
        {"name": "post_dropout", "type": "number", "default": "0.0"},
    ],
    "dense": [
        {"name": "units", "type": "number", "default": "64"},
        {"name": "dropout", "type": "number", "default": "0.0"},
    ],
}


def _render_nn_builder(
    cnn_layers: Any, lstm_layers: Any, dense_layers: Any,
    output_value: Any, cnn_activation: str,
) -> str:
    """Render horizontal accordion NN layer builder."""
    def _parse_layers(raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return [d for d in raw if isinstance(d, dict)]
        if isinstance(raw, str) and raw.strip():
            try:
                import yaml as _yaml  # type: ignore[import]
                parsed = _yaml.safe_load(raw)
                if isinstance(parsed, list):
                    return [d for d in parsed if isinstance(d, dict)]
            except Exception:  # noqa: BLE001
                pass
        return []

    def _layer_card(layer_type: str, idx: int, params: Dict[str, Any]) -> str:
        card_id = f"nn_{layer_type}_{idx}"
        fields_html = ""
        for pdef in _NN_LAYER_PARAMS.get(layer_type, []):
            pname = pdef["name"]
            ptype = pdef["type"]
            pval = str(params.get(pname, pdef["default"]))
            if ptype == "select":
                opts = pdef.get("options", "").split(",")
                opt_html = ""
                for o in opts:
                    sel = "selected" if pval == o else ""
                    opt_html += f'<option value="{_escape_text(o)}" {sel}>{_escape_text(o)}</option>'
                fields_html += (
                    f'<div class="nn-param"><label>{_escape_text(pname)}</label>'
                    f'<select class="nn-p" data-param="{_escape_text(pname)}">{opt_html}</select></div>'
                )
            elif ptype == "number":
                fields_html += (
                    f'<div class="nn-param"><label>{_escape_text(pname)}</label>'
                    f'<input type="number" step="any" class="nn-p" data-param="{_escape_text(pname)}" '
                    f'value="{_escape_text(pval)}" /></div>'
                )
            else:
                fields_html += (
                    f'<div class="nn-param"><label>{_escape_text(pname)}</label>'
                    f'<input type="text" class="nn-p" data-param="{_escape_text(pname)}" '
                    f'value="{_escape_text(pval)}" /></div>'
                )
        return (
            f'<div class="nn-card" data-layer-type="{layer_type}" data-idx="{idx}" id="{card_id}">'
            f'<div class="nn-card-header">'
            f'<span class="nn-card-type">{layer_type.upper()} {idx + 1}</span>'
            f'<button type="button" class="list-btn-sm danger" '
            f'onclick="removeNNCard(this)">-</button>'
            f'</div>'
            f'<div class="nn-card-body">{fields_html}</div>'
            f'</div>'
        )

    cards_html = ""
    cnn_list = _parse_layers(cnn_layers)
    lstm_list = _parse_layers(lstm_layers)
    dense_list = _parse_layers(dense_layers)

    cards_html += '<div class="nn-fixed-card input">INPUT</div>'
    for i, layer in enumerate(cnn_list):
        cards_html += _layer_card("cnn", i, layer)
    for i, layer in enumerate(lstm_list):
        cards_html += _layer_card("lstm", i, layer)
    for i, layer in enumerate(dense_list):
        cards_html += _layer_card("dense", i, layer)
    cards_html += '<div class="nn-fixed-card output">OUTPUT</div>'

    add_btns = (
        '<div class="nn-add-bar">'
        '<span class="muted" style="font-size:0.75rem">Add layer:</span>'
        '<button type="button" class="list-btn-sm primary" style="width:auto;padding:2px 8px" '
        'onclick="addNNCard(\'cnn\')">+ CNN</button>'
        '<button type="button" class="list-btn-sm primary" style="width:auto;padding:2px 8px" '
        'onclick="addNNCard(\'lstm\')">+ LSTM</button>'
        '<button type="button" class="list-btn-sm primary" style="width:auto;padding:2px 8px" '
        'onclick="addNNCard(\'dense\')">+ Dense</button>'
        '</div>'
    )

    return (
        '<div class="nn-builder-wrap">'
        f'<div class="nn-cards-row" id="nn-cards-row">{cards_html}</div>'
        f'{add_btns}'
        f'<textarea name="model.cnn.layers" class="list-hidden-ta" id="ta_nn_cnn" rows="1"></textarea>'
        f'<textarea name="model.lstm.layers" class="list-hidden-ta" id="ta_nn_lstm" rows="1"></textarea>'
        f'<textarea name="model.dense.layers" class="list-hidden-ta" id="ta_nn_dense" rows="1"></textarea>'
        '</div>'
    )


def _render_connections_table(value: Any) -> str:
    """Render multi_database.connections as an editable table with +/- buttons."""
    connections: List[Dict[str, Any]] = []
    if isinstance(value, list):
        connections = [c for c in value if isinstance(c, dict)]
    elif isinstance(value, str) and value.strip():
        try:
            import yaml as _yaml  # type: ignore[import]
            parsed = _yaml.safe_load(value)
            if isinstance(parsed, list):
                connections = [c for c in parsed if isinstance(c, dict)]
        except Exception:  # noqa: BLE001
            pass

    rows_html = ""
    for idx, conn in enumerate(connections):
        tr = conn.get("time_range") or {}
        rows_html += (
            f'<tr class="conn-row" data-idx="{idx}">'
            f'<td><input type="text" class="conn-f" data-field="name" value="{_escape_text(conn.get("name", ""))}" /></td>'
            f'<td><input type="text" class="conn-f" data-field="database_uri" value="{_escape_text(conn.get("database_uri", ""))}" /></td>'
            f'<td><input type="text" class="conn-f" data-field="table_prefix" value="{_escape_text(conn.get("table_prefix", ""))}" /></td>'
            f'<td><input type="date" class="conn-f" data-field="start_date" value="{_escape_text(tr.get("start_date", ""))}" /></td>'
            f'<td><input type="date" class="conn-f" data-field="end_date" value="{_escape_text(tr.get("end_date", ""))}" /></td>'
            f'<td><button type="button" class="list-btn-sm danger" onclick="removeConnRow(this)">-</button></td>'
            f'</tr>'
        )

    try:
        import yaml as _yaml  # type: ignore[import]
        ta_val = _yaml.safe_dump(connections, default_flow_style=False).strip() if connections else "[]"
    except Exception:  # noqa: BLE001
        ta_val = str(connections)

    return (
        '<div class="conn-table-wrap">'
        '<table class="conn-table"><thead><tr>'
        '<th>Name</th><th>Database URI</th><th>Table Prefix</th>'
        '<th>Start Date</th><th>End Date</th><th></th>'
        '</tr></thead>'
        f'<tbody id="conn-tbody">{rows_html}</tbody></table>'
        '<button type="button" class="list-btn-sm primary" style="margin-top:4px;width:auto;padding:2px 10px" '
        'onclick="addConnRow()">+ Add Connection</button>'
        f'<textarea name="data.multi_database.connections" class="list-hidden-ta" '
        f'id="ta_conn" rows="1">{_escape_text(ta_val)}</textarea>'
        '</div>'
    )


def _render_list_field(full_key: str, value: Any) -> str:
    """Render a list field with +/- buttons and optional selectable options."""
    options = _LIST_FIELD_OPTIONS.get(full_key, [])
    escaped_key = _escape_text(full_key)
    uid = full_key.replace(".", "_")

    current_items: List[str] = []
    if isinstance(value, list):
        current_items = [str(v) for v in value]
    elif isinstance(value, str) and value.strip():
        try:
            import yaml as _yaml  # type: ignore[import]
            parsed = _yaml.safe_load(value)
            if isinstance(parsed, list):
                current_items = [str(v) for v in parsed]
        except Exception:  # noqa: BLE001
            pass

    items_html = ""
    for idx, item in enumerate(current_items):
        items_html += (
            f'<div class="list-item" data-list="{uid}">'
            f'<span class="list-item-text">{_escape_text(item)}</span>'
            f'<button type="button" class="list-btn-sm danger" '
            f'onclick="removeListItem(this,\'{uid}\')">-</button>'
            f'</div>'
        )

    add_html = ""
    if options:
        opt_html = '<option value="">Add...</option>'
        for opt in options:
            opt_html += f'<option value="{_escape_text(opt)}">{_escape_text(opt)}</option>'
        add_html = (
            f'<div class="list-add-row">'
            f'<select id="sel_{uid}" class="list-add-select">{opt_html}</select>'
            f'<button type="button" class="list-btn-sm primary" '
            f'onclick="addListItemFromSelect(\'{uid}\')">+</button>'
            f'</div>'
        )
    else:
        add_html = (
            f'<div class="list-add-row">'
            f'<input type="text" id="inp_{uid}" class="list-add-input" placeholder="value" />'
            f'<button type="button" class="list-btn-sm primary" '
            f'onclick="addListItemFromInput(\'{uid}\')">+</button>'
            f'</div>'
        )

    return (
        f'<div class="list-field-wrap" id="lf_{uid}">'
        f'<div class="list-items" id="items_{uid}">{items_html}</div>'
        f'{add_html}'
        f'<textarea name="{escaped_key}" class="list-hidden-ta" '
        f'id="ta_{uid}" rows="1">{_escape_text(_list_to_yaml(current_items))}</textarea>'
        f'</div>'
    )


def _list_to_yaml(items: List[str]) -> str:
    """Convert a list of string items to YAML list format."""
    if not items:
        return "[]"
    try:
        import yaml as _yaml  # type: ignore[import]
        coerced: List[Any] = []
        for item in items:
            try:
                coerced.append(int(item))
            except ValueError:
                try:
                    coerced.append(float(item))
                except ValueError:
                    coerced.append(item)
        return _yaml.safe_dump(coerced, default_flow_style=True).strip()
    except Exception:  # noqa: BLE001
        return str(items)


def _render_config_fields(
    values: Dict[str, Any],
    missing: List[str],
    schema_fields: List[SchemaField],
) -> str:
    hint_map = _get_hint_map()
    # Group fields by section, then subsection
    sections: "OrderedDict[str, OrderedDict[str, List[SchemaField]]]" = OrderedDict()
    for field in schema_fields:
        if field.section not in sections:
            sections[field.section] = OrderedDict()
        parts = field.dotted_key.split(".") if field.dotted_key else [""]
        subsection = parts[0] if parts else ""
        if subsection not in sections[field.section]:
            sections[field.section][subsection] = []
        sections[field.section][subsection].append(field)

    out: List[str] = []
    for section_name, subsections in sections.items():
        if section_name in _HIDDEN_SECTIONS:
            continue
        out.append(
            f'<details class="cfg-section" open>'
            f'<summary class="cfg-section-title">{_escape_text(section_name)}</summary>'
            f'<div class="cfg-section-body">'
        )
        for sub_name, fields in subsections.items():
            if sub_name:
                out.append(
                    f'<div class="cfg-subsection">'
                    f'<div class="cfg-subsection-title">{_escape_text(sub_name)}</div>'
                )
            else:
                out.append('<div class="cfg-subsection">')

            for field in fields:
                if field.full_key in _HIDDEN_FIELDS:
                    continue
                parts = field.dotted_key.split(".") if field.dotted_key else []
                label = parts[-1] if parts else field.section
                depth = max(len(parts) - 1, 0)
                hint = hint_map.get(field.full_key) or f"Required. Type: {field.field_type}."
                is_missing = field.full_key in missing
                field_class = "cfg-field missing" if is_missing else "cfg-field"
                indent_style = f"margin-left:{depth * 16}px" if depth > 1 else ""
                value = values.get(field.full_key, "")

                if field.field_type == "boolean":
                    checked = "checked" if value else ""
                    input_html = (
                        f'<input type="checkbox" name="{_escape_text(field.full_key)}" '
                        f'value="true" {checked} />'
                    )
                elif field.full_key == "model.cnn.layers":
                    nn_cnn = values.get("model.cnn.layers", "")
                    nn_lstm = values.get("model.lstm.layers", "")
                    nn_dense = values.get("model.dense.layers", "")
                    nn_output = values.get("model.output.type", "")
                    nn_act = str(values.get("model.cnn.activation", "relu"))
                    input_html = _render_nn_builder(nn_cnn, nn_lstm, nn_dense, nn_output, nn_act)
                elif field.full_key in ("model.lstm.layers", "model.dense.layers"):
                    continue
                elif field.full_key in _ASSET_SELECTOR_FIELDS:
                    val_str = _escape_text(value) if value else ""
                    input_html = (
                        f'<div class="asset-select-wrap">'
                        f'<select name="{_escape_text(field.full_key)}" id="asset_sel_{_escape_text(field.full_key)}" '
                        f'class="asset-select">'
                        f'<option value="{val_str}" selected>{val_str or "Loading..."}</option>'
                        f'</select>'
                        f'<button type="button" class="list-btn-sm primary" '
                        f'onclick="loadAssets(\'{_escape_text(field.full_key)}\')">&orarr;</button>'
                        f'</div>'
                    )
                elif field.full_key in _ASSET_LIST_FIELDS:
                    input_html = (
                        f'<div class="asset-list-wrap" id="al_{field.full_key.replace(".", "_")}">'
                        f'{_render_list_field(field.full_key, value)}'
                        f'<button type="button" class="list-btn-sm primary" style="margin-top:2px;width:auto;padding:2px 8px" '
                        f'onclick="loadAssetsForList(\'{field.full_key.replace(".", "_")}\')">&orarr; DB</button>'
                        f'</div>'
                    )
                elif field.full_key == "data.multi_database.connections":
                    input_html = _render_connections_table(value)
                elif field.full_key == "targets.price_classes.boundaries":
                    input_html = _render_price_class_boundaries_widget(values)
                elif field.field_type in {"list"} and field.full_key in _LIST_FIELD_OPTIONS:
                    input_html = _render_list_field(field.full_key, value)
                elif field.field_type in {"list", "dict", "any"}:
                    input_html = (
                        f'<textarea name="{_escape_text(field.full_key)}" rows="2">'
                        f'{_escape_text(value)}</textarea>'
                    )
                elif field.full_key in _DATE_FIELDS:
                    date_val = str(value).strip() if value else ""
                    input_html = (
                        f'<input type="date" name="{_escape_text(field.full_key)}" '
                        f'value="{_escape_text(date_val)}" />'
                    )
                elif field.full_key in _SELECT_OPTIONS:
                    options = list(_SELECT_OPTIONS[field.full_key])
                    if value and str(value) not in options:
                        options = [str(value)] + options
                    option_html = ""
                    if not value:
                        option_html += '<option value="">Select...</option>'
                    for option in options:
                        sel = "selected" if str(value) == option else ""
                        option_html += (
                            f'<option value="{_escape_text(option)}" {sel}>'
                            f'{_escape_text(option)}</option>'
                        )
                    input_html = f'<select name="{_escape_text(field.full_key)}">{option_html}</select>'
                else:
                    step = "1" if field.field_type == "integer" else "any"
                    input_type = "number" if field.field_type in {"integer", "number"} else "text"
                    input_html = (
                        f'<input type="{input_type}" name="{_escape_text(field.full_key)}" '
                        f'step="{step}" value="{_escape_text(value)}" />'
                    )

                out.append(
                    f'<div class="{field_class}" style="{indent_style}">'
                    f'<label>{_escape_text(label)} '
                    f'<span class="hint" data-hint="{_escape_text(hint)}">i</span></label>'
                    f'{input_html}'
                    f'</div>'
                )

            out.append('</div>')  # close cfg-subsection

        out.append('</div></details>')  # close cfg-section-body + cfg-section

    return "".join(out)


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
    chain: Optional[List[Dict[str, Any]]] = None,
    merged_config: Optional[Dict[str, Any]] = None,
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

    # Inheritance chain graph
    chain_html = ""
    if chain and len(chain) > 0:
        chain_html = _render_config_chain_graph(chain)

    # YAML viewer blocks (per-layer overrides + merged effective)
    yaml_viewer_html = ""
    if chain and len(chain) > 0:
        yaml_parts: List[str] = []
        yaml_parts.append('<div class="yaml-viewer-section">')
        yaml_parts.append('<div class="cfg-subsection-title" style="margin-bottom:8px">'
                         'YAML Viewer (read-only)</div>')
        for idx, entry in enumerate(chain):
            entry_data = entry.get("data", {})
            entry_path = entry.get("path", "")
            entry_name = Path(entry_path).name if entry_path else "?"
            tag = " (leaf)" if idx == 0 else " (base)" if idx == len(chain) - 1 else ""
            is_open = "open" if idx == 0 else ""
            yaml_parts.append(
                _render_yaml_block(
                    entry_data,
                    title=f"{entry_name}{tag}",
                    css_id=f"yaml-layer-{idx}",
                ).replace('<details ', f'<details {is_open} ')
                if is_open else
                _render_yaml_block(
                    entry_data,
                    title=f"{entry_name}{tag}",
                    css_id=f"yaml-layer-{idx}",
                )
            )
        if merged_config:
            yaml_parts.append(
                _render_yaml_block(
                    merged_config,
                    title="Effective (merged)",
                    css_id="yaml-merged",
                )
            )
        yaml_parts.append('</div>')
        yaml_viewer_html = "".join(yaml_parts)

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
      {chain_html}
      {yaml_viewer_html}
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
      <div class="run-actions">
        <form hx-post="/ui/start" hx-target="#run-status">
          <input type="hidden" name="config" value="{_escape_text(selected_normalized)}" />
          <button type="submit" class="primary" {start_disabled}>Start Run</button>
          <span class="htmx-indicator">Starting...</span>
        </form>
        <form hx-post="/ui/stop" hx-target="#run-status" hx-confirm="Stop the running job?">
          <input type="hidden" name="confirm" value="yes" />
          <button type="submit" class="danger" {stop_disabled}>Stop Run</button>
          <span class="htmx-indicator">Stopping...</span>
        </form>
      </div>
    </div>
    """




_PIPELINE_STAGES = [
    ("initializing", "Init"),
    ("diagnostics", "Diagnostics"),
    ("snapshot_build", "Snapshot"),
    ("trial", "HPO"),
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


def _render_ui_page_scripts() -> str:
    return '<script src="/static/app.js"></script><script src="/static/charts.js"></script>'


def _render_ui_page(_config: ServerConfig) -> str:
    nn_params_json = _escape_text(json.dumps(_NN_LAYER_PARAMS, separators=(",", ":")))
    scripts_html = _render_ui_page_scripts()
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
    <link rel="stylesheet" href="/static/app.css" />
  </head>
  <body
    data-theme-storage-key="obs-theme"
    data-api-run-state="/api/run-state"
    data-api-assets="/api/config/assets"
    data-nn-params="{nn_params_json}"
  >
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
        <div id="config-panel" hx-get="/ui/config" hx-trigger="load" hx-swap="outerHTML">
          <span class="htmx-indicator">Loading configuration...</span>
        </div>
        <div id="run-control" hx-get="/ui/run-control" hx-trigger="load, every 5s" style="margin-top:16px">
          <span class="htmx-indicator">Starting...</span>
          <span class="htmx-indicator">Stopping...</span>
        </div>
      </div>
      <div id="tab-logs" class="tab-panel">
        <div class="card">
          <div class="card-header"><h3>Logs</h3></div>
          <div class="log-controls">
            <input type="text" id="log-filter" placeholder="Filter logs..." oninput="filterLogs()" />
            <button onclick="toggleAutoScroll()" id="autoscroll-btn">Auto-scroll: ON</button>
          </div>
          <div id="logs-panel" hx-get="/ui/logs" hx-trigger="load, every 2s"
               hx-on::after-swap="if(window._autoScroll!==false){{const el=this.querySelector('.log-viewer');if(el)el.scrollTop=0;}}"></div>
        </div>
      </div>
      <div id="tab-history" class="tab-panel">
        <div class="card" hx-get="/ui/history" hx-trigger="load, every 15s"></div>
      </div>
    </div>
    {scripts_html}
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
        self.wfile.write(body.encode("utf-8", errors="replace"))

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _send_plain(self, body: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8", errors="replace"))

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

        # Static files — no auth required (for htmx.min.js, etc.)
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
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)
            return

        if not self._require_auth():
            return

        if path == "/" or path == "/ui" or path == "/ui/":
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
            stage_class = stage if stage in ("training", "snapshot_build", "evaluation", "trial", "diagnostics") else ""
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

            body = '<div class="card-header"><h3>Duty Cycle</h3></div>'

            duty_history = state.get("duty_cycle_history")
            if isinstance(duty_history, list) and duty_history:
                res_days = self.server_state.config.duty_cycle_chart_resolution_days
                res_seconds = max(res_days, 1) * 86400
                bins: Dict[int, List[float]] = {}
                for entry in duty_history:
                    ts = entry.get("timestamp")
                    med = entry.get("median")
                    if ts is None or med is None:
                        continue
                    bucket = int(float(ts) // res_seconds)
                    bins.setdefault(bucket, []).append(float(med))
                if bins:
                    sorted_buckets = sorted(bins.keys())
                    chart_labels: List[str] = []
                    chart_values: List[str] = []
                    for b in sorted_buckets:
                        ts_label = datetime.fromtimestamp(b * res_seconds).strftime("%Y-%m-%d")
                        vals = bins[b]
                        mean_val = sum(vals) / len(vals)
                        chart_labels.append(ts_label)
                        chart_values.append(f"{mean_val:.4f}")
                    chart_id = "dutyCycleChart"
                    chart_config = {
                        "type": "line",
                        "data": {
                            "labels": chart_labels,
                            "datasets": [{
                                "label": "Median Duty Cycle",
                                "data": [float(v) for v in chart_values],
                                "borderColor": "#3b82f6",
                                "backgroundColor": "#3b82f622",
                                "tension": 0.3,
                                "pointRadius": 2,
                                "borderWidth": 2,
                                "fill": True,
                            }],
                        },
                        "options": {
                            "responsive": True,
                            "maintainAspectRatio": False,
                            "animation": False,
                            "plugins": {"legend": {"display": False}},
                            "scales": {
                                "x": {
                                    "grid": {"color": "rgba(255,255,255,0.08)"},
                                    "ticks": {"color": "#9499ad", "font": {"size": 10}},
                                },
                                "y": {
                                    "grid": {"color": "rgba(255,255,255,0.08)"},
                                    "ticks": {"color": "#9499ad", "font": {"size": 10}},
                                    "title": {"display": True, "text": "Duty Cycle", "color": "#9499ad"},
                                },
                            },
                        },
                    }
                    chart_config_json = json.dumps(chart_config, separators=(",", ":"))
                    body += (
                        f'<canvas id="{chart_id}" style="width:100%;max-height:200px" '
                        f'data-chart-config=\'{chart_config_json}\'></canvas>'
                    )

            body += f'<div class="kv"><span class="k">Min</span><span class="v">{_fmt_metric(state.get("duty_cycle_min"))}</span></div>'
            body += f'<div class="kv"><span class="k">Median</span><span class="v">{_fmt_metric(state.get("duty_cycle_median"))}</span></div>'
            body += f'<div class="kv"><span class="k">P95</span><span class="v">{_fmt_metric(state.get("duty_cycle_p95"))}</span></div>'
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

            run_stage = str(state.get("stage", "idle"))
            is_hpo_phase = run_stage == "trial"
            hpo_na = '<span class="muted">\u2014 (HPO phase)</span>'

            body = '<div class="card-header"><h3>Run Stats</h3></div>'
            body += f'<div class="kv"><span class="k">Start</span><span class="v">{_fmt_time(start_ts)}</span></div>'
            body += f'<div class="kv"><span class="k">Runtime</span><span class="v">{_fmt_duration(runtime)}</span></div>'
            body += f'<div class="kv"><span class="k">Last Update</span><span class="v">{_fmt_time(updated_ts)}</span></div>'
            body += (
                f'<div class="kv"><span class="k">Snapshots</span>'
                f'<span class="v">{state.get("snapshot_chunks_processed", 0)} / {state.get("snapshot_chunks_total", 0)}</span></div>'
            )
            if is_hpo_phase:
                body += f'<div class="kv"><span class="k">Epochs</span><span class="v">{hpo_na}</span></div>'
                body += f'<div class="kv"><span class="k">Batches</span><span class="v">{hpo_na}</span></div>'
                body += f'<div class="kv"><span class="k">Eval</span><span class="v">{hpo_na}</span></div>'
            else:
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

            def _fmt_bytes_short(value: Any) -> str:
                if value is None:
                    return "-"
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    return "-"
                if num <= 0:
                    return "-"
                units = ["B", "KiB", "MiB", "GiB", "TiB"]
                idx = 0
                while num >= 1024.0 and idx < len(units) - 1:
                    num /= 1024.0
                    idx += 1
                return f"{num:.2f}{units[idx]}"
            for t in trials:
                if t.get("status") == "completed" and t.get("value") is not None:
                    v = float(t["value"])
                    if best_value is None or v < best_value:
                        best_value = v
                        best_number = t.get("number")

            param_keys = sorted({k for t in trials for k in (t.get("params") or {})})
            body += '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:0.82rem">'
            body += '<thead><tr style="border-bottom:2px solid var(--border);text-align:left">'
            body += '<th style="padding:5px 8px">#</th><th style="padding:5px 8px">Status</th><th style="padding:5px 8px">Objective Value</th><th style="padding:5px 8px">Duration</th><th style="padding:5px 8px">Phase RSS</th>'
            for pk in param_keys:
                body += f'<th style="padding:5px 8px">{_escape_text(pk)}</th>'
            body += '</tr></thead><tbody>'

            run_status = str(state.get("status", "idle"))
            for t in trials:
                number = t.get("number", "-")
                status = str(t.get("status", "-"))
                if status == "running" and run_status in ("completed", "failed"):
                    status = "abandoned"
                badge_class = status if status in ("completed", "pruned", "failed", "running", "abandoned") else "idle"
                value = t.get("value")
                if status == "running":
                    value_str = "\u2014"
                else:
                    value_str = f"{float(value):.5f}" if value is not None else "-"
                dur = t.get("duration")
                if status == "running":
                    dur_str = "\u2014"
                else:
                    dur_str = f"{int(dur)}s" if dur is not None else "-"

                phase_summary = "-"
                phase_max_raw = t.get("phase_memory_max_by_phase")
                if isinstance(phase_max_raw, dict) and phase_max_raw:
                    phase_aliases = [
                        ("after_snapshot_load", "snap"),
                        ("after_normalization_stats", "norm"),
                        ("after_long_term_features", "lt"),
                        ("before_first_fit_batch", "prefit"),
                        ("after_first_batch", "batch1"),
                    ]
                    remaining_keys = {
                        str(key)
                        for key in phase_max_raw
                        if isinstance(key, str)
                    }
                    parts: List[str] = []
                    for phase_key, alias in phase_aliases:
                        if phase_key in phase_max_raw:
                            parts.append(f"{alias}:{_fmt_bytes_short(phase_max_raw.get(phase_key))}")
                            if phase_key in remaining_keys:
                                remaining_keys.remove(phase_key)
                    for phase_key in sorted(remaining_keys):
                        parts.append(f"{phase_key}:{_fmt_bytes_short(phase_max_raw.get(phase_key))}")
                    if parts:
                        phase_summary = " | ".join(parts)

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
                body += f'<td style="padding:5px 8px;font-family:var(--mono);font-size:0.72rem">{_escape_text(phase_summary)}</td>'
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

            def _fmt_metric_history(v: Any) -> str:
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
                    f'<td style="padding:6px 8px">{_fmt_metric_history(run.get("final_loss"))}</td>'
                    f'<td style="padding:6px 8px">{_fmt_metric_history(run.get("final_val_loss"))}</td>'
                    f'</tr>'
                )
            body += '</tbody></table></div>'
            self._send_html(body)
            return

        if path == "/ui/training-chart":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            epoch_metrics = state.get("training_epoch_metrics")
            if not isinstance(epoch_metrics, list) or not epoch_metrics:
                run_stage = str(state.get("stage", "idle"))
                if run_stage == "trial":
                    msg = "Training curves available after HPO completes."
                else:
                    msg = "No epoch data yet."
                self._send_html(
                    '<div class="card-header"><h3>Training Curves</h3></div>'
                    f'<div class="muted" style="padding:20px 0;text-align:center">{msg}</div>'
                )
                return
            epochs = [str(m.get("epoch", i + 1)) for i, m in enumerate(epoch_metrics)]
            metric_keys = sorted({k for m in epoch_metrics for k in m if k != "epoch"})
            palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]
            datasets: List[Dict[str, Any]] = []
            for idx, key in enumerate(metric_keys):
                color = palette[idx % len(palette)]
                values = [m.get(key) for m in epoch_metrics]
                is_val = key.startswith("val_")
                dataset: Dict[str, Any] = {
                    "label": str(key),
                    "data": values,
                    "borderColor": color,
                    "backgroundColor": f"{color}22",
                    "tension": 0.3,
                    "pointRadius": 2,
                    "borderWidth": 2,
                    "fill": False,
                }
                if is_val:
                    dataset["borderDash"] = [5, 3]
                datasets.append(dataset)
            chart_id = "trainChart"
            chart_config = {
                "type": "line",
                "data": {"labels": epochs, "datasets": datasets},
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "animation": False,
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "labels": {"color": "#9499ad", "font": {"size": 11}},
                        }
                    },
                    "scales": {
                        "x": {
                            "grid": {"color": "rgba(255,255,255,0.08)"},
                            "ticks": {"color": "#9499ad", "font": {"size": 10}},
                            "title": {"display": True, "text": "Epoch", "color": "#9499ad"},
                        },
                        "y": {
                            "grid": {"color": "rgba(255,255,255,0.08)"},
                            "ticks": {"color": "#9499ad", "font": {"size": 10}},
                        },
                    },
                },
            }
            chart_config_json = json.dumps(chart_config, separators=(",", ":"))
            body = (
                '<div class="card-header"><h3>Training Curves</h3></div>'
                f'<canvas id="{chart_id}" style="width:100%;max-height:320px" '
                f'data-chart-config=\'{chart_config_json}\'></canvas>'
            )
            self._send_html(body)
            return

        if path == "/ui/logs":
            cfg = self.server_state.config
            lines = _tail_log(cfg.run_log_path, max_lines=cfg.tail_max_lines)
            lines.reverse()
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

        if path == "/api/system":
            metrics = self.server_state.get_system_metrics()
            self._send_json(metrics)
            return

        if path == "/api/history":
            history = load_run_history(self.server_state.config.run_state_path, limit=50)
            self._send_json({"runs": history})
            return

        if path == "/api/hpo":
            state = load_run_state(self.server_state.config.run_state_path) or {}
            trials = state.get("hpo_trial_results")
            self._send_json({
                "trials": trials if isinstance(trials, list) else [],
                "hpo_trials_total": state.get("hpo_trials_total", 0),
                "hpo_trials_completed": state.get("hpo_trials_completed", 0),
                "hpo_trials_pruned": state.get("hpo_trials_pruned", 0),
                "hpo_trials_failed": state.get("hpo_trials_failed", 0),
            })
            return

        if path == "/api/config/assets":
            qs = parse_qs(urlparse(self.path).query)
            db_uri = qs.get("db_uri", [""])[0]
            table_prefix = qs.get("table_prefix", ["orderbook_"])[0]
            if not db_uri:
                selected = self.server_state.get_selected_config_path()
                if selected:
                    resolved = _resolve_selected_path(selected)
                    if resolved and resolved.exists():
                        try:
                            cfg_data = _load_training_config(resolved)
                            data_cfg = cfg_data.get("data", {})
                            conn_cfg = data_cfg.get("connection", {})
                            db_uri = str(conn_cfg.get("database_uri", ""))
                            table_prefix = str(conn_cfg.get("table_prefix", table_prefix))
                        except Exception:  # noqa: BLE001
                            pass
            if not db_uri:
                self._send_json({"assets": [], "error": "No database_uri available"})
                return
            try:
                import requests as _requests  # type: ignore[import]
                url = db_uri.rstrip("/") + "/v1/sql"
                resp = _requests.post(
                    url,
                    data={"sql": "SHOW TABLES"},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=(5, 10),
                )
                assets: List[str] = []
                if resp.status_code == 200:
                    result = resp.json()
                    rows = result.get("output", [{}])[0].get("records", {}).get("rows", [])
                    prefix_lower = table_prefix.lower()
                    for row in rows:
                        table_name = str(row[0]) if isinstance(row, (list, tuple)) and row else str(row)
                        if table_name.lower().startswith(prefix_lower):
                            asset = table_name[len(prefix_lower):].upper()
                            if asset:
                                assets.append(asset)
                    assets.sort()
                self._send_json({"assets": assets})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"assets": [], "error": str(exc)})
            return

        if path == "/api/config/list":
            configs = sorted(str(p) for p in _CONFIG_ROOT.rglob("*.yaml"))
            configs += sorted(str(p) for p in _CONFIG_ROOT.rglob("*.yml"))
            rel_configs = []
            for c in sorted(set(configs)):
                try:
                    rel_configs.append(str(Path(c).relative_to(_CONFIG_ROOT.parent)))
                except ValueError:
                    rel_configs.append(c)
            self._send_json({"configs": rel_configs})
            return

        if path == "/api/config/effective":
            qs = parse_qs(urlparse(self.path).query)
            config_path_param = qs.get("path", [""])[0]
            if not config_path_param:
                self._send_json({"error": "path parameter is required"}, status=400)
                return
            resolved = _resolve_selected_path(config_path_param)
            if resolved is None or not resolved.exists():
                self._send_json({"error": "config file not found"}, status=404)
                return
            try:
                effective = _load_training_config(resolved)
                self._send_json(effective)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)
            return

        if path == "/api/config/chain":
            qs = parse_qs(urlparse(self.path).query)
            config_path_param = qs.get("path", [""])[0]
            if not config_path_param:
                self._send_json({"error": "path parameter is required"}, status=400)
                return
            resolved = _resolve_selected_path(config_path_param)
            if resolved is None or not resolved.exists():
                self._send_json({"error": "config file not found"}, status=404)
                return
            try:
                chain = _build_config_inheritance_chain(resolved)
                result = []
                for entry in chain:
                    entry_path = entry.get("path", "")
                    entry_data = entry.get("data", {})
                    filtered = {k: v for k, v in entry_data.items() if k != "base_config"}
                    result.append({
                        "path": entry_path,
                        "name": Path(entry_path).name if entry_path else "",
                        "keys": sorted(filtered.keys()),
                    })
                self._send_json({"chain": result})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        # Validate Content-Length header
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            self._send_plain("Invalid Content-Length header", status=400)
            return

        # Enforce max body size (1 MB)
        MAX_BODY_SIZE = 1_048_576
        if content_length > MAX_BODY_SIZE:
            self._send_plain(f"Request body exceeds {MAX_BODY_SIZE} bytes", status=413)
            return

        # Validate Content-Type for POST requests
        content_type = self.headers.get("Content-Type", "").lower()
        allowed_content_types = {
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }
        if content_length > 0 and not any(content_type.startswith(ct) for ct in allowed_content_types):
            self._send_plain(f"Unsupported Content-Type: {self.headers.get('Content-Type', 'missing')}", status=415)
            return

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
                chain = _build_config_inheritance_chain(resolved)
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
                    chain=chain,
                    merged_config=merged_config,
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
            _apply_price_class_auto_params(config=config_data, params=params, errors=errors)
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
            chain_data: Optional[List[Dict[str, Any]]] = None
            merged_cfg: Optional[Dict[str, Any]] = None
            if selected_path:
                resolved = _resolve_selected_path(selected_path)
                if resolved is None or not resolved.exists():
                    errors = ["Selected config file was not found."]
                    level = "error"
                else:
                    try:
                        chain_data = _build_config_inheritance_chain(resolved)
                        config_data = _load_training_config(resolved)
                        merged_cfg = config_data
                        if mode == "simple":
                            defaults = _load_default_simple_config()
                            merged_cfg = _deep_merge(defaults, config_data)
                        if merged_cfg is None:
                            raise ValueError("Failed to resolve merged config")
                        values = _build_render_values_from_config(merged_cfg, schema_fields)
                        missing = _missing_fields(merged_cfg, schema_fields)
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
                chain=chain_data,
                merged_config=merged_cfg,
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
                status_class = "status-ok" if ok else "status-error"
                self._send_html(f'<div class="{status_class}"><strong>{message}</strong></div>', status=status)
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
                status_class = "status-ok" if ok else "status-error"
                self._send_html(f'<div class="{status_class}"><strong>{message}</strong></div>', status=status)
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
