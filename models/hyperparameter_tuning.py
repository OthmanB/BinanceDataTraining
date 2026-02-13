"""Hyperparameter optimization using Optuna.

This module defines the interface for running hyperparameter optimization
driven entirely by the ``hyperparameter_optimization`` section of the
configuration.
"""

from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import fcntl
import json
import logging
import math
import multiprocessing
import os
import re
import subprocess
import tempfile
import time

from utils.config_loader import ConfigError

logger = logging.getLogger(__name__)


_PHASE_MEMORY_EVENTS_ATTR = "phase_memory_events"
_PHASE_MEMORY_MAX_BY_PHASE_ATTR = "phase_memory_max_by_phase"
_PHASE_MEMORY_EVENTS_LIMIT = 256
_PHASE_MEMORY_EXPORT_LIMIT = 64


def _summarize_trial_states(study: Any) -> Dict[str, int]:
    counts = {
        "completed": 0,
        "pruned": 0,
        "failed": 0,
        "running": 0,
    }
    for trial in study.trials:
        state_name = str(getattr(trial.state, "name", "")).upper()
        if state_name == "COMPLETE":
            counts["completed"] += 1
        elif state_name == "PRUNED":
            counts["pruned"] += 1
        elif state_name in {"FAIL", "FAILED"}:
            counts["failed"] += 1
        elif state_name == "RUNNING":
            counts["running"] += 1
    return counts


def _format_search_space_guidance(search_space: Dict[str, Any]) -> str:
    guidance_lines: List[str] = []
    for name, values in search_space.items():
        if not isinstance(values, list):
            continue
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            low, high = float(values[0]), float(values[1])
            if low == high:
                guidance_lines.append(
                    f"- {name}: currently fixed at {low}. Consider widening the range (e.g. [{low * 0.5}, {low * 1.5}])."
                )
            else:
                guidance_lines.append(
                    f"- {name}: range [{low}, {high}]. Consider widening or shifting based on observed failures."
                )
        elif len(values) > 2:
            guidance_lines.append(
                f"- {name}: categorical candidates {values}. Consider adding smaller/larger values."
            )
    if not guidance_lines:
        return "- search_space: no numeric ranges detected; add or widen numeric ranges for key parameters."
    return "\n".join(guidance_lines)


def _normalize_phase_memory_events(raw_events: Any, *, limit: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not isinstance(raw_events, list):
        return events

    for item in raw_events[-max(0, int(limit)) :]:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "").strip()
        if not phase:
            continue

        normalized: Dict[str, Any] = {"phase": phase}

        timestamp_value = item.get("timestamp")
        try:
            if timestamp_value is not None:
                normalized["timestamp"] = float(timestamp_value)
        except (TypeError, ValueError):
            pass

        rss_value = item.get("rss_bytes")
        try:
            if rss_value is not None:
                normalized["rss_bytes"] = float(rss_value)
        except (TypeError, ValueError):
            pass

        resource_value = item.get("resource")
        if resource_value is not None:
            normalized["resource"] = str(resource_value)

        details_value = item.get("details")
        if isinstance(details_value, dict):
            details: Dict[str, Any] = {}
            for key, value in details_value.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    details[str(key)] = value
            if details:
                normalized["details"] = details

        events.append(normalized)

    return events


def _normalize_phase_memory_max(raw_max: Any) -> Dict[str, float]:
    max_by_phase: Dict[str, float] = {}
    if not isinstance(raw_max, dict):
        return max_by_phase

    for key, value in raw_max.items():
        phase = str(key).strip()
        if not phase:
            continue
        try:
            rss_bytes = float(value)
        except (TypeError, ValueError):
            continue
        if rss_bytes <= 0:
            continue
        max_by_phase[phase] = rss_bytes

    return max_by_phase


def _record_trial_phase_memory_event(
    trial: Any,
    *,
    phase: str,
    resource: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> None:
    phase_name = str(phase).strip()
    if not phase_name or not hasattr(trial, "set_user_attr"):
        return

    rss_bytes = _read_process_rss_bytes(os.getpid())
    event: Dict[str, Any] = {
        "phase": phase_name,
        "timestamp": float(time.time()),
    }
    if rss_bytes is not None and rss_bytes > 0:
        event["rss_bytes"] = float(rss_bytes)
    if resource is not None:
        event["resource"] = str(resource)
    if isinstance(details, dict):
        sanitized_details: Dict[str, Any] = {}
        for key, value in details.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized_details[str(key)] = value
        if sanitized_details:
            event["details"] = sanitized_details

    existing_events = _normalize_phase_memory_events(
        getattr(trial, "user_attrs", {}).get(_PHASE_MEMORY_EVENTS_ATTR),
        limit=max(0, _PHASE_MEMORY_EVENTS_LIMIT - 1),
    )
    existing_events.append(event)
    trial.set_user_attr(_PHASE_MEMORY_EVENTS_ATTR, existing_events[-_PHASE_MEMORY_EVENTS_LIMIT:])

    max_by_phase = _normalize_phase_memory_max(getattr(trial, "user_attrs", {}).get(_PHASE_MEMORY_MAX_BY_PHASE_ATTR))
    if rss_bytes is not None and rss_bytes > 0:
        current_max = max_by_phase.get(phase_name)
        if current_max is None or float(rss_bytes) > float(current_max):
            max_by_phase[phase_name] = float(rss_bytes)
    trial.set_user_attr(_PHASE_MEMORY_MAX_BY_PHASE_ATTR, max_by_phase)


def _extract_trial_details(study: Any, *, max_trials: int = 200) -> List[Dict[str, Any]]:
    """Extract per-trial details from an Optuna study for observability."""
    results: List[Dict[str, Any]] = []
    for trial in study.trials[-max_trials:]:
        state_name = str(getattr(trial.state, "name", "UNKNOWN")).upper()
        if state_name == "COMPLETE":
            status = "completed"
        elif state_name == "PRUNED":
            status = "pruned"
        elif state_name in {"FAIL", "FAILED"}:
            status = "failed"
        elif state_name == "RUNNING":
            status = "running"
        else:
            status = state_name.lower()
        duration: Optional[float] = None
        if trial.datetime_start is not None and trial.datetime_complete is not None:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        entry: Dict[str, Any] = {
            "number": trial.number,
            "status": status,
            "value": trial.value if trial.value is not None else None,
            "duration": duration,
            "params": dict(trial.params) if trial.params else {},
        }

        user_attrs = getattr(trial, "user_attrs", {})
        if isinstance(user_attrs, dict):
            phase_events = _normalize_phase_memory_events(
                user_attrs.get(_PHASE_MEMORY_EVENTS_ATTR),
                limit=_PHASE_MEMORY_EXPORT_LIMIT,
            )
            if phase_events:
                entry[_PHASE_MEMORY_EVENTS_ATTR] = phase_events

            phase_max = _normalize_phase_memory_max(user_attrs.get(_PHASE_MEMORY_MAX_BY_PHASE_ATTR))
            if phase_max:
                entry[_PHASE_MEMORY_MAX_BY_PHASE_ATTR] = phase_max

        results.append(entry)
    return results


def _default_regime_memory() -> Dict[str, Any]:
    return {
        "version": 1,
        "global": {
            "success_count": 0,
            "failure_count": 0,
            "max_success_batch": None,
            "min_oom_batch": None,
        },
        "by_signature": {},
    }


def _resolve_regime_settings(hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    regime_cfg = hpo_cfg.get("regime")
    if not isinstance(regime_cfg, dict):
        return {
            "enabled": False,
            "retry_on_oom": True,
            "max_retry_attempts": 2,
            "batch_backoff_factor": 0.5,
            "min_batch_size": 4,
            "failure_policy": "prune",
            "failure_penalty_value": 1_000_000.0,
            "max_vram_fraction": None,
            "vram_penalty_weight": 1.0,
            "low_utilization_penalty_enabled": False,
            "low_utilization_min_samples_per_second": 0.0,
            "low_utilization_penalty_weight": 0.0,
            "safe_envelope_enabled": True,
            "safe_envelope_min_success_trials": 2,
            "safe_envelope_headroom_fraction": 1.0,
            "safe_envelope_persistence_enabled": True,
        }

    enabled = bool(regime_cfg.get("enabled", False))
    retry_on_oom = bool(regime_cfg.get("retry_on_oom", True))
    max_retry_attempts = int(regime_cfg.get("max_retry_attempts", 2))
    if max_retry_attempts < 0:
        raise ValueError("hyperparameter_optimization.regime.max_retry_attempts must be >= 0")

    batch_backoff_factor = float(regime_cfg.get("batch_backoff_factor", 0.5))
    if batch_backoff_factor <= 0 or batch_backoff_factor >= 1:
        raise ValueError("hyperparameter_optimization.regime.batch_backoff_factor must be in (0, 1)")

    min_batch_size = int(regime_cfg.get("min_batch_size", 4))
    if min_batch_size < 1:
        raise ValueError("hyperparameter_optimization.regime.min_batch_size must be >= 1")

    failure_policy = str(regime_cfg.get("failure_policy", "prune")).strip().lower()
    if failure_policy not in {"prune", "penalize"}:
        raise ValueError("hyperparameter_optimization.regime.failure_policy must be 'prune' or 'penalize'")

    failure_penalty_value = float(regime_cfg.get("failure_penalty_value", 1_000_000.0))
    if failure_penalty_value <= 0:
        raise ValueError("hyperparameter_optimization.regime.failure_penalty_value must be > 0")

    max_vram_fraction_raw = regime_cfg.get("max_vram_fraction")
    max_vram_fraction: Optional[float]
    if max_vram_fraction_raw is None:
        max_vram_fraction = None
    else:
        max_vram_fraction = float(max_vram_fraction_raw)
        if max_vram_fraction <= 0 or max_vram_fraction > 1:
            raise ValueError("hyperparameter_optimization.regime.max_vram_fraction must be in (0, 1]")

    vram_penalty_weight = float(regime_cfg.get("vram_penalty_weight", 1.0))
    if vram_penalty_weight < 0:
        raise ValueError("hyperparameter_optimization.regime.vram_penalty_weight must be >= 0")

    low_util_cfg = regime_cfg.get("low_utilization_penalty")
    if low_util_cfg is None:
        low_util_cfg = {}
    if not isinstance(low_util_cfg, dict):
        raise ValueError("hyperparameter_optimization.regime.low_utilization_penalty must be a mapping")
    low_utilization_penalty_enabled = bool(low_util_cfg.get("enabled", False))
    low_utilization_min_samples_per_second = float(low_util_cfg.get("min_samples_per_second", 0.0))
    if low_utilization_min_samples_per_second < 0:
        raise ValueError(
            "hyperparameter_optimization.regime.low_utilization_penalty.min_samples_per_second must be >= 0"
        )
    low_utilization_penalty_weight = float(low_util_cfg.get("weight", 0.0))
    if low_utilization_penalty_weight < 0:
        raise ValueError("hyperparameter_optimization.regime.low_utilization_penalty.weight must be >= 0")

    safe_envelope_cfg = regime_cfg.get("safe_envelope")
    if safe_envelope_cfg is None:
        safe_envelope_cfg = {}
    if not isinstance(safe_envelope_cfg, dict):
        raise ValueError("hyperparameter_optimization.regime.safe_envelope must be a mapping")
    safe_envelope_enabled = bool(safe_envelope_cfg.get("enabled", True))
    safe_envelope_min_success_trials = int(safe_envelope_cfg.get("min_success_trials", 2))
    if safe_envelope_min_success_trials < 1:
        raise ValueError("hyperparameter_optimization.regime.safe_envelope.min_success_trials must be >= 1")
    safe_envelope_headroom_fraction = float(safe_envelope_cfg.get("headroom_fraction", 1.0))
    if safe_envelope_headroom_fraction <= 0 or safe_envelope_headroom_fraction > 1:
        raise ValueError(
            "hyperparameter_optimization.regime.safe_envelope.headroom_fraction must be in (0, 1]"
        )
    safe_envelope_persistence_enabled = bool(safe_envelope_cfg.get("persistence_enabled", True))

    return {
        "enabled": enabled,
        "retry_on_oom": retry_on_oom,
        "max_retry_attempts": max_retry_attempts,
        "batch_backoff_factor": batch_backoff_factor,
        "min_batch_size": min_batch_size,
        "failure_policy": failure_policy,
        "failure_penalty_value": failure_penalty_value,
        "max_vram_fraction": max_vram_fraction,
        "vram_penalty_weight": vram_penalty_weight,
        "low_utilization_penalty_enabled": low_utilization_penalty_enabled,
        "low_utilization_min_samples_per_second": low_utilization_min_samples_per_second,
        "low_utilization_penalty_weight": low_utilization_penalty_weight,
        "safe_envelope_enabled": safe_envelope_enabled,
        "safe_envelope_min_success_trials": safe_envelope_min_success_trials,
        "safe_envelope_headroom_fraction": safe_envelope_headroom_fraction,
        "safe_envelope_persistence_enabled": safe_envelope_persistence_enabled,
    }


def _sanitize_study_name(study_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", study_name)
    return cleaned or "binance_hpo"


def _build_regime_memory_path(base_config: Dict[str, Any], study_name: str) -> str:
    local_tmp_dir = str(base_config["mlflow"]["local_tmp_dir"])
    optuna_dir = os.path.join(local_tmp_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    return os.path.join(optuna_dir, f"regime_memory_{_sanitize_study_name(study_name)}.json")


_REGIME_MEMORY_WRITE_FAILURES: set[str] = set()


def _read_regime_memory(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _default_regime_memory()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            logger.debug(
                "Invalid regime memory payload type at %s (type=%s); using defaults",
                path,
                type(payload).__name__,
            )
            return _default_regime_memory()
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read regime memory at %s: %s; using defaults", path, exc)
        return _default_regime_memory()


def _build_model_signature(params: Dict[str, Any]) -> str:
    parts: List[str] = []
    # CNN layer params
    cnn_params = {k: v for k, v in params.items() if k.startswith("cnn_layer_")}
    for k in sorted(cnn_params):
        parts.append(f"{k}={int(cnn_params[k])}")
    # LSTM layer params
    lstm_params = {k: v for k, v in params.items() if k.startswith("lstm_layer_")}
    for k in sorted(lstm_params):
        parts.append(f"{k}={int(lstm_params[k])}")
    return ";".join(parts) if parts else "no_arch_params"


def _update_regime_memory(
    path: str,
    *,
    signature: str,
    successful_batch: Optional[int] = None,
    oom_batch: Optional[int] = None,
) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read().strip()
                if raw:
                    try:
                        payload = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to parse regime memory JSON at %s; resetting to defaults", path)
                        payload = _default_regime_memory()
                else:
                    payload = _default_regime_memory()
                if not isinstance(payload, dict):
                    logger.debug(
                        "Invalid regime memory payload type at %s (type=%s); resetting to defaults",
                        path,
                        type(payload).__name__,
                    )
                    payload = _default_regime_memory()

                payload.setdefault("global", {})
                payload.setdefault("by_signature", {})

                global_state = payload["global"]
                signature_state = payload["by_signature"].get(signature)
                if not isinstance(signature_state, dict):
                    signature_state = {
                        "success_count": 0,
                        "failure_count": 0,
                        "max_success_batch": None,
                        "min_oom_batch": None,
                    }

                if successful_batch is not None:
                    signature_state["success_count"] = int(signature_state.get("success_count", 0)) + 1
                    global_state["success_count"] = int(global_state.get("success_count", 0)) + 1
                    current_sig_max = signature_state.get("max_success_batch")
                    current_global_max = global_state.get("max_success_batch")
                    signature_state["max_success_batch"] = (
                        successful_batch
                        if current_sig_max is None
                        else max(int(current_sig_max), successful_batch)
                    )
                    global_state["max_success_batch"] = (
                        successful_batch
                        if current_global_max is None
                        else max(int(current_global_max), successful_batch)
                    )

                if oom_batch is not None:
                    signature_state["failure_count"] = int(signature_state.get("failure_count", 0)) + 1
                    global_state["failure_count"] = int(global_state.get("failure_count", 0)) + 1
                    current_sig_min = signature_state.get("min_oom_batch")
                    current_global_min = global_state.get("min_oom_batch")
                    signature_state["min_oom_batch"] = (
                        oom_batch
                        if current_sig_min is None
                        else min(int(current_sig_min), oom_batch)
                    )
                    global_state["min_oom_batch"] = (
                        oom_batch
                        if current_global_min is None
                        else min(int(current_global_min), oom_batch)
                    )

                payload["by_signature"][signature] = signature_state
                payload["global"] = global_state

                handle.seek(0)
                handle.truncate(0)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:  # noqa: BLE001
        if path not in _REGIME_MEMORY_WRITE_FAILURES:
            logger.warning("Failed to update regime memory at %s: %s", path, exc)
            _REGIME_MEMORY_WRITE_FAILURES.add(path)
        else:
            logger.debug("Failed to update regime memory at %s: %s", path, exc)
        return


def _compute_safe_batch_cap_from_memory(
    payload: Dict[str, Any],
    *,
    signature: str,
    min_batch_size: int,
    min_success_trials: int,
    headroom_fraction: float,
) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    global_state = payload.get("global")
    by_signature = payload.get("by_signature")
    if not isinstance(global_state, dict):
        global_state = {}
    signature_state = {}
    if isinstance(by_signature, dict):
        candidate_state = by_signature.get(signature)
        if isinstance(candidate_state, dict):
            signature_state = candidate_state

    caps: list[int] = []

    def _ingest_oom_cap(state: Dict[str, Any]) -> None:
        min_oom_batch = state.get("min_oom_batch")
        if min_oom_batch is None:
            return
        cap = int(min_oom_batch) - 1
        if cap >= min_batch_size:
            caps.append(cap)

    def _ingest_success_cap(state: Dict[str, Any]) -> None:
        success_count = int(state.get("success_count", 0))
        if success_count < min_success_trials:
            return
        max_success_batch = state.get("max_success_batch")
        if max_success_batch is None:
            return
        cap = int(max_success_batch)
        if cap >= min_batch_size:
            caps.append(cap)

    _ingest_oom_cap(global_state)
    _ingest_oom_cap(signature_state)
    _ingest_success_cap(global_state)
    _ingest_success_cap(signature_state)

    if not caps:
        return None
    cap = min(caps)
    cap = int(math.floor(float(cap) * headroom_fraction))
    return max(min_batch_size, cap)


def _is_resource_exhaustion_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    patterns = [
        "resourceexhausted",
        "out of memory",
        "oom",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "no dnn in stream executor",
    ]
    return any(pattern in message for pattern in patterns)


def _is_configuration_error(exc: Exception) -> bool:
    """Heuristic for config/validation failures vs resource pressure.

    In regime mode we want to distinguish deterministic configuration problems
    (which cannot be fixed by batch backoff) from OOM/resource errors.
    """

    try:
        from utils.config_loader import ConfigError

        if isinstance(exc, ConfigError):
            return True
    except Exception:
        pass

    message = str(exc)
    if "model.output.num_classes must equal len(targets.price_classes.boundaries) + 1" in message:
        return True
    if "For two_head_intensity, model.output.num_classes must equal len(targets.price_classes.boundaries) + 1" in message:
        return True

    return False


def _try_import_mlflow() -> Optional[Any]:
    try:
        import mlflow  # type: ignore[import]

        return mlflow
    except Exception:  # noqa: BLE001
        return None


def _configure_mlflow_from_config(config: Dict[str, Any]) -> Optional[Any]:
    """Configure MLflow client for this process.

    Returns the imported mlflow module, or None if mlflow is unavailable.
    """

    mlflow = _try_import_mlflow()
    if mlflow is None:
        return None

    try:
        mlflow_cfg = config.get("mlflow")
        if not isinstance(mlflow_cfg, dict):
            return mlflow
        tracking_uri = mlflow_cfg.get("tracking_uri")
        experiment_name = mlflow_cfg.get("experiment_name")
        if tracking_uri:
            mlflow.set_tracking_uri(str(tracking_uri))
        if experiment_name:
            mlflow.set_experiment(str(experiment_name))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to configure MLflow client in worker process", exc_info=True)
    return mlflow


def _start_hpo_trial_mlflow_run(
    *,
    config: Dict[str, Any],
    study_name: str,
    trial_number: int,
    resource: Optional[str],
    parent_run_id: Optional[str],
) -> Optional[Any]:
    """Start an MLflow run for an HPO trial.

    Without an explicit run, calls to mlflow.log_* will implicitly create a run
    with an auto-generated name, which is confusing under parallel HPO.
    """

    mlflow_cfg = config.get("mlflow")
    if not isinstance(mlflow_cfg, dict):
        return None
    if not mlflow_cfg.get("tracking_uri") or not mlflow_cfg.get("experiment_name"):
        # Avoid creating implicit/local runs when MLflow isn't configured.
        return None

    mlflow = _configure_mlflow_from_config(config)
    if mlflow is None:
        return None

    target_asset = None
    try:
        data_cfg = config.get("data")
        if isinstance(data_cfg, dict):
            asset_pairs = data_cfg.get("asset_pairs")
            if isinstance(asset_pairs, dict):
                target_asset = asset_pairs.get("target_asset")
    except Exception:
        target_asset = None

    safe_resource = str(resource or "cpu").replace(":", "_")
    run_name = f"{target_asset or 'asset'}_hpo_trial_{int(trial_number)}_{safe_resource}"

    tags: Dict[str, Any] = {
        "run_type": "hpo_trial",
        "hpo.study_name": str(study_name),
        "hpo.trial_number": str(int(trial_number)),
        "hpo.resource": str(resource or ""),
    }

    active = None
    try:
        active = mlflow.active_run()
    except Exception:
        active = None

    # When a parent run is already active (sequential HPO), we must start a
    # nested run. In parallel workers there is no active run, so we link to the
    # supervisor run via parent_run_id.
    try:
        if active is not None:
            return mlflow.start_run(run_name=run_name, nested=True, tags=tags)
        if parent_run_id:
            return mlflow.start_run(run_name=run_name, parent_run_id=str(parent_run_id), tags=tags)
        return mlflow.start_run(run_name=run_name, tags=tags)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to start MLflow run for HPO trial", exc_info=True)
        return None


def _compute_next_batch_size(current_batch: int, factor: float, min_batch_size: int) -> int:
    candidate = int(math.floor(float(current_batch) * factor))
    if candidate >= current_batch:
        candidate = current_batch - 1
    if candidate < min_batch_size:
        candidate = min_batch_size
    return candidate


def _resolve_failure_objective_value(direction: str, penalty: float) -> float:
    if direction == "maximize":
        return -abs(penalty)
    return abs(penalty)


def _apply_objective_penalties(
    *,
    raw_metric: float,
    direction: str,
    regime_settings: Dict[str, Any],
    vram_fraction: Optional[float],
    samples_per_second_est: Optional[float],
) -> tuple[float, float]:
    penalty_total = 0.0

    max_vram_fraction = regime_settings.get("max_vram_fraction")
    if max_vram_fraction is not None and vram_fraction is not None and vram_fraction > float(max_vram_fraction):
        overflow = float(vram_fraction) - float(max_vram_fraction)
        penalty_total += overflow * float(regime_settings["vram_penalty_weight"])

    if bool(regime_settings["low_utilization_penalty_enabled"]):
        min_rate = float(regime_settings["low_utilization_min_samples_per_second"])
        weight = float(regime_settings["low_utilization_penalty_weight"])
        if min_rate > 0 and weight > 0 and samples_per_second_est is not None and samples_per_second_est < min_rate:
            deficit = (min_rate - samples_per_second_est) / min_rate
            penalty_total += deficit * weight

    if direction == "maximize":
        return raw_metric - penalty_total, penalty_total
    return raw_metric + penalty_total, penalty_total


def _try_read_gpu_memory_stats(resource: Optional[str]) -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_memory_fraction": None,
    }
    normalized = str(resource or "").strip().lower()
    if not normalized.startswith("gpu:"):
        return stats

    gpu_id = normalized.split(":", 1)[1].strip()
    if not gpu_id:
        return stats

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        line = completed.stdout.strip().splitlines()[0]
        used_mb_text, total_mb_text = [part.strip() for part in line.split(",", 1)]
        used_bytes = float(used_mb_text) * 1024 * 1024
        total_bytes = float(total_mb_text) * 1024 * 1024
        stats["gpu_memory_used_bytes"] = used_bytes
        stats["gpu_memory_total_bytes"] = total_bytes
        if total_bytes > 0:
            stats["gpu_memory_fraction"] = used_bytes / total_bytes
        return stats
    except Exception:  # noqa: BLE001
        return stats


def _cleanup_trial_runtime() -> None:
    """Best-effort cleanup between HPO trials to limit process RSS growth."""
    try:
        import tensorflow as tf  # type: ignore[import]

        tf.keras.backend.clear_session()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to clear TensorFlow session after trial: %s", exc)

    try:
        import gc

        gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to run GC collection after trial: %s", exc)


def _read_process_rss_bytes(pid: int) -> Optional[int]:
    status_path = f"/proc/{pid}/status"
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read VmRSS for pid %s: %s", pid, exc)
        return None
    return None


def _collect_wave_worker_rss_bytes(executor: ProcessPoolExecutor) -> Dict[int, int]:
    rss_by_pid: Dict[int, int] = {}
    processes = getattr(executor, "_processes", {})
    if not isinstance(processes, dict):
        return rss_by_pid

    for pid, proc in processes.items():
        try:
            if proc is None or not proc.is_alive():
                continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to inspect worker process %s: %s", pid, exc)
            continue
        rss = _read_process_rss_bytes(int(pid))
        if rss is not None and rss > 0:
            rss_by_pid[int(pid)] = int(rss)

    return rss_by_pid


def _should_trigger_rss_watchdog(
    *,
    rss_watchdog_enabled: bool,
    rss_by_pid: Dict[int, int],
    rss_watchdog_limit_bytes: int,
    wave_started_monotonic: float,
    now_monotonic: float,
    startup_grace_seconds: float,
    require_trial_start: bool,
    trial_started_in_wave: bool,
    startup_timeout_seconds: float,
) -> Tuple[bool, Optional[int], Optional[int]]:
    if not rss_watchdog_enabled or not rss_by_pid:
        return False, None, None

    hottest_pid, hottest_rss = max(rss_by_pid.items(), key=lambda item: item[1])
    if hottest_rss < rss_watchdog_limit_bytes:
        return False, None, None

    elapsed = max(0.0, float(now_monotonic) - float(wave_started_monotonic))
    if elapsed < startup_grace_seconds:
        return False, None, None

    if require_trial_start and not trial_started_in_wave:
        if startup_timeout_seconds > 0 and elapsed >= startup_timeout_seconds:
            return True, int(hottest_pid), int(hottest_rss)
        return False, None, None

    return True, int(hottest_pid), int(hottest_rss)


def _select_wave_resources(resources: List[str], remaining_trials: int, force_single_worker: bool) -> List[str]:
    if remaining_trials <= 0:
        return []
    limit = min(len(resources), remaining_trials)
    if limit <= 0:
        return []
    if force_single_worker:
        return resources[:1]
    return resources[:limit]


def _update_adaptive_scheduler_state(
    *,
    current_worker_cap: int,
    current_trials_per_worker_cap: int,
    max_worker_cap: int,
    max_trials_per_worker_cap: int,
    min_worker_cap: int,
    min_trials_per_worker_cap: int,
    stable_waves: int,
    recovery_waves: int,
    wave_had_progress: bool,
    worker_error_count: int,
    watchdog_triggered: bool,
) -> Tuple[int, int, int]:
    """Update adaptive parallel HPO scheduler state after a wave.

    The scheduler scales down worker parallelism and wave trial budgets after
    instability (watchdog trigger, worker errors, or no progress), and scales
    back up after enough stable waves.
    """
    next_worker_cap = int(current_worker_cap)
    next_trials_per_worker_cap = int(current_trials_per_worker_cap)
    next_stable_waves = int(stable_waves)

    unstable_wave = bool(watchdog_triggered) or int(worker_error_count) > 0 or not bool(wave_had_progress)

    if unstable_wave:
        next_stable_waves = 0
        next_worker_cap = max(int(min_worker_cap), int(current_worker_cap) - 1)
        next_trials_per_worker_cap = max(
            int(min_trials_per_worker_cap),
            int(current_trials_per_worker_cap) - 1,
        )
        return next_worker_cap, next_trials_per_worker_cap, next_stable_waves

    next_stable_waves = int(stable_waves) + 1
    if next_stable_waves >= int(recovery_waves):
        next_worker_cap = min(int(max_worker_cap), int(current_worker_cap) + 1)
        next_trials_per_worker_cap = min(
            int(max_trials_per_worker_cap),
            int(current_trials_per_worker_cap) + 1,
        )
        if next_worker_cap != int(current_worker_cap) or next_trials_per_worker_cap != int(current_trials_per_worker_cap):
            next_stable_waves = 0

    return next_worker_cap, next_trials_per_worker_cap, next_stable_waves


def _allocate_trials_to_workers(n_trials: int, n_workers: int) -> list[int]:
    if n_trials <= 0:
        return []
    if n_workers <= 0:
        return []
    base = n_trials // n_workers
    extra = n_trials % n_workers
    allocations: list[int] = []
    for idx in range(n_workers):
        allocations.append(base + (1 if idx < extra else 0))
    return allocations


def _apply_worker_resource(config: Dict[str, Any], resource: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    training_cfg = cfg["training"]
    runtime_cfg = training_cfg["runtime"]

    normalized = str(resource).strip().lower()
    if normalized == "cpu":
        runtime_cfg["device"] = "cpu"
        runtime_cfg["gpu_visible_devices"] = None
    elif normalized.startswith("gpu:"):
        gpu_id = normalized.split(":", 1)[1].strip()
        if not gpu_id:
            raise ValueError("GPU resource must be of form 'gpu:<id>'")
        runtime_cfg["device"] = "gpu"
        runtime_cfg["gpu_visible_devices"] = gpu_id
    else:
        raise ValueError(
            "Unsupported HPO parallel resource. Use 'cpu' or 'gpu:<id>'. "
            f"Got: {resource!r}"
        )

    # Each HPO worker runs on a single resource; disable distributed
    # training to prevent MirroredStrategy deadlocks.
    dist_cfg = runtime_cfg.get("distributed")
    if isinstance(dist_cfg, dict):
        dist_cfg["enabled"] = False

    training_cfg["runtime"] = runtime_cfg
    cfg["training"] = training_cfg
    return cfg


def _resolve_parallel_settings(hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    parallel_cfg = hpo_cfg.get("parallel")
    if not isinstance(parallel_cfg, dict):
        raise ValueError("hyperparameter_optimization.parallel must be a mapping")

    enabled = bool(parallel_cfg.get("enabled", False))
    resources_raw = parallel_cfg.get("resources", [])
    resources: list[str] = []
    if isinstance(resources_raw, list):
        resources = [str(item).strip().lower() for item in resources_raw if str(item).strip()]

    storage_uri_raw = parallel_cfg.get("storage_uri")
    storage_uri: Optional[str] = None
    if storage_uri_raw is not None:
        candidate = str(storage_uri_raw).strip()
        if candidate and candidate.lower() != "null":
            storage_uri = candidate

    study_name_raw = parallel_cfg.get("study_name")
    study_name: Optional[str] = None
    if study_name_raw is not None:
        candidate = str(study_name_raw).strip()
        if candidate:
            study_name = candidate

    max_trials_per_worker_process_raw = parallel_cfg.get("max_trials_per_worker_process")
    if max_trials_per_worker_process_raw is None:
        raise ValueError(
            "hyperparameter_optimization.parallel.max_trials_per_worker_process is required and must be >= 1"
        )
    try:
        max_trials_per_worker_process = int(max_trials_per_worker_process_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.max_trials_per_worker_process must be an integer >= 1"
        ) from exc
    if max_trials_per_worker_process < 1:
        raise ValueError(
            "hyperparameter_optimization.parallel.max_trials_per_worker_process must be >= 1"
        )

    rss_watchdog_enabled = bool(parallel_cfg.get("rss_watchdog_enabled", True))

    rss_watchdog_max_worker_rss_gb_raw = parallel_cfg.get("rss_watchdog_max_worker_rss_gb", 28.0)
    try:
        rss_watchdog_max_worker_rss_gb = float(rss_watchdog_max_worker_rss_gb_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_max_worker_rss_gb must be a number > 0"
        ) from exc
    if rss_watchdog_max_worker_rss_gb <= 0:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_max_worker_rss_gb must be > 0"
        )

    resume_study = bool(parallel_cfg.get("resume_study", False))

    rss_watchdog_check_interval_seconds_raw = parallel_cfg.get("rss_watchdog_check_interval_seconds", 5.0)
    try:
        rss_watchdog_check_interval_seconds = float(rss_watchdog_check_interval_seconds_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_check_interval_seconds must be a number > 0"
        ) from exc
    if rss_watchdog_check_interval_seconds <= 0:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_check_interval_seconds must be > 0"
        )

    rss_watchdog_startup_grace_seconds_raw = parallel_cfg.get("rss_watchdog_startup_grace_seconds", 60.0)
    try:
        rss_watchdog_startup_grace_seconds = float(rss_watchdog_startup_grace_seconds_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_startup_grace_seconds must be a number >= 0"
        ) from exc
    if rss_watchdog_startup_grace_seconds < 0:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_startup_grace_seconds must be >= 0"
        )

    rss_watchdog_require_trial_start = bool(parallel_cfg.get("rss_watchdog_require_trial_start", True))

    rss_watchdog_startup_timeout_seconds_raw = parallel_cfg.get("rss_watchdog_startup_timeout_seconds", 1800.0)
    try:
        rss_watchdog_startup_timeout_seconds = float(rss_watchdog_startup_timeout_seconds_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_startup_timeout_seconds must be a number > 0"
        ) from exc
    if rss_watchdog_startup_timeout_seconds <= 0:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_startup_timeout_seconds must be > 0"
        )

    rss_watchdog_single_worker_fallback_enabled = bool(
        parallel_cfg.get("rss_watchdog_single_worker_fallback_enabled", True)
    )

    rss_watchdog_max_restarts_before_single_worker_raw = parallel_cfg.get(
        "rss_watchdog_max_restarts_before_single_worker",
        2,
    )
    try:
        rss_watchdog_max_restarts_before_single_worker = int(
            rss_watchdog_max_restarts_before_single_worker_raw
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_max_restarts_before_single_worker must be an integer >= 1"
        ) from exc
    if rss_watchdog_max_restarts_before_single_worker < 1:
        raise ValueError(
            "hyperparameter_optimization.parallel.rss_watchdog_max_restarts_before_single_worker must be >= 1"
        )

    adaptive_scheduler_enabled = bool(parallel_cfg.get("adaptive_scheduler_enabled", True))

    adaptive_scheduler_min_workers_raw = parallel_cfg.get("adaptive_scheduler_min_workers", 1)
    try:
        adaptive_scheduler_min_workers = int(adaptive_scheduler_min_workers_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_min_workers must be an integer >= 1"
        ) from exc
    if adaptive_scheduler_min_workers < 1:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_min_workers must be >= 1"
        )

    adaptive_scheduler_recovery_waves_raw = parallel_cfg.get("adaptive_scheduler_recovery_waves", 2)
    try:
        adaptive_scheduler_recovery_waves = int(adaptive_scheduler_recovery_waves_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_recovery_waves must be an integer >= 1"
        ) from exc
    if adaptive_scheduler_recovery_waves < 1:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_recovery_waves must be >= 1"
        )

    adaptive_scheduler_min_trials_per_worker_process_raw = parallel_cfg.get(
        "adaptive_scheduler_min_trials_per_worker_process",
        1,
    )
    try:
        adaptive_scheduler_min_trials_per_worker_process = int(
            adaptive_scheduler_min_trials_per_worker_process_raw
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_min_trials_per_worker_process must be an integer >= 1"
        ) from exc
    if adaptive_scheduler_min_trials_per_worker_process < 1:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_min_trials_per_worker_process must be >= 1"
        )
    if adaptive_scheduler_min_trials_per_worker_process > max_trials_per_worker_process:
        raise ValueError(
            "hyperparameter_optimization.parallel.adaptive_scheduler_min_trials_per_worker_process must be <= parallel.max_trials_per_worker_process"
        )

    return {
        "enabled": enabled,
        "resources": resources,
        "storage_uri": storage_uri,
        "study_name": study_name,
        "max_trials_per_worker_process": max_trials_per_worker_process,
        "resume_study": resume_study,
        "rss_watchdog_enabled": rss_watchdog_enabled,
        "rss_watchdog_max_worker_rss_gb": rss_watchdog_max_worker_rss_gb,
        "rss_watchdog_check_interval_seconds": rss_watchdog_check_interval_seconds,
        "rss_watchdog_startup_grace_seconds": rss_watchdog_startup_grace_seconds,
        "rss_watchdog_require_trial_start": rss_watchdog_require_trial_start,
        "rss_watchdog_startup_timeout_seconds": rss_watchdog_startup_timeout_seconds,
        "rss_watchdog_single_worker_fallback_enabled": rss_watchdog_single_worker_fallback_enabled,
        "rss_watchdog_max_restarts_before_single_worker": rss_watchdog_max_restarts_before_single_worker,
        "adaptive_scheduler_enabled": adaptive_scheduler_enabled,
        "adaptive_scheduler_min_workers": adaptive_scheduler_min_workers,
        "adaptive_scheduler_recovery_waves": adaptive_scheduler_recovery_waves,
        "adaptive_scheduler_min_trials_per_worker_process": adaptive_scheduler_min_trials_per_worker_process,
    }


def _resolve_worker_runtime_options(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    gpu_memory_growth = runtime_cfg.get("gpu_memory_growth")
    if gpu_memory_growth is None:
        enable_memory_growth = False
    elif isinstance(gpu_memory_growth, bool):
        enable_memory_growth = gpu_memory_growth
    else:
        raise ValueError("training.runtime.gpu_memory_growth must be a boolean when provided")

    gpu_allocator_raw = runtime_cfg.get("gpu_allocator")
    gpu_allocator: Optional[str]
    if gpu_allocator_raw is None:
        gpu_allocator = None
    else:
        gpu_allocator = str(gpu_allocator_raw).strip().lower()
        if gpu_allocator not in {"default", "cuda_malloc_async"}:
            raise ValueError(
                "training.runtime.gpu_allocator must be 'default' or 'cuda_malloc_async' when provided"
            )

    gpu_init_lock_enabled = runtime_cfg.get("gpu_init_lock_enabled")
    if gpu_init_lock_enabled is None:
        enable_init_lock = False
    elif isinstance(gpu_init_lock_enabled, bool):
        enable_init_lock = gpu_init_lock_enabled
    else:
        raise ValueError("training.runtime.gpu_init_lock_enabled must be a boolean when provided")

    gpu_init_stagger_seconds = runtime_cfg.get("gpu_init_stagger_seconds")
    if gpu_init_stagger_seconds is None:
        stagger_seconds = 0.0
    else:
        stagger_seconds = float(gpu_init_stagger_seconds)
        if stagger_seconds < 0:
            raise ValueError("training.runtime.gpu_init_stagger_seconds must be >= 0")

    return {
        "gpu_memory_growth": enable_memory_growth,
        "gpu_allocator": gpu_allocator,
        "gpu_init_lock_enabled": enable_init_lock,
        "gpu_init_stagger_seconds": stagger_seconds,
    }


def _configure_worker_tensorflow_runtime(*, enable_memory_growth: bool) -> None:
    if not enable_memory_growth:
        return

    import tensorflow as tf  # type: ignore[import]

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ValueError("GPU resource requested but TensorFlow cannot see visible GPU devices")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _apply_worker_runtime_environment(base_config: Dict[str, Any], resource: str) -> None:
    worker_config = _apply_worker_resource(base_config, resource)
    runtime_cfg = worker_config["training"]["runtime"]
    runtime_options = _resolve_worker_runtime_options(runtime_cfg)

    device = str(runtime_cfg["device"]).strip().lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Worker resource %s configured for CPU (CUDA_VISIBLE_DEVICES='').", resource)
        return

    visible_devices = str(runtime_cfg["gpu_visible_devices"]).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    logger.info(
        "Worker resource %s configured for GPU (CUDA_VISIBLE_DEVICES=%s).",
        resource,
        visible_devices,
    )

    gpu_allocator = runtime_options["gpu_allocator"]
    if gpu_allocator == "cuda_malloc_async":
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        logger.info("Worker resource %s set TF_GPU_ALLOCATOR=cuda_malloc_async.", resource)
    elif gpu_allocator == "default":
        os.environ.pop("TF_GPU_ALLOCATOR", None)

    stagger_seconds = float(runtime_options["gpu_init_stagger_seconds"])
    if stagger_seconds > 0:
        logger.info("Worker resource %s initialization stagger: sleeping %.2fs", resource, stagger_seconds)
        time.sleep(stagger_seconds)

    if bool(runtime_options["gpu_init_lock_enabled"]):
        local_tmp_dir = str(base_config["mlflow"]["local_tmp_dir"])
        lock_dir = os.path.join(local_tmp_dir, "optuna")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, "gpu_init.lock")
        with open(lock_path, "a+", encoding="utf-8") as handle:
            logger.info("Worker resource %s waiting for GPU init lock (%s)", resource, lock_path)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                _configure_worker_tensorflow_runtime(
                    enable_memory_growth=bool(runtime_options["gpu_memory_growth"]),
                )
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        _apply_worker_mixed_precision(runtime_cfg, resource)
        return

    _configure_worker_tensorflow_runtime(
        enable_memory_growth=bool(runtime_options["gpu_memory_growth"]),
    )

    _apply_worker_mixed_precision(runtime_cfg, resource)


def _apply_worker_mixed_precision(runtime_cfg: Dict[str, Any], resource: str) -> None:
    policy_raw = runtime_cfg.get("mixed_precision")
    if policy_raw is None:
        return
    policy = str(policy_raw).strip().lower()
    if policy in {"", "float32"}:
        return
    if policy != "float16":
        logger.warning(
            "Worker resource %s: unsupported mixed_precision=%r, skipping.",
            resource,
            policy,
        )
        return

    import tensorflow as tf  # type: ignore[import]

    tf_policy = f"mixed_{policy}"
    tf.keras.mixed_precision.set_global_policy(tf_policy)
    logger.info(
        "Worker resource %s: mixed precision policy set to '%s'.",
        resource,
        tf_policy,
    )


def _apply_hpo_resume_namespace(
    trial_config: Dict[str, Any],
    *,
    study_name: str,
    trial_number: int,
    resource: Optional[str],
    attempt: int,
) -> Optional[str]:
    training_cfg = trial_config.get("training")
    if not isinstance(training_cfg, dict):
        return None

    sequential_cfg = training_cfg.get("sequential_training")
    if not isinstance(sequential_cfg, dict):
        return None

    if not bool(sequential_cfg.get("enabled", False)):
        return None
    if not bool(sequential_cfg.get("resume_enabled", False)):
        return None

    resource_label = str(resource).strip() if resource is not None else "default"
    namespace = (
        f"hpo__study={study_name}"
        f"__trial={trial_number}"
        f"__attempt={attempt}"
        f"__resource={resource_label}"
    )

    sequential_cfg["resume_namespace"] = namespace
    training_cfg["sequential_training"] = sequential_cfg
    trial_config["training"] = training_cfg
    return namespace


def _build_trial_storage_uri(base_config: Dict[str, Any], storage_uri: Optional[str]) -> str:
    if storage_uri:
        return storage_uri

    local_tmp_dir = str(base_config["mlflow"]["local_tmp_dir"])
    optuna_dir = os.path.join(local_tmp_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    fd, db_path = tempfile.mkstemp(prefix="hpo_", suffix=".db", dir=optuna_dir)
    os.close(fd)
    return f"sqlite:///{db_path}"


def _evaluate_trial_objective(
    base_config: Dict[str, Any],
    data_object: Optional[Dict[str, Any]],
    hpo_cfg: Dict[str, Any],
    metric_name: str,
    direction: str,
    trial_log_models: bool,
    trial: Any,
    resource: Optional[str] = None,
    study_name: str = "binance_hpo",
    parent_mlflow_run_id: Optional[str] = None,
) -> float:
    params = _sample_hyperparameters(trial, hpo_cfg)
    regime_settings = _resolve_regime_settings(hpo_cfg)
    regime_enabled = bool(regime_settings["enabled"])

    signature = _build_model_signature(params)
    memory_path = _build_regime_memory_path(base_config, study_name)

    initial_batch_size = int(params["batch_size"])
    active_batch_size = initial_batch_size

    if regime_enabled and bool(regime_settings["safe_envelope_enabled"]):
        memory_payload = _read_regime_memory(memory_path)
        cap = _compute_safe_batch_cap_from_memory(
            memory_payload,
            signature=signature,
            min_batch_size=int(regime_settings["min_batch_size"]),
            min_success_trials=int(regime_settings["safe_envelope_min_success_trials"]),
            headroom_fraction=float(regime_settings["safe_envelope_headroom_fraction"]),
        )
        if cap is not None and active_batch_size > cap:
            logger.info(
                "Applying safe-envelope batch clamp for trial %s: requested=%s capped=%s signature=%s",
                trial.number,
                active_batch_size,
                cap,
                signature,
            )
            active_batch_size = cap

    from training.pipeline import run_training_pipeline

    trial_mlflow_run = _start_hpo_trial_mlflow_run(
        config=base_config,
        study_name=study_name,
        trial_number=int(trial.number),
        resource=resource,
        parent_run_id=parent_mlflow_run_id,
    )
    if trial_mlflow_run is not None:
        try:
            run_id = getattr(getattr(trial_mlflow_run, "info", None), "run_id", None)
            if run_id:
                trial.set_user_attr("mlflow_run_id", str(run_id))
        except Exception:  # noqa: BLE001
            pass

        try:
            mlflow = _try_import_mlflow()
            if mlflow is not None:
                mlflow.log_param("hpo_study_name", study_name)
                mlflow.log_param("hpo_trial_number", int(trial.number))
                if resource is not None:
                    mlflow.log_param("hpo_resource", str(resource))
                for name, value in params.items():
                    mlflow.log_param(f"hpo_param_{name}", value)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to log initial HPO trial parameters to MLflow", exc_info=True)

    max_retry_attempts = int(regime_settings["max_retry_attempts"]) if regime_enabled else 0
    attempt = 0
    last_exception: Optional[Exception] = None

    try:
        while True:
            params_for_attempt = dict(params)
            params_for_attempt["batch_size"] = int(active_batch_size)

            trial_config = _apply_hyperparameters(base_config, params_for_attempt)
            if resource is not None:
                trial_config = _apply_worker_resource(trial_config, resource)

            def _phase_memory_probe(phase: str, details: Optional[Dict[str, Any]] = None) -> None:
                payload: Dict[str, Any] = {
                    "attempt": int(attempt),
                    "batch_size": int(active_batch_size),
                }
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            payload[str(key)] = value
                _record_trial_phase_memory_event(
                    trial,
                    phase=phase,
                    resource=resource,
                    details=payload,
                )

            trial_config["_hpo_phase_memory_probe"] = _phase_memory_probe

            resume_namespace = _apply_hpo_resume_namespace(
                trial_config,
                study_name=study_name,
                trial_number=int(trial.number),
                resource=resource,
                attempt=attempt,
            )
            if resume_namespace is not None:
                trial.set_user_attr("sequential_resume_namespace", resume_namespace)

            mlflow_cfg = trial_config["mlflow"]
            if not trial_log_models:
                try:
                    artifact_logging_cfg = mlflow_cfg["artifact_logging"]
                    artifact_logging_cfg["trained_model"] = False
                    mlflow_cfg["artifact_logging"] = artifact_logging_cfg

                    model_registry_cfg = mlflow_cfg["model_registry"]
                    model_registry_cfg["register_model"] = False
                    mlflow_cfg["model_registry"] = model_registry_cfg

                    trial_config["mlflow"] = mlflow_cfg
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to adjust MLFlow logging configuration for HPO trial: %s",
                        exc,
                    )

            started = time.monotonic()
            try:
                run_training_pipeline(trial_config, data_object)

                value_obj = trial_config.get("_hpo_last_metric")
                if value_obj is None and isinstance(data_object, dict):
                    metadata = data_object.get("metadata", {})
                    value_obj = metadata.get("last_hpo_metric")
                if value_obj is None:
                    raise ValueError(
                        "Training pipeline did not populate an HPO metric; ensure "
                        "hyperparameter_optimization.metric matches a key in Keras History.",
                    )

                duration_seconds = max(time.monotonic() - started, 1e-9)
                metric_raw = float(value_obj)

                sample_count: Optional[float] = None
                if isinstance(data_object, dict):
                    metadata = data_object.get("metadata")
                    if isinstance(metadata, dict) and metadata.get("num_samples") is not None:
                        try:
                            sample_count = float(metadata["num_samples"])
                        except (TypeError, ValueError):
                            sample_count = None
                if sample_count is None:
                    sample_count = float(active_batch_size)
                samples_per_second_est = sample_count / duration_seconds if duration_seconds > 0 else None

                gpu_stats = _try_read_gpu_memory_stats(resource)
                metric_effective = metric_raw
                penalty_total = 0.0
                if regime_enabled:
                    metric_effective, penalty_total = _apply_objective_penalties(
                        raw_metric=metric_raw,
                        direction=direction,
                        regime_settings=regime_settings,
                        vram_fraction=(
                            float(gpu_stats["gpu_memory_fraction"])
                            if gpu_stats["gpu_memory_fraction"] is not None
                            else None
                        ),
                        samples_per_second_est=samples_per_second_est,
                    )

                trial.set_user_attr("metric_name", metric_name)
                trial.set_user_attr("metric_raw", metric_raw)
                trial.set_user_attr("metric_effective", metric_effective)
                trial.set_user_attr("regime_penalty_total", penalty_total)
                trial.set_user_attr("regime_retry_count", attempt)
                trial.set_user_attr("batch_size_initial", initial_batch_size)
                trial.set_user_attr("batch_size_effective", int(active_batch_size))
                trial.set_user_attr("trial_duration_seconds", duration_seconds)
                trial.set_user_attr("samples_per_second_est", samples_per_second_est)
                trial.set_user_attr("resource", resource)
                if gpu_stats["gpu_memory_used_bytes"] is not None:
                    trial.set_user_attr("gpu_memory_used_bytes", float(gpu_stats["gpu_memory_used_bytes"]))
                if gpu_stats["gpu_memory_total_bytes"] is not None:
                    trial.set_user_attr("gpu_memory_total_bytes", float(gpu_stats["gpu_memory_total_bytes"]))
                if gpu_stats["gpu_memory_fraction"] is not None:
                    trial.set_user_attr("gpu_memory_fraction", float(gpu_stats["gpu_memory_fraction"]))

                if regime_enabled and bool(regime_settings["safe_envelope_persistence_enabled"]):
                    _update_regime_memory(
                        memory_path,
                        signature=signature,
                        successful_batch=int(active_batch_size),
                    )

                return metric_effective
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                is_oom = _is_resource_exhaustion_error(exc)
                is_config_error = _is_configuration_error(exc)
                if regime_enabled and is_oom and bool(regime_settings["safe_envelope_persistence_enabled"]):
                    _update_regime_memory(
                        memory_path,
                        signature=signature,
                        oom_batch=int(active_batch_size),
                    )

                can_retry = (
                    regime_enabled
                    and bool(regime_settings["retry_on_oom"])
                    and is_oom
                    and attempt < max_retry_attempts
                )
                if can_retry:
                    next_batch_size = _compute_next_batch_size(
                        int(active_batch_size),
                        float(regime_settings["batch_backoff_factor"]),
                        int(regime_settings["min_batch_size"]),
                    )
                    if next_batch_size >= int(active_batch_size):
                        logger.warning(
                            "Trial %s resource=%s attempt=%s OOM but batch already at floor (%s); no further backoff possible.",
                            trial.number,
                            resource,
                            attempt + 1,
                            active_batch_size,
                        )
                        can_retry = False
                    else:
                        logger.warning(
                            "Trial %s resource=%s attempt=%s failed with resource pressure; retrying with smaller batch (%s -> %s). Error: %s",
                            trial.number,
                            resource,
                            attempt + 1,
                            active_batch_size,
                            next_batch_size,
                            exc,
                        )
                        active_batch_size = next_batch_size
                        attempt += 1
                        continue

                if regime_enabled:
                    trial.set_user_attr("regime_retry_count", attempt)
                    trial.set_user_attr("batch_size_initial", initial_batch_size)
                    trial.set_user_attr("batch_size_effective", int(active_batch_size))
                    trial.set_user_attr("regime_failure_type", type(exc).__name__)
                    trial.set_user_attr("regime_failure_message", str(exc))
                    trial.set_user_attr("regime_failure_is_oom", bool(is_oom))
                    trial.set_user_attr(
                        "regime_failure_category",
                        "configuration" if is_config_error else ("resource" if is_oom else "other"),
                    )
                    trial.set_user_attr("resource", resource)
                    if str(regime_settings["failure_policy"]) == "penalize":
                        return _resolve_failure_objective_value(
                            direction,
                            float(regime_settings["failure_penalty_value"]),
                        )

                    import optuna  # type: ignore[import]

                    if is_oom:
                        raise optuna.TrialPruned(
                            f"Resource-constrained trial pruned after {attempt} retries: {exc}"
                        ) from exc
                    if is_config_error:
                        raise optuna.TrialPruned(
                            f"Configuration error: {type(exc).__name__}: {exc}"
                        ) from exc
                    raise optuna.TrialPruned(
                        f"Trial pruned due to non-resource error: {type(exc).__name__}: {exc}"
                    ) from exc
                raise
            finally:
                # Ensure model/runtime state is reclaimed between attempts/trials.
                trial_config = None
                _cleanup_trial_runtime()
    finally:
        if trial_mlflow_run is not None:
            try:
                mlflow = _try_import_mlflow()
                if mlflow is not None:
                    mlflow.end_run()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to end MLflow run for HPO trial", exc_info=True)

    if last_exception is not None:
        raise last_exception

    raise ValueError("Hyperparameter objective failed without recording an exception")


def _run_optuna_worker(
    *,
    base_config: Dict[str, Any],
    data_object: Optional[Dict[str, Any]],
    hpo_cfg: Dict[str, Any],
    metric_name: str,
    direction: str,
    trial_log_models: bool,
    storage_uri: str,
    study_name: str,
    resource: str,
    n_trials: int,
    parent_mlflow_run_id: Optional[str] = None,
) -> None:
    if n_trials <= 0:
        return

    _apply_worker_runtime_environment(base_config, resource)

    import optuna  # type: ignore[import]

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        logger.info(
            "Worker pid=%s resource=%s starting trial %s",
            os.getpid(),
            resource,
            trial.number,
        )
        value = _evaluate_trial_objective(
            base_config,
            data_object,
            hpo_cfg,
            metric_name,
            direction,
            trial_log_models,
            trial,
            resource=resource,
            study_name=study_name,
            parent_mlflow_run_id=parent_mlflow_run_id,
        )
        logger.info(
            "Worker pid=%s resource=%s finished trial %s with metric %s",
            os.getpid(),
            resource,
            trial.number,
            value,
        )
        return value

    study.optimize(objective, n_trials=n_trials)


def _sample_hyperparameters(trial: Any, hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    search_space = hpo_cfg["search_space"]

    def _suggest_int(name: str, bounds: Any) -> int:
        if not isinstance(bounds, list) or len(bounds) < 2:
            raise ValueError(f"Integer search space for {name!r} must be a list with at least [low, high]")
        low = int(bounds[0])
        high = int(bounds[1])
        return int(trial.suggest_int(name, low, high))

    def _suggest_float(name: str, bounds: Any) -> float:
        if not isinstance(bounds, list) or len(bounds) < 2:
            raise ValueError(f"Float search space for {name!r} must be a list with at least [low, high]")
        low = float(bounds[0])
        high = float(bounds[1])
        log_scale = False
        if len(bounds) >= 3 and str(bounds[2]).lower() == "log":
            log_scale = True
        return float(trial.suggest_float(name, low, high, log=log_scale))

    params: Dict[str, Any] = {}

    # CNN per-layer search ranges
    cnn_space = search_space.get("cnn", [])
    if not isinstance(cnn_space, list):
        raise ValueError("hyperparameter_optimization.search_space.cnn must be a list")
    for i, layer_space in enumerate(cnn_space):
        if not isinstance(layer_space, dict):
            raise ValueError(f"search_space.cnn[{i}] must be a dict")
        if "filters" in layer_space:
            params[f"cnn_layer_{i}_filters"] = _suggest_int(
                f"cnn_layer_{i}_filters", layer_space["filters"],
            )

    # LSTM per-layer search ranges
    lstm_space = search_space.get("lstm", [])
    if not isinstance(lstm_space, list):
        raise ValueError("hyperparameter_optimization.search_space.lstm must be a list")
    for i, layer_space in enumerate(lstm_space):
        if not isinstance(layer_space, dict):
            raise ValueError(f"search_space.lstm[{i}] must be a dict")
        if "units" in layer_space:
            params[f"lstm_layer_{i}_units"] = _suggest_int(
                f"lstm_layer_{i}_units", layer_space["units"],
            )

    # Scalar search parameters
    if "learning_rate" not in search_space:
        raise ValueError("Missing hyperparameter_optimization.search_space.learning_rate")
    params["learning_rate"] = _suggest_float("learning_rate", search_space["learning_rate"])

    if "batch_size" not in search_space:
        raise ValueError("Missing hyperparameter_optimization.search_space.batch_size")
    batch_size_space = search_space["batch_size"]
    if not isinstance(batch_size_space, list) or len(batch_size_space) < 2:
        raise ValueError("hyperparameter_optimization.search_space.batch_size must have at least two values")
    if len(batch_size_space) >= 3:
        choices = sorted({int(v) for v in batch_size_space})
        params["batch_size"] = int(trial.suggest_categorical("batch_size", choices))
    else:
        params["batch_size"] = _suggest_int("batch_size", batch_size_space)

    return params


def _apply_hyperparameters(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)

    model_cfg = cfg["model"]
    cnn_layers = model_cfg["cnn"]["layers"]
    lstm_layers = model_cfg["lstm"]["layers"]
    compilation_cfg = model_cfg["compilation"]
    training_cfg = cfg["training"]

    # Apply CNN per-layer params
    for key, value in params.items():
        if key.startswith("cnn_layer_") and key.endswith("_filters"):
            idx = int(key.split("_")[2])
            if idx < len(cnn_layers):
                cnn_layers[idx]["filters"] = int(value)
            else:
                raise ValueError(
                    f"search_space references cnn layer {idx} but model.cnn.layers has only {len(cnn_layers)} entries"
                )

    # Apply LSTM per-layer params
    for key, value in params.items():
        if key.startswith("lstm_layer_") and key.endswith("_units"):
            idx = int(key.split("_")[2])
            if idx < len(lstm_layers):
                lstm_layers[idx]["units"] = int(value)
            else:
                raise ValueError(
                    f"search_space references lstm layer {idx} but model.lstm.layers has only {len(lstm_layers)} entries"
                )

    # Scalar params
    compilation_cfg["learning_rate"] = float(params["learning_rate"])
    training_cfg["batch_size"] = int(params["batch_size"])

    model_cfg["compilation"] = compilation_cfg
    cfg["model"] = model_cfg
    cfg["training"] = training_cfg

    return cfg


def _resolve_best_params_for_final_training(best_trial: Any) -> tuple[Dict[str, Any], Optional[int], Optional[int]]:
    """Resolve the best hyperparameters to apply to the final config.

    In regime mode, the sampled batch size may be clamped (safe envelope) or
    backed off (OOM retry). The objective value recorded for a trial therefore
    corresponds to the **effective** batch size, not necessarily the originally
    sampled one. To keep final training consistent with the evaluated trial, we
    prefer the effective batch size when it is available.

    Returns
    -------
    tuple
        (best_params, requested_batch_size, effective_batch_size)
    """

    params_raw = getattr(best_trial, "params", None)
    best_params = dict(params_raw) if isinstance(params_raw, dict) else {}

    requested_batch_size: Optional[int] = None
    if "batch_size" in best_params:
        try:
            requested_batch_size = int(best_params["batch_size"])
        except (TypeError, ValueError):
            requested_batch_size = None

    user_attrs_raw = getattr(best_trial, "user_attrs", None)
    user_attrs: Dict[str, Any] = user_attrs_raw if isinstance(user_attrs_raw, dict) else {}
    effective_batch_size: Optional[int] = None
    if user_attrs.get("batch_size_effective") is not None:
        try:
            effective_batch_size = int(user_attrs["batch_size_effective"])
        except (TypeError, ValueError):
            effective_batch_size = None

    if (
        effective_batch_size is not None
        and effective_batch_size > 0
        and "batch_size" in best_params
    ):
        best_params["batch_size"] = effective_batch_size

    return best_params, requested_batch_size, effective_batch_size


def run_hyperparameter_search(
    config: Dict[str, Any],
    data_object: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    hpo_cfg = config["hyperparameter_optimization"]
    if not bool(hpo_cfg["enabled"]):
        logger.info("Hyperparameter optimization disabled in configuration.")
        return None

    framework = str(hpo_cfg["framework"])
    if framework != "optuna":
        logger.warning("Only hyperparameter_optimization.framework='optuna' is supported; got %s", framework)
        return None

    try:
        import optuna  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import Optuna for hyperparameter optimization: %s", exc)
        return None

    n_trials = int(hpo_cfg["n_trials"])
    direction = str(hpo_cfg["direction"])
    metric_name = str(hpo_cfg["metric"])
    parallel_settings = _resolve_parallel_settings(hpo_cfg)
    _resolve_regime_settings(hpo_cfg)

    writer = None
    try:
        from observability.run_state import get_run_state_writer

        writer = get_run_state_writer()
    except Exception:  # noqa: BLE001
        writer = None
    if writer is not None:
        try:
            writer.set_stage("trial")
            writer.update_hpo_progress(completed=0, total=n_trials, pruned=0, failed=0)
            writer.update_hpo_wave_memory({})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize HPO run-state progress: %s", exc)

    trial_logging_cfg = hpo_cfg["trial_model_logging"]
    trial_log_models = bool(trial_logging_cfg["enabled"])

    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "hyperparameter_optimization.direction must be either 'minimize' or 'maximize'; "
            f"got {direction!r}",
        )

    if "search_space" not in hpo_cfg or not isinstance(hpo_cfg["search_space"], dict):
        raise ValueError("hyperparameter_optimization.search_space must be a dict in configuration")

    use_parallel = bool(parallel_settings["enabled"]) and n_trials > 1 and len(parallel_settings["resources"]) > 0

    if bool(config.get("snapshot", {}).get("enabled", False)):
        if bool(config.get("_snapshot_prebuild_complete", False)):
            logger.info("Snapshot pre-build already completed in current process; skipping duplicate pre-build.")
        else:
            try:
                from training.pipeline import pre_build_snapshots

                logger.info("Pre-building snapshots before hyperparameter search.")
                pre_build_snapshots(config)
                config["_snapshot_prebuild_complete"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Snapshot pre-build failed; workers/trials will build on demand: %s",
                    exc,
                )

    study: Any
    if use_parallel:
        if isinstance(data_object, dict):
            logger.warning(
                "Parallel HPO is enabled but data_object is populated; falling back to sequential HPO."
            )
            use_parallel = False

    if use_parallel:
        resources = list(parallel_settings["resources"])[:n_trials]
        n_workers = len(resources)
        max_trials_per_worker_process = int(parallel_settings["max_trials_per_worker_process"])
        adaptive_scheduler_enabled = bool(parallel_settings["adaptive_scheduler_enabled"])
        adaptive_scheduler_min_workers = int(parallel_settings["adaptive_scheduler_min_workers"])
        adaptive_scheduler_recovery_waves = int(parallel_settings["adaptive_scheduler_recovery_waves"])
        adaptive_scheduler_min_trials_per_worker_process = int(
            parallel_settings["adaptive_scheduler_min_trials_per_worker_process"]
        )
        rss_watchdog_enabled = bool(parallel_settings["rss_watchdog_enabled"])
        rss_watchdog_max_worker_rss_bytes = int(float(parallel_settings["rss_watchdog_max_worker_rss_gb"]) * (1024**3))
        rss_watchdog_check_interval_seconds = float(parallel_settings["rss_watchdog_check_interval_seconds"])
        rss_watchdog_startup_grace_seconds = float(parallel_settings["rss_watchdog_startup_grace_seconds"])
        rss_watchdog_require_trial_start = bool(parallel_settings["rss_watchdog_require_trial_start"])
        rss_watchdog_startup_timeout_seconds = float(parallel_settings["rss_watchdog_startup_timeout_seconds"])
        rss_watchdog_single_worker_fallback_enabled = bool(
            parallel_settings["rss_watchdog_single_worker_fallback_enabled"]
        )
        rss_watchdog_max_restarts_before_single_worker = int(
            parallel_settings["rss_watchdog_max_restarts_before_single_worker"]
        )
        storage_uri = _build_trial_storage_uri(config, parallel_settings["storage_uri"])
        study_name = str(parallel_settings["study_name"] or "binance_hpo")

        resume_study = bool(parallel_settings["resume_study"])

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=True,
        )

        pre_existing_counts = _summarize_trial_states(study)
        pre_existing_finished = (
            int(pre_existing_counts["completed"])
            + int(pre_existing_counts["pruned"])
            + int(pre_existing_counts["failed"])
        )

        if pre_existing_finished > 0:
            if resume_study:
                logger.info(
                    "Resuming existing study '%s' with %d pre-existing trials "
                    "(completed=%d pruned=%d failed=%d). Only %d new trial(s) will be launched.",
                    study_name,
                    pre_existing_finished,
                    pre_existing_counts["completed"],
                    pre_existing_counts["pruned"],
                    pre_existing_counts["failed"],
                    max(0, n_trials - pre_existing_finished),
                )
            else:
                logger.warning(
                    "Existing study '%s' has %d stale trials from a previous run "
                    "(completed=%d pruned=%d failed=%d). Deleting and recreating study "
                    "(set parallel.resume_study=true to keep them).",
                    study_name,
                    pre_existing_finished,
                    pre_existing_counts["completed"],
                    pre_existing_counts["pruned"],
                    pre_existing_counts["failed"],
                )
                optuna.delete_study(study_name=study_name, storage=storage_uri)
                study = optuna.create_study(
                    direction=direction,
                    study_name=study_name,
                    storage=storage_uri,
                )
                pre_existing_finished = 0

        logger.info(
            "Starting parallel HPO with %s workers over %s trials. resources=%s storage=%s study=%s",
            n_workers,
            n_trials,
            resources,
            storage_uri,
            study_name,
        )

        parent_mlflow_run_id = None
        try:
            mlflow = _try_import_mlflow()
            if mlflow is not None:
                active = mlflow.active_run()
                if active is not None:
                    parent_mlflow_run_id = str(active.info.run_id)
        except Exception:  # noqa: BLE001
            parent_mlflow_run_id = None

        adaptive_scheduler_min_workers = min(max(1, adaptive_scheduler_min_workers), max(1, len(resources)))
        adaptive_scheduler_min_trials_per_worker_process = min(
            max(1, adaptive_scheduler_min_trials_per_worker_process),
            max_trials_per_worker_process,
        )
        adaptive_worker_cap = len(resources)
        adaptive_trials_per_worker_cap = max_trials_per_worker_process
        adaptive_stable_waves = 0

        no_progress_cycles = 0
        max_no_progress_cycles = 8
        last_finished = -1
        force_single_worker = False
        watchdog_restarts_without_progress = 0
        spawn_ctx = multiprocessing.get_context("spawn")

        while True:
            refreshed = optuna.load_study(study_name=study_name, storage=storage_uri)
            counts = _summarize_trial_states(refreshed)
            finished_trials = int(counts["completed"]) + int(counts["pruned"]) + int(counts["failed"])
            remaining_trials = max(0, n_trials - finished_trials)

            if writer is not None:
                try:
                    writer.update_hpo_progress(
                        completed=int(counts["completed"]),
                        total=n_trials,
                        pruned=int(counts["pruned"]),
                        failed=int(counts["failed"]),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to update HPO run-state progress: %s", exc)

            if remaining_trials <= 0:
                break

            if finished_trials <= last_finished:
                no_progress_cycles += 1
            else:
                no_progress_cycles = 0
                watchdog_restarts_without_progress = 0
            last_finished = finished_trials

            if no_progress_cycles > max_no_progress_cycles:
                raise RuntimeError(
                    "Parallel HPO made no progress across multiple recovery cycles; "
                    "worker processes may be repeatedly terminating."
                )

            active_resources = resources[:adaptive_worker_cap]
            if force_single_worker:
                active_resources = resources[:1]

            wave_resources = _select_wave_resources(
                active_resources,
                remaining_trials,
                force_single_worker=force_single_worker,
            )
            if not wave_resources:
                raise RuntimeError(
                    "Parallel HPO could not select any worker resources for the current wave. "
                    "Check hyperparameter_optimization.parallel.resources."
                )

            wave_trial_budget = min(
                remaining_trials,
                adaptive_trials_per_worker_cap * len(wave_resources),
            )
            allocations = _allocate_trials_to_workers(wave_trial_budget, len(wave_resources))

            logger.info(
                "Launching parallel HPO wave: remaining_trials=%s wave_budget=%s max_trials_per_worker_process=%s allocations=%s resources=%s force_single_worker=%s adaptive_worker_cap=%s adaptive_trial_cap=%s",
                remaining_trials,
                wave_trial_budget,
                adaptive_trials_per_worker_cap,
                allocations,
                wave_resources,
                force_single_worker,
                adaptive_worker_cap,
                adaptive_trials_per_worker_cap,
            )

            worker_errors: list[str] = []
            watchdog_triggered = False
            wave_started_monotonic = time.monotonic()
            wave_finished_baseline = finished_trials
            with ProcessPoolExecutor(max_workers=len(wave_resources), mp_context=spawn_ctx) as executor:
                futures = []
                for resource, worker_trials in zip(wave_resources, allocations):
                    if worker_trials <= 0:
                        continue
                    futures.append(
                        executor.submit(
                            _run_optuna_worker,
                            base_config=config,
                            data_object=data_object,
                            hpo_cfg=hpo_cfg,
                            metric_name=metric_name,
                            direction=direction,
                            trial_log_models=trial_log_models,
                            storage_uri=storage_uri,
                            study_name=study_name,
                            resource=resource,
                            n_trials=worker_trials,
                            parent_mlflow_run_id=parent_mlflow_run_id,
                        )
                    )

                pending = set(futures)
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=rss_watchdog_check_interval_seconds,
                        return_when=FIRST_COMPLETED,
                    )

                    counts: Optional[Dict[str, int]] = None
                    if writer is not None or rss_watchdog_require_trial_start:
                        try:
                            refreshed = optuna.load_study(study_name=study_name, storage=storage_uri)
                            counts = _summarize_trial_states(refreshed)
                            if writer is not None:
                                writer.update_hpo_progress(
                                    completed=int(counts["completed"]),
                                    total=n_trials,
                                    pruned=int(counts["pruned"]),
                                    failed=int(counts["failed"]),
                                )
                                writer.update_hpo_trial_results(_extract_trial_details(refreshed))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to update HPO run-state progress: %s", exc)

                    for fut in done:
                        try:
                            fut.result()
                        except Exception as exc:  # noqa: BLE001
                            error_text = str(exc)
                            worker_errors.append(error_text)
                            logger.error(
                                "Parallel HPO worker failed; continuing with recovery wave. Error: %s",
                                exc,
                                exc_info=True,
                            )

                    rss_by_pid = _collect_wave_worker_rss_bytes(executor)
                    if writer is not None:
                        try:
                            writer.update_hpo_wave_memory(rss_by_pid)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to publish HPO wave RSS to run-state: %s", exc)

                    trial_started_in_wave = False
                    if counts is not None:
                        current_finished = (
                            int(counts["completed"])
                            + int(counts["pruned"])
                            + int(counts["failed"])
                        )
                        current_running = int(counts.get("running", 0))
                        trial_started_in_wave = current_running > 0 or current_finished > wave_finished_baseline

                    should_terminate, hottest_pid, hottest_rss = _should_trigger_rss_watchdog(
                        rss_watchdog_enabled=rss_watchdog_enabled,
                        rss_by_pid=rss_by_pid,
                        rss_watchdog_limit_bytes=rss_watchdog_max_worker_rss_bytes,
                        wave_started_monotonic=wave_started_monotonic,
                        now_monotonic=time.monotonic(),
                        startup_grace_seconds=rss_watchdog_startup_grace_seconds,
                        require_trial_start=rss_watchdog_require_trial_start,
                        trial_started_in_wave=trial_started_in_wave,
                        startup_timeout_seconds=rss_watchdog_startup_timeout_seconds,
                    )
                    if should_terminate and hottest_pid is not None and hottest_rss is not None:
                        rss_gb = hottest_rss / float(1024**3)
                        limit_gb = rss_watchdog_max_worker_rss_bytes / float(1024**3)
                        logger.warning(
                            "Parallel HPO RSS watchdog triggered: pid=%s rss=%.2fGiB >= limit=%.2fGiB. "
                            "Terminating current wave and relaunching remaining trials.",
                            hottest_pid,
                            rss_gb,
                            limit_gb,
                        )
                        if writer is not None:
                            try:
                                writer.mark_hpo_rss_watchdog_trigger(
                                    pid=int(hottest_pid),
                                    rss_bytes=int(hottest_rss),
                                    limit_bytes=int(rss_watchdog_max_worker_rss_bytes),
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("Failed to publish HPO RSS watchdog event: %s", exc)
                        for process in getattr(executor, "_processes", {}).values():
                            try:
                                if process is not None and process.is_alive():
                                    process.terminate()
                            except Exception:  # noqa: BLE001
                                continue
                        for fut in pending:
                            fut.cancel()
                        pending.clear()
                        watchdog_triggered = True
                        break

            if worker_errors:
                logger.warning(
                    "Parallel HPO wave completed with %s worker failure(s). The supervisor will continue remaining trials.",
                    len(worker_errors),
                )

            wave_had_progress = False
            try:
                post_wave = optuna.load_study(study_name=study_name, storage=storage_uri)
                post_counts = _summarize_trial_states(post_wave)
                post_finished = (
                    int(post_counts["completed"])
                    + int(post_counts["pruned"])
                    + int(post_counts["failed"])
                )
                wave_had_progress = post_finished > wave_finished_baseline
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to evaluate post-wave progress for adaptive scheduler: %s", exc)

            if adaptive_scheduler_enabled and not force_single_worker:
                prev_worker_cap = adaptive_worker_cap
                prev_trial_cap = adaptive_trials_per_worker_cap
                prev_stable_waves = adaptive_stable_waves
                adaptive_worker_cap, adaptive_trials_per_worker_cap, adaptive_stable_waves = _update_adaptive_scheduler_state(
                    current_worker_cap=adaptive_worker_cap,
                    current_trials_per_worker_cap=adaptive_trials_per_worker_cap,
                    max_worker_cap=len(resources),
                    max_trials_per_worker_cap=max_trials_per_worker_process,
                    min_worker_cap=adaptive_scheduler_min_workers,
                    min_trials_per_worker_cap=adaptive_scheduler_min_trials_per_worker_process,
                    stable_waves=adaptive_stable_waves,
                    recovery_waves=adaptive_scheduler_recovery_waves,
                    wave_had_progress=wave_had_progress,
                    worker_error_count=len(worker_errors),
                    watchdog_triggered=watchdog_triggered,
                )

                if adaptive_worker_cap != prev_worker_cap or adaptive_trials_per_worker_cap != prev_trial_cap:
                    instability_reasons: List[str] = []
                    if watchdog_triggered:
                        instability_reasons.append("watchdog")
                    if worker_errors:
                        instability_reasons.append("worker_errors")
                    if not wave_had_progress:
                        instability_reasons.append("no_progress")

                    if instability_reasons:
                        logger.warning(
                            "Adaptive HPO scheduler scaled down after unstable wave: reasons=%s workers=%s->%s trials_per_worker=%s->%s",
                            instability_reasons,
                            prev_worker_cap,
                            adaptive_worker_cap,
                            prev_trial_cap,
                            adaptive_trials_per_worker_cap,
                        )
                    else:
                        logger.info(
                            "Adaptive HPO scheduler scaled up after stable waves=%s: workers=%s->%s trials_per_worker=%s->%s",
                            prev_stable_waves + 1,
                            prev_worker_cap,
                            adaptive_worker_cap,
                            prev_trial_cap,
                            adaptive_trials_per_worker_cap,
                        )

            if watchdog_triggered:
                watchdog_restarts_without_progress += 1
                if (
                    rss_watchdog_single_worker_fallback_enabled
                    and not force_single_worker
                    and len(resources) > 1
                    and watchdog_restarts_without_progress >= rss_watchdog_max_restarts_before_single_worker
                ):
                    force_single_worker = True
                    logger.warning(
                        "Parallel HPO RSS watchdog triggered %s time(s) without progress; "
                        "falling back to single-worker recovery waves.",
                        watchdog_restarts_without_progress,
                    )
                logger.info("Parallel HPO wave restarted after RSS watchdog trigger.")

        study = optuna.load_study(study_name=study_name, storage=storage_uri)
    else:
        def objective(trial: Any) -> float:
            return _evaluate_trial_objective(
                config,
                data_object,
                hpo_cfg,
                metric_name,
                direction,
                trial_log_models,
                trial,
                resource=None,
                study_name="binance_hpo",
            )

        study = optuna.create_study(direction=direction)

        def _callback(study_obj: Any, _trial: Any) -> None:
            if writer is None:
                return
            try:
                counts = _summarize_trial_states(study_obj)
                writer.update_hpo_progress(
                    completed=int(counts["completed"]),
                    total=n_trials,
                    pruned=int(counts["pruned"]),
                    failed=int(counts["failed"]),
                )
                writer.update_hpo_trial_results(_extract_trial_details(study_obj))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to update sequential HPO run-state progress: %s", exc)

        study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    if writer is not None:
        try:
            counts = _summarize_trial_states(study)
            writer.update_hpo_progress(
                completed=int(counts["completed"]),
                total=n_trials,
                pruned=int(counts["pruned"]),
                failed=int(counts["failed"]),
            )
            writer.update_hpo_trial_results(_extract_trial_details(study))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to finalize HPO run-state progress: %s", exc)

    counts = _summarize_trial_states(study)
    total_finished = int(counts["completed"]) + int(counts["pruned"]) + int(counts["failed"])
    if int(counts["completed"]) <= 0:
        search_space = hpo_cfg.get("search_space") if isinstance(hpo_cfg, dict) else None
        if not isinstance(search_space, dict):
            search_space = {}
        guidance = _format_search_space_guidance(search_space)
        if total_finished == 0:
            detail = (
                "No trials were executed at all. This usually means the study DB "
                "contained stale trials from a previous run that exhausted the trial budget. "
                "Delete the study DB or set parallel.resume_study=false (default)."
            )
        else:
            failure_categories: Dict[str, int] = {}
            failure_messages: List[str] = []
            for trial in getattr(study, "trials", []) or []:
                state_obj = getattr(trial, "state", None)
                state_name = getattr(state_obj, "name", str(state_obj))
                if state_name not in {"PRUNED", "FAIL"}:
                    continue

                user_attrs = getattr(trial, "user_attrs", {}) or {}
                category = str(user_attrs.get("regime_failure_category") or "unknown")
                failure_categories[category] = int(failure_categories.get(category, 0)) + 1

                msg = user_attrs.get("regime_failure_message")
                if msg:
                    failure_messages.append(str(msg))

            # Prefer deterministic config errors over generic resource guidance.
            config_mismatch_detail = None
            try:
                boundaries = config["targets"]["price_classes"]["boundaries"]
                num_classes = int(config["model"]["output"]["num_classes"])
                expected = (len(boundaries) + 1) if isinstance(boundaries, list) else None
                if expected is not None and num_classes != expected:
                    config_mismatch_detail = (
                        "Configuration mismatch: model.output.num_classes must equal "
                        "len(targets.price_classes.boundaries) + 1. "
                        f"num_classes={num_classes}, boundaries_len={len(boundaries)}, expected={expected}."
                    )
            except Exception:  # noqa: BLE001
                config_mismatch_detail = None

            if int(failure_categories.get("configuration", 0)) >= max(1, total_finished):
                example = failure_messages[0] if failure_messages else "(no message)"
                detail = (
                    f"All {total_finished} trial(s) failed due to configuration errors "
                    f"(pruned={counts['pruned']} failed={counts['failed']}). "
                    f"Example: {example}"
                )
                if config_mismatch_detail is not None:
                    detail = detail + " " + config_mismatch_detail
            else:
                detail = (
                    f"All {total_finished} trial(s) ended without a successful completion "
                    f"(pruned={counts['pruned']} failed={counts['failed']}). "
                    f"Failure categories={failure_categories}. "
                    "This typically indicates OOM errors exhausting all batch-backoff retries. "
                    "Review the search space and relax constraints. Suggested adjustments:\n"
                    f"{guidance}"
                )
        raise ConfigError(
            "Hyperparameter optimization completed with no successful trials. "
            f"completed={counts['completed']} pruned={counts['pruned']} failed={counts['failed']}. "
            f"{detail}"
        )

    best_trial = study.best_trial
    best_params, requested_batch_size, effective_batch_size = _resolve_best_params_for_final_training(best_trial)
    best_value = best_trial.value
    if best_value is None:
        raise ValueError("Optuna best trial has no objective value")
    best_value_float = float(best_value)

    try:
        best_trial_number = int(getattr(best_trial, "number"))
    except Exception:  # noqa: BLE001
        best_trial_number = None

    if (
        requested_batch_size is not None
        and effective_batch_size is not None
        and requested_batch_size != effective_batch_size
    ):
        logger.info(
            "Best trial%s used effective batch_size=%s (requested=%s); applying effective batch size for final training.",
            f" {best_trial_number}" if best_trial_number is not None else "",
            effective_batch_size,
            requested_batch_size,
        )

    best_config = _apply_hyperparameters(config, best_params)

    try:
        import mlflow  # type: ignore[import]
    except Exception:  # noqa: BLE001
        pass
    else:
        try:
            mlflow.log_param("hpo_enabled", True)
            mlflow.log_param("hpo_framework", framework)
            mlflow.log_param("hpo_n_trials", n_trials)
            mlflow.log_param("hpo_direction", direction)
            mlflow.log_param("hpo_metric", metric_name)
            if requested_batch_size is not None:
                mlflow.log_param("hpo_best_batch_size_requested", int(requested_batch_size))
            if effective_batch_size is not None:
                mlflow.log_param("hpo_best_batch_size_effective", int(effective_batch_size))
            for name, value in best_params.items():
                mlflow.log_param(f"hpo_best_{name}", value)
            mlflow.log_metric("hpo_best_value", best_value_float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log hyperparameter optimization results to MLFlow: %s", exc)

    logger.info(
        "Hyperparameter optimization completed: best_value=%s, best_params=%s",
        best_value_float,
        best_params,
    )

    return best_config


__all__ = ["run_hyperparameter_search"]
