"""Hyperparameter optimization using Optuna.

This module defines the interface for running hyperparameter optimization
driven entirely by the ``hyperparameter_optimization`` section of the
configuration.
"""

from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
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


logger = logging.getLogger(__name__)


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


def _read_regime_memory(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _default_regime_memory()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return _default_regime_memory()
        return payload
    except Exception:  # noqa: BLE001
        return _default_regime_memory()


def _build_model_signature(params: Dict[str, Any]) -> str:
    return (
        f"cnn1={int(params.get('cnn_filters_1', 0))};"
        f"cnn2={int(params.get('cnn_filters_2', 0))};"
        f"lstm={int(params.get('lstm_units', 0))}"
    )


def _update_regime_memory(
    path: str,
    *,
    signature: str,
    successful_batch: Optional[int] = None,
    oom_batch: Optional[int] = None,
) -> None:
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
                    payload = _default_regime_memory()
            else:
                payload = _default_regime_memory()
            if not isinstance(payload, dict):
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

    training_cfg["runtime"] = runtime_cfg
    cfg["training"] = training_cfg
    return cfg


def _resolve_parallel_settings(hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    parallel_cfg = hpo_cfg.get("parallel")
    if not isinstance(parallel_cfg, dict):
        return {"enabled": False, "resources": [], "storage_uri": None, "study_name": None}

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

    return {
        "enabled": enabled,
        "resources": resources,
        "storage_uri": storage_uri,
        "study_name": study_name,
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
        return

    _configure_worker_tensorflow_runtime(
        enable_memory_growth=bool(runtime_options["gpu_memory_growth"]),
    )


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

    max_retry_attempts = int(regime_settings["max_retry_attempts"]) if regime_enabled else 0
    attempt = 0
    last_exception: Optional[Exception] = None

    while True:
        params_for_attempt = dict(params)
        params_for_attempt["batch_size"] = int(active_batch_size)

        trial_config = _apply_hyperparameters(base_config, params_for_attempt)
        if resource is not None:
            trial_config = _apply_worker_resource(trial_config, resource)

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
                and int(active_batch_size) > int(regime_settings["min_batch_size"])
            )
            if can_retry:
                next_batch_size = _compute_next_batch_size(
                    int(active_batch_size),
                    float(regime_settings["batch_backoff_factor"]),
                    int(regime_settings["min_batch_size"]),
                )
                if next_batch_size >= int(active_batch_size):
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
                trial.set_user_attr("resource", resource)
                if str(regime_settings["failure_policy"]) == "penalize":
                    return _resolve_failure_objective_value(
                        direction,
                        float(regime_settings["failure_penalty_value"]),
                    )

                import optuna  # type: ignore[import]

                raise optuna.TrialPruned(
                    f"Resource-constrained trial pruned after {attempt} retries: {exc}"
                ) from exc
            raise

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

    def _ensure_list(name: str) -> Any:
        if name not in search_space:
            raise ValueError(f"Missing hyperparameter_optimization.search_space entry for {name!r}")
        value = search_space[name]
        if not isinstance(value, list):
            raise ValueError(
                "Each hyperparameter_optimization.search_space entry must be a list; "
                f"got type={type(value).__name__} for {name!r}",
            )
        if not value:
            raise ValueError(f"hyperparameter_optimization.search_space.{name} must be a non-empty list")
        return value

    def _suggest_int(name: str, bounds: Any) -> int:
        if len(bounds) < 2:
            raise ValueError(f"Integer search space for {name!r} must have at least two elements [low, high]")
        low = int(bounds[0])
        high = int(bounds[1])
        return int(trial.suggest_int(name, low, high))

    def _suggest_float(name: str, bounds: Any) -> float:
        if len(bounds) < 2:
            raise ValueError(f"Float search space for {name!r} must have at least two elements [low, high]")
        low = float(bounds[0])
        high = float(bounds[1])
        log_scale = False
        if len(bounds) >= 3 and str(bounds[2]).lower() == "log":
            log_scale = True
        return float(trial.suggest_float(name, low, high, log=log_scale))

    params: Dict[str, Any] = {}

    cnn_filters_1_space = _ensure_list("cnn_filters_1")
    cnn_filters_2_space = _ensure_list("cnn_filters_2")
    lstm_units_space = _ensure_list("lstm_units")
    learning_rate_space = _ensure_list("learning_rate")
    batch_size_space = _ensure_list("batch_size")

    params["cnn_filters_1"] = _suggest_int("cnn_filters_1", cnn_filters_1_space)
    params["cnn_filters_2"] = _suggest_int("cnn_filters_2", cnn_filters_2_space)
    params["lstm_units"] = _suggest_int("lstm_units", lstm_units_space)
    params["learning_rate"] = _suggest_float("learning_rate", learning_rate_space)

    if len(batch_size_space) >= 3:
        choices = sorted({int(v) for v in batch_size_space})
        params["batch_size"] = int(trial.suggest_categorical("batch_size", choices))
    elif len(batch_size_space) == 2:
        params["batch_size"] = _suggest_int("batch_size", batch_size_space)
    else:
        raise ValueError("hyperparameter_optimization.search_space.batch_size must have at least two values")

    return params


def _apply_hyperparameters(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)

    model_cfg = cfg["model"]
    cnn_cfg = model_cfg["cnn"]
    lstm_cfg = model_cfg["lstm"]
    compilation_cfg = model_cfg["compilation"]
    training_cfg = cfg["training"]

    filters = list(cnn_cfg["filters"])
    if len(filters) < 2:
        raise ValueError("model.cnn.filters must have length at least 2 to apply cnn_filters_1 and cnn_filters_2")
    filters[0] = int(params["cnn_filters_1"])
    filters[1] = int(params["cnn_filters_2"])
    cnn_cfg["filters"] = filters

    lstm_cfg["units"] = int(params["lstm_units"])
    compilation_cfg["learning_rate"] = float(params["learning_rate"])
    training_cfg["batch_size"] = int(params["batch_size"])

    model_cfg["cnn"] = cnn_cfg
    model_cfg["lstm"] = lstm_cfg
    model_cfg["compilation"] = compilation_cfg
    cfg["model"] = model_cfg
    cfg["training"] = training_cfg

    return cfg


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
        allocations = _allocate_trials_to_workers(n_trials, n_workers)
        storage_uri = _build_trial_storage_uri(config, parallel_settings["storage_uri"])
        study_name = str(parallel_settings["study_name"] or "binance_hpo")

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=True,
        )

        logger.info(
            "Starting parallel HPO with %s workers over %s trials. resources=%s storage=%s study=%s",
            n_workers,
            n_trials,
            resources,
            storage_uri,
            study_name,
        )

        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=spawn_ctx) as executor:
            futures = []
            for resource, worker_trials in zip(resources, allocations):
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
                    )
                )
            for fut in futures:
                fut.result()

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
        study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_value = best_trial.value
    if best_value is None:
        raise ValueError("Optuna best trial has no objective value")
    best_value_float = float(best_value)

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
