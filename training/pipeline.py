"""Training pipeline for snapshot-based datasets.

Builds training datasets from snapshots, trains models, and logs metrics and
artifacts to MLflow when configured.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
from datetime import datetime, timedelta
import contextlib
import copy
import hashlib
import json
import logging
import os
import shutil

import numpy as np

from utils.config_loader import ConfigError
from preprocessing.snapshot_sequence_builder import (
    build_top_of_book_sequence_tensor,
    build_hybrid_depth_sequence_tensor,
)
from preprocessing.normalizer import create_normalizer_from_config
from preprocessing.feature_engineering import FeatureEngineer
from mlflow_integration.model_registry import register_model
from .snapshot_dataset import (
    NormalizationStats,
    build_training_generator,
    compute_label_distribution,
    compute_normalization_stats,
    get_chunk_x_shape,
    get_mask_channel_info,
    load_label_stats_from_manifest,
    load_normalization_stats,
    prepare_snapshot_dataset,
    save_label_stats_to_manifest,
    save_normalization_stats,
)
from .snapshot_store import load_or_create_manifest, resolve_snapshot_context, save_manifest
from .callbacks import create_callbacks
from .class_weights import compute_class_weights, compute_class_weights_from_counts
from .distributed import (
    DistributedContext,
    build_distributed_context,
    wrap_generator_as_dataset,
)
from .long_term_context import (
    compute_long_term_features_for_dataset,
    is_long_term_enabled,
    wrap_generator_with_long_term,
)


logger = logging.getLogger(__name__)


def _compute_split_boundaries(
    n_samples: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * validation_ratio)
    if val_end > n_samples:
        val_end = n_samples
    test_end = n_samples
    return train_end, val_end, test_end


def _resolve_snapshot_training_indices(
    config: Dict[str, Any],
    n_samples: int,
) -> Optional[Tuple[int, int, int]]:
    """Resolve train/validation index boundaries for snapshot training."""
    training_cfg = config["training"]
    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    configured_val_split = float(training_cfg["validation_split"])
    if abs(configured_val_split - validation_ratio) > 1e-6:
        raise ValueError(
            "training.validation_split must match preprocessing.train_test_split.validation_ratio",
        )

    train_end, val_end, _ = _compute_split_boundaries(
        n_samples,
        train_ratio,
        validation_ratio,
        test_ratio,
    )
    if train_end <= 0:
        return None

    debug_max_samples = int(training_cfg["debug_max_samples"])
    effective_train_n = min(train_end, debug_max_samples)
    if effective_train_n <= 0:
        return None

    val_start = train_end
    val_end = min(val_end, n_samples)
    return effective_train_n, val_start, val_end


def _precompute_trial_invariant_snapshot_artifacts(config: Dict[str, Any], snapshot_dataset: Any) -> None:
    """Warm trial-invariant snapshot artifacts used by HPO trials.

    This computes and caches data artifacts that do not depend on sampled
    hyperparameters (normalization stats, label stats for class weighting, and
    long-term features) so parallel HPO workers can focus on model training.
    """
    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Skipping precompute: snapshot dataset has no samples.")
        return
    if not getattr(snapshot_dataset, "chunks", None):
        logger.info("Skipping precompute: snapshot dataset has no chunk files.")
        return

    _enforce_production_sample_cap_snapshot(config, n_samples)

    split_indices = _resolve_snapshot_training_indices(config, n_samples)
    if split_indices is None:
        logger.info("Skipping precompute: no effective training samples available.")
        return
    effective_train_n, val_start, val_end = split_indices
    val_count = max(0, val_end - val_start)

    context = resolve_snapshot_context(config)
    manifest = load_or_create_manifest(context, config)

    normalization_cfg = config["preprocessing"]["normalization"]
    fit_on_train_only = bool(normalization_cfg["fit_on_train_only"])

    _get_normalization_stats(
        config,
        context,
        manifest,
        snapshot_dataset,
        0,
        effective_train_n,
        "train",
    )
    if not fit_on_train_only and val_count > 0:
        _get_normalization_stats(
            config,
            context,
            manifest,
            snapshot_dataset,
            val_start,
            val_end,
            "val",
        )

    training_cfg = config["training"]
    output_cfg = config["model"]["output"]
    use_class_weights = bool(training_cfg["class_weights"]["compute_from_train"])
    if use_class_weights:
        num_classes = int(output_cfg["num_classes"])
        train_label_dist = load_label_stats_from_manifest(manifest, "train")
        if train_label_dist is None:
            train_label_dist = compute_label_distribution(
                snapshot_dataset,
                start_index=0,
                end_index=effective_train_n,
                num_classes=num_classes,
            )
            save_label_stats_to_manifest(context, manifest, train_label_dist, "train")

    if is_long_term_enabled(config):
        cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
        compute_long_term_features_for_dataset(
            config,
            snapshot_dataset,
            cadence_seconds=cadence_seconds,
        )

    logger.info(
        "Precomputed trial-invariant snapshot artifacts: train_samples=%s, val_samples=%s, class_weights=%s, long_term=%s",
        effective_train_n,
        val_count,
        use_class_weights,
        is_long_term_enabled(config),
    )


def _enforce_production_sample_cap_snapshot(config: Dict[str, Any], n_samples: int) -> None:
    run_mode_cfg = config["run_mode"]
    mode = str(run_mode_cfg["mode"])
    if mode != "production":
        return

    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])
    if debug_max_samples < n_samples:
        raise ConfigError(
            "training.debug_max_samples must be >= metadata.num_samples when run_mode.mode='production'. "
            f"debug_max_samples={debug_max_samples}, num_samples={n_samples}."
        )


def _get_normalization_stats(
    config: Dict[str, Any],
    context: Any,
    manifest: Dict[str, Any],
    dataset: Any,
    start_index: int,
    end_index: int,
    stats_key: str,
) -> NormalizationStats:
    norm_cfg = config["preprocessing"]["normalization"]
    method = str(norm_cfg["method"])
    stats_path = os.path.join(context.snapshot_dir, f"normalization_stats_{stats_key}.npz")

    mask_start, mask_count = get_mask_channel_info(config)

    if os.path.exists(stats_path):
        stats = load_normalization_stats(stats_path)
    else:
        stats = compute_normalization_stats(
            dataset,
            start_index,
            end_index,
            method,
            mask_start=mask_start,
            mask_count=mask_count,
        )
        save_normalization_stats(stats_path, stats)

    stats_meta = manifest.get("normalization_stats", {})
    stats_meta[stats_key] = {
        "method": stats.method,
        "path": stats_path,
        "start_index": start_index,
        "end_index": end_index,
    }
    manifest["normalization_stats"] = stats_meta
    save_manifest(context, manifest)

    return stats


def _extract_hpo_metric_from_history(config: Dict[str, Any], history: Any) -> Optional[float]:
    try:
        hpo_cfg = config["hyperparameter_optimization"]
        if not bool(hpo_cfg["enabled"]):
            return None
        metric_name = str(hpo_cfg["metric"])
        history_dict = getattr(history, "history", None)
        if not isinstance(history_dict, dict):
            return None
        series = history_dict.get(metric_name)
        if not series:
            return None
        return float(series[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract HPO metric from training history: %s", exc)
        return None


def _resolve_hpo_metric_weight(metric_name: str, effective_train_n: int, val_count: int) -> float:
    if metric_name.startswith("val_"):
        return float(max(0, val_count))
    return float(max(0, effective_train_n))


def _aggregate_hpo_window_metrics(window_metrics: List[Tuple[float, float]]) -> Optional[float]:
    if not window_metrics:
        return None

    weighted_sum = 0.0
    total_weight = 0.0
    simple_values: List[float] = []
    for metric_value, metric_weight in window_metrics:
        simple_values.append(float(metric_value))
        if metric_weight > 0:
            weighted_sum += float(metric_value) * float(metric_weight)
            total_weight += float(metric_weight)

    if total_weight > 0:
        return weighted_sum / total_weight

    return float(sum(simple_values) / len(simple_values))


def _sanitize_resume_component(value: str) -> str:
    safe_chars: List[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("-")
    return "".join(safe_chars)


def _resolve_sequential_resume_paths(config: Dict[str, Any], windows: List[Tuple[str, str]]) -> Tuple[str, str]:
    snapshot_cfg = config["snapshot"]
    data_cfg = config["data"]
    target_asset = str(data_cfg["asset_pairs"]["target_asset"])
    training_cfg = config.get("training")

    root_dir = str(snapshot_cfg["directory"])
    root_name = str(snapshot_cfg["root_name"])
    snapshot_name = str(snapshot_cfg["name"])
    start_date = str(data_cfg["time_range"]["start_date"])
    end_date = str(data_cfg["time_range"]["end_date"])

    resume_namespace = ""
    if isinstance(training_cfg, dict):
        sequential_cfg = training_cfg.get("sequential_training")
        if isinstance(sequential_cfg, dict):
            resume_namespace = str(sequential_cfg.get("resume_namespace", "")).strip()

    windows_payload = json.dumps(windows, sort_keys=True, separators=(",", ":"))
    windows_hash = hashlib.sha256(windows_payload.encode("utf-8")).hexdigest()[:12]

    run_key_components = [
        _sanitize_resume_component(root_name),
        _sanitize_resume_component(snapshot_name),
        _sanitize_resume_component(target_asset),
        _sanitize_resume_component(start_date),
        _sanitize_resume_component(end_date),
        windows_hash,
    ]
    if resume_namespace:
        run_key_components.append(_sanitize_resume_component(resume_namespace))

    run_key = "__".join(run_key_components)
    state_dir = os.path.join(root_dir, "_sequential_resume")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"{run_key}.json")
    model_path = os.path.join(state_dir, f"{run_key}.keras")
    return state_path, model_path


def _load_sequential_resume_state(state_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load sequential resume state %s: %s", state_path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Sequential resume state is not a JSON object: %s", state_path)
        return None
    return payload


def _save_sequential_resume_state(state_path: str, payload: Dict[str, Any]) -> None:
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def _load_sequential_resume_model(model_path: str) -> Any:
    tf_keras_models = __import__("tensorflow.keras.models", fromlist=["load_model"])
    return tf_keras_models.load_model(model_path)


def _save_sequential_resume_model(model: Any, model_path: str) -> None:
    model.save(model_path)


def _cleanup_completed_window_dirs(completed_dirs: List[str], keep_last_windows: int) -> List[str]:
    if keep_last_windows < 0:
        raise ConfigError("training.sequential_training.cleanup_keep_last_windows must be >= 0")

    retained_dirs = list(completed_dirs)
    while len(retained_dirs) > keep_last_windows:
        old_dir = retained_dirs.pop(0)
        if not os.path.isdir(old_dir):
            continue
        try:
            shutil.rmtree(old_dir)
            logger.info("Removed completed window snapshot directory: %s", old_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove window snapshot directory %s: %s", old_dir, exc)

    return retained_dirs


def _resolve_sequential_windows(config: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    training_cfg = config["training"]
    sequential_cfg = training_cfg.get("sequential_training")
    if not isinstance(sequential_cfg, dict):
        return None

    enabled = bool(sequential_cfg.get("enabled", False))
    if not enabled:
        return None

    window_days_raw = sequential_cfg.get("window_days")
    if window_days_raw is None:
        raise ConfigError(
            "training.sequential_training.window_days is required when sequential_training.enabled is true"
        )

    window_days = int(window_days_raw)
    if window_days <= 0:
        raise ConfigError("training.sequential_training.window_days must be positive")

    start_date = str(config["data"]["time_range"]["start_date"])
    end_date = str(config["data"]["time_range"]["end_date"])

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ConfigError(
            "data.time_range.start_date and end_date must be in YYYY-MM-DD format for sequential training"
        ) from exc

    if start_dt > end_dt:
        raise ConfigError("data.time_range.start_date must be <= end_date")

    windows: List[Tuple[str, str]] = []
    cursor = start_dt
    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days - 1), end_dt)
        windows.append((cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        cursor = window_end + timedelta(days=1)

    return windows


def _fit_snapshot_model_once(
    config: Dict[str, Any],
    snapshot_dataset: Any,
    model: Optional[Any],
    *,
    epoch_step_offset: int,
    dist_ctx: Optional[DistributedContext] = None,
) -> Tuple[Optional[Any], int, Optional[float], float]:
    training_cfg = config["training"]
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    # MirroredStrategy splits each dataset element across replicas, so the
    # generator must yield global-batch-size samples per step.
    generator_batch_size = (
        dist_ctx.global_batch_size(batch_size) if dist_ctx is not None
        else batch_size
    )

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError(
            "Only model.output.type='two_head_intensity' is supported in snapshot training",
        )

    class_weights_cfg = training_cfg["class_weights"]
    use_class_weights = bool(class_weights_cfg["compute_from_train"])

    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Snapshot training window skipped: snapshot dataset has no samples.")
        return model, 0, None, 0.0

    _enforce_production_sample_cap_snapshot(config, n_samples)

    split_indices = _resolve_snapshot_training_indices(config, n_samples)
    if split_indices is None:
        logger.info("Snapshot training window skipped: no effective training samples available.")
        return model, 0, None, 0.0
    effective_train_n, val_start, val_end = split_indices

    val_count = max(0, val_end - val_start)

    if not snapshot_dataset.chunks:
        logger.info("Snapshot training window skipped: no chunk files found.")
        return model, 0, None, 0.0

    first_chunk = snapshot_dataset.chunks[0]
    x_shape = get_chunk_x_shape(first_chunk)
    if len(x_shape) != 5:
        raise ValueError("Snapshot input tensors must have rank 5")
    input_shape = tuple(int(d) for d in x_shape[1:])

    long_term_features = None
    long_term_input_dim: Optional[int] = None
    if is_long_term_enabled(config):
        cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
        long_term_features = compute_long_term_features_for_dataset(
            config,
            snapshot_dataset,
            cadence_seconds=cadence_seconds,
        )
        if long_term_features is None:
            raise ConfigError("Long-term features enabled but computation returned None")
        if long_term_features.shape[0] != n_samples:
            raise ConfigError(
                "Long-term feature rows do not match snapshot dataset sample count: "
                f"features={long_term_features.shape[0]}, samples={n_samples}"
            )
        long_term_input_dim = int(long_term_features.shape[1])

    context = resolve_snapshot_context(config)
    manifest = load_or_create_manifest(context, config)

    normalization_cfg = config["preprocessing"]["normalization"]
    fit_on_train_only = bool(normalization_cfg["fit_on_train_only"])

    train_stats = _get_normalization_stats(
        config,
        context,
        manifest,
        snapshot_dataset,
        0,
        effective_train_n,
        "train",
    )

    mask_start, mask_count = get_mask_channel_info(config)

    if fit_on_train_only or val_count <= 0:
        val_stats = train_stats
    else:
        val_stats = _get_normalization_stats(
            config,
            context,
            manifest,
            snapshot_dataset,
            val_start,
            val_end,
            "val",
        )

    class_weights_up: Optional[Dict[int, float]] = None
    class_weights_down: Optional[Dict[int, float]] = None
    if use_class_weights:
        num_classes = int(output_cfg["num_classes"])
        train_label_dist = load_label_stats_from_manifest(manifest, "train")
        if train_label_dist is None:
            train_label_dist = compute_label_distribution(
                snapshot_dataset,
                start_index=0,
                end_index=effective_train_n,
                num_classes=num_classes,
            )
            save_label_stats_to_manifest(context, manifest, train_label_dist, "train")

        class_weights_up = compute_class_weights_from_counts(train_label_dist.up_counts, num_classes)
        class_weights_down = compute_class_weights_from_counts(train_label_dist.down_counts, num_classes)

    # When distributed, model must be built/compiled inside strategy.scope().
    strategy_scope = dist_ctx.scope() if dist_ctx is not None else contextlib.nullcontext()

    with strategy_scope:
        if model is None:
            fine_tuning_cfg = training_cfg["fine_tuning"]
            fine_tuning_enabled = bool(fine_tuning_cfg["enabled"])
            if fine_tuning_enabled:
                from .fine_tuning import (
                    FineTuningError,
                    load_model_from_registry,
                    load_model_from_run,
                    prepare_fine_tuning,
                )

                use_registry = bool(fine_tuning_cfg["use_model_registry"])
                if use_registry:
                    registry_name = fine_tuning_cfg["registry_name"]
                    if not registry_name:
                        raise ConfigError(
                            "training.fine_tuning.registry_name is required when use_model_registry is true"
                        )
                    stage = str(fine_tuning_cfg["base_model_stage"])
                    try:
                        model = load_model_from_registry(registry_name, stage=stage)
                    except FineTuningError as exc:
                        raise ConfigError(f"Failed to load base model for fine-tuning: {exc}") from exc
                else:
                    run_id = fine_tuning_cfg["base_model_run_id"]
                    if not run_id:
                        raise ConfigError(
                            "training.fine_tuning.base_model_run_id is required when fine_tuning.enabled is true "
                            "and use_model_registry is false"
                        )
                    try:
                        model = load_model_from_run(run_id)
                    except FineTuningError as exc:
                        raise ConfigError(f"Failed to load base model for fine-tuning: {exc}") from exc

                try:
                    model = prepare_fine_tuning(
                        config,
                        model,
                        input_shape=input_shape,
                        long_term_input_dim=long_term_input_dim,
                    )
                except FineTuningError as exc:
                    raise ConfigError(f"Failed to prepare model for fine-tuning: {exc}") from exc
            else:
                from models.cnn_lstm_multiclass import build_cnn_lstm_model

                model = build_cnn_lstm_model(
                    config,
                    input_shape=input_shape,
                    long_term_input_dim=long_term_input_dim,
                )

    num_classes = int(output_cfg["num_classes"])

    def _make_train_gen() -> Iterator[Tuple[Any, ...]]:
        gen, _ = build_training_generator(
            dataset=snapshot_dataset,
            start_index=0,
            end_index=effective_train_n,
            batch_size=generator_batch_size,
            num_classes=num_classes,
            normalization=train_stats,
            sample_weight_cfg=training_cfg["sample_weighting"],
            mask_start=mask_start,
            mask_count=mask_count,
            class_weights_up=class_weights_up,
            class_weights_down=class_weights_down,
        )
        if long_term_features is not None:
            gen = wrap_generator_with_long_term(
                gen, long_term_features, start_index=0, end_index=effective_train_n,
            )
        return gen

    _, train_steps = build_training_generator(
        dataset=snapshot_dataset,
        start_index=0,
        end_index=effective_train_n,
        batch_size=generator_batch_size,
        num_classes=num_classes,
        normalization=train_stats,
        sample_weight_cfg=training_cfg["sample_weighting"],
        mask_start=mask_start,
        mask_count=mask_count,
        class_weights_up=class_weights_up,
        class_weights_down=class_weights_down,
    )

    if dist_ctx is not None:
        global_batch_size = generator_batch_size
        train_data: Any = wrap_generator_as_dataset(
            generator_factory=_make_train_gen,
            input_shape=input_shape,
            num_classes=num_classes,
            long_term_dim=long_term_input_dim,
            global_batch_size=global_batch_size,
            steps_per_epoch=train_steps,
            distributed_ctx=dist_ctx,
        )
    else:
        train_data = _make_train_gen()

    callbacks = create_callbacks(config)
    fit_kwargs: Dict[str, Any] = {
        "x": train_data,
        "epochs": epochs,
        "steps_per_epoch": train_steps,
        "callbacks": callbacks,
        "verbose": 1,
    }

    if val_count > 0:
        def _make_val_gen() -> Iterator[Tuple[Any, ...]]:
            gen, _ = build_training_generator(
                dataset=snapshot_dataset,
                start_index=val_start,
                end_index=val_end,
                batch_size=generator_batch_size,
                num_classes=num_classes,
                normalization=val_stats,
                sample_weight_cfg=None,
                mask_start=mask_start,
                mask_count=mask_count,
                class_weights_up=None,
                class_weights_down=None,
            )
            if long_term_features is not None:
                gen = wrap_generator_with_long_term(
                    gen, long_term_features, start_index=val_start, end_index=val_end,
                )
            return gen

        _, val_steps = build_training_generator(
            dataset=snapshot_dataset,
            start_index=val_start,
            end_index=val_end,
            batch_size=generator_batch_size,
            num_classes=num_classes,
            normalization=val_stats,
            sample_weight_cfg=None,
            mask_start=mask_start,
            mask_count=mask_count,
            class_weights_up=None,
            class_weights_down=None,
        )

        if dist_ctx is not None:
            val_data: Any = wrap_generator_as_dataset(
                generator_factory=_make_val_gen,
                input_shape=input_shape,
                num_classes=num_classes,
                long_term_dim=long_term_input_dim,
                global_batch_size=generator_batch_size,
                steps_per_epoch=val_steps,
                distributed_ctx=dist_ctx,
            )
        else:
            val_data = _make_val_gen()

        fit_kwargs["validation_data"] = val_data
        fit_kwargs["validation_steps"] = val_steps

    if model is None:
        raise ConfigError("Model is not initialized for snapshot training window")
    history = model.fit(**fit_kwargs)

    hpo_metric_value = _extract_hpo_metric_from_history(config, history)
    hpo_metric_weight = 0.0
    if hpo_metric_value is not None:
        config["_hpo_last_metric"] = hpo_metric_value
        metric_name = str(config["hyperparameter_optimization"]["metric"])
        hpo_metric_weight = _resolve_hpo_metric_weight(metric_name, effective_train_n, val_count)

    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for training history logging: %s", exc)
    else:
        if hasattr(history, "history") and isinstance(history.history, dict):
            for metric_name, values in history.history.items():
                try:
                    series = list(values)  # type: ignore[arg-type]
                except TypeError:
                    continue
                for step, value in enumerate(series):
                    mlflow.log_metric(metric_name, float(value), step=epoch_step_offset + step)

    return model, epochs, hpo_metric_value, hpo_metric_weight


def _run_snapshot_training_pipeline_sequential(
    config: Dict[str, Any],
    windows: List[Tuple[str, str]],
    *,
    dist_ctx: Optional[DistributedContext] = None,
) -> Optional[Any]:
    sequential_cfg = config["training"].get("sequential_training") or {}
    cleanup_completed = bool(sequential_cfg.get("cleanup_completed_windows", False))
    cleanup_keep_last_windows = int(sequential_cfg.get("cleanup_keep_last_windows", 0))
    resume_enabled = bool(sequential_cfg.get("resume_enabled", False))

    model: Optional[Any] = None
    epoch_offset = 0
    processed_windows = 0
    hpo_window_metrics: List[Tuple[float, float]] = []
    completed_snapshot_dirs: List[str] = []
    start_window_index = 0
    state_path, resume_model_path = _resolve_sequential_resume_paths(config, windows)

    if resume_enabled:
        resume_state = _load_sequential_resume_state(state_path)
        if resume_state is not None:
            saved_windows = resume_state.get("windows")
            saved_windows_normalized: Optional[List[Tuple[str, str]]] = None
            if isinstance(saved_windows, list):
                normalized: List[Tuple[str, str]] = []
                valid = True
                for item in saved_windows:
                    if not (isinstance(item, list) and len(item) == 2):
                        valid = False
                        break
                    normalized.append((str(item[0]), str(item[1])))
                if valid:
                    saved_windows_normalized = normalized

            if saved_windows_normalized == windows:
                start_window_index = int(resume_state.get("next_window_index", 0))
                if start_window_index < 0 or start_window_index > len(windows):
                    start_window_index = 0
                epoch_offset = int(resume_state.get("epoch_offset", 0))
                if epoch_offset < 0:
                    epoch_offset = 0
                stored_metrics = resume_state.get("hpo_window_metrics", [])
                if isinstance(stored_metrics, list):
                    for item in stored_metrics:
                        if isinstance(item, list) and len(item) == 2:
                            hpo_window_metrics.append((float(item[0]), float(item[1])))

                if start_window_index > 0 and os.path.exists(resume_model_path):
                    try:
                        model = _load_sequential_resume_model(resume_model_path)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to load sequential resume model %s; restarting from window 1: %s",
                            resume_model_path,
                            exc,
                        )
                        model = None
                        start_window_index = 0
                        epoch_offset = 0
                        hpo_window_metrics = []
                    else:
                        logger.info(
                            "Resuming sequential snapshot training from window %s/%s",
                            start_window_index + 1,
                            len(windows),
                        )
                elif start_window_index > 0:
                    logger.warning(
                        "Sequential resume state found but model checkpoint missing (%s); restarting from window 1",
                        resume_model_path,
                    )
                    start_window_index = 0
                    epoch_offset = 0
                    hpo_window_metrics = []
            else:
                logger.warning(
                    "Sequential resume state windows mismatch; ignoring state file %s",
                    state_path,
                )

    for idx, (window_start, window_end) in enumerate(windows):
        if idx < start_window_index:
            continue
        window_config = copy.deepcopy(config)
        window_config["data"]["time_range"]["start_date"] = window_start
        window_config["data"]["time_range"]["end_date"] = window_end

        logger.info(
            "Sequential snapshot training window %s/%s: %s -> %s",
            idx + 1,
            len(windows),
            window_start,
            window_end,
        )

        snapshot_dataset = prepare_snapshot_dataset(window_config)
        model, epochs_ran, hpo_metric_value, hpo_metric_weight = _fit_snapshot_model_once(
            window_config,
            snapshot_dataset,
            model,
            epoch_step_offset=epoch_offset,
            dist_ctx=dist_ctx,
        )
        if hpo_metric_value is not None:
            hpo_window_metrics.append((hpo_metric_value, hpo_metric_weight))
        epoch_offset += epochs_ran
        processed_windows += 1

        if resume_enabled and model is not None:
            try:
                _save_sequential_resume_model(model, resume_model_path)
                _save_sequential_resume_state(
                    state_path,
                    {
                        "version": 1,
                        "windows": windows,
                        "next_window_index": idx + 1,
                        "epoch_offset": epoch_offset,
                        "hpo_window_metrics": [
                            [float(metric), float(weight)] for metric, weight in hpo_window_metrics
                        ],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist sequential resume state after window %s: %s", idx + 1, exc)

        if cleanup_completed:
            context = resolve_snapshot_context(window_config)
            completed_snapshot_dirs.append(context.snapshot_dir)
            completed_snapshot_dirs = _cleanup_completed_window_dirs(
                completed_snapshot_dirs,
                cleanup_keep_last_windows,
            )

    if model is None:
        logger.info("Sequential snapshot training completed with no trainable windows.")
        return None

    aggregated_hpo_metric = _aggregate_hpo_window_metrics(hpo_window_metrics)
    if aggregated_hpo_metric is not None:
        config["_hpo_last_metric"] = aggregated_hpo_metric
        logger.info(
            "Aggregated sequential HPO metric across windows: metric=%s, windows=%s",
            aggregated_hpo_metric,
            len(hpo_window_metrics),
        )

    if resume_enabled:
        try:
            if os.path.exists(state_path):
                os.remove(state_path)
            if os.path.exists(resume_model_path):
                os.remove(resume_model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove sequential resume artifacts: %s", exc)

    mlflow_cfg = config["mlflow"]
    artifact_logging_cfg = mlflow_cfg["artifact_logging"]
    if bool(artifact_logging_cfg["trained_model"]):
        try:
            import mlflow  # type: ignore[import]
            mlflow_tf = __import__("mlflow.tensorflow", fromlist=["log_model"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to import MLFlow TensorFlow integration for sequential model logging: %s", exc)
        else:
            signature = None
            try:
                from mlflow.models import infer_signature  # type: ignore[import]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to import MLFlow infer_signature for sequential model logging: %s",
                    exc,
                )
            else:
                try:
                    inp_shape = model.input_shape
                    _safe = lambda s: tuple(1 if d is None else d for d in s)
                    if isinstance(inp_shape, list):
                        x_sample = [
                            np.zeros(_safe(s), dtype=np.float32) for s in inp_shape
                        ]
                    else:
                        x_sample = np.zeros(_safe(inp_shape), dtype=np.float32)
                    y_pred = model.predict(x_sample, verbose=0)
                    signature = infer_signature(x_sample, y_pred)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to infer MLFlow model signature for sequential model: %s",
                        exc,
                    )

            logger.info("Logging sequentially trained model to MLFlow using mlflow.tensorflow.log_model.")
            try:
                if signature is not None:
                    mlflow_tf.log_model(model, "model", signature=signature)
                else:
                    mlflow_tf.log_model(model, "model")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log sequentially trained model to MLFlow: %s", exc)

    logger.info(
        "Sequential snapshot training completed: processed_windows=%s, total_epoch_steps_logged=%s",
        processed_windows,
        epoch_offset,
    )
    return model


def _run_snapshot_training_pipeline(config: Dict[str, Any]) -> Optional[Any]:
    training_cfg = config["training"]
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])  # per-replica batch size

    # Distributed training setup (opt-in via config)
    runtime_cfg = training_cfg.get("runtime") or {}
    dist_ctx: Optional[DistributedContext] = build_distributed_context(runtime_cfg)
    # MirroredStrategy splits each dataset element across replicas, so the
    # generator must yield global-batch-size samples per step.
    generator_batch_size = (
        dist_ctx.global_batch_size(batch_size) if dist_ctx is not None
        else batch_size
    )

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError(
            "Only model.output.type='two_head_intensity' is supported in snapshot training",
        )

    windows = _resolve_sequential_windows(config)
    if windows is not None and len(windows) > 1:
        return _run_snapshot_training_pipeline_sequential(config, windows, dist_ctx=dist_ctx)

    class_weights_cfg = training_cfg["class_weights"]
    use_class_weights = bool(class_weights_cfg["compute_from_train"])

    snapshot_dataset = prepare_snapshot_dataset(config)
    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Snapshot training skipped: snapshot dataset has no samples.")
        return None

    _enforce_production_sample_cap_snapshot(config, n_samples)

    split_indices = _resolve_snapshot_training_indices(config, n_samples)
    if split_indices is None:
        logger.info("Snapshot training skipped: no effective training samples available.")
        return None
    effective_train_n, val_start, val_end = split_indices

    val_count = max(0, val_end - val_start)

    if not snapshot_dataset.chunks:
        logger.info("Snapshot training skipped: no chunk files found.")
        return None

    first_chunk = snapshot_dataset.chunks[0]
    x_shape = get_chunk_x_shape(first_chunk)
    if len(x_shape) != 5:
        raise ValueError("Snapshot input tensors must have rank 5")
    input_shape = tuple(int(d) for d in x_shape[1:])

    long_term_features = None
    long_term_input_dim: Optional[int] = None
    if is_long_term_enabled(config):
        cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
        long_term_features = compute_long_term_features_for_dataset(
            config,
            snapshot_dataset,
            cadence_seconds=cadence_seconds,
        )
        if long_term_features is None:
            raise ConfigError("Long-term features enabled but computation returned None")
        if long_term_features.shape[0] != n_samples:
            raise ConfigError(
                "Long-term feature rows do not match snapshot dataset sample count: "
                f"features={long_term_features.shape[0]}, samples={n_samples}"
            )
        long_term_input_dim = int(long_term_features.shape[1])

    context = resolve_snapshot_context(config)
    manifest = load_or_create_manifest(context, config)

    normalization_cfg = config["preprocessing"]["normalization"]
    fit_on_train_only = bool(normalization_cfg["fit_on_train_only"])

    train_stats = _get_normalization_stats(
        config,
        context,
        manifest,
        snapshot_dataset,
        0,
        effective_train_n,
        "train",
    )

    mask_start, mask_count = get_mask_channel_info(config)

    if fit_on_train_only or val_count <= 0:
        val_stats = train_stats
    else:
        val_stats = _get_normalization_stats(
            config,
            context,
            manifest,
            snapshot_dataset,
            val_start,
            val_end,
            "val",
        )

    # Compute class weights for imbalanced label handling if configured.
    # Class weights are computed from training data only to avoid data leakage.
    class_weights_up: Optional[Dict[int, float]] = None
    class_weights_down: Optional[Dict[int, float]] = None

    if use_class_weights:
        num_classes = int(output_cfg["num_classes"])

        # Try to load cached label stats from manifest first
        train_label_dist = load_label_stats_from_manifest(manifest, "train")

        if train_label_dist is None:
            # Compute label distribution by streaming through training data
            logger.info(
                "Computing label distribution for class weights (train samples 0 to %s)...",
                effective_train_n,
            )
            train_label_dist = compute_label_distribution(
                snapshot_dataset,
                start_index=0,
                end_index=effective_train_n,
                num_classes=num_classes,
            )
            # Cache in manifest for future runs
            save_label_stats_to_manifest(context, manifest, train_label_dist, "train")

        # Compute class weights from label counts
        class_weights_up = compute_class_weights_from_counts(
            train_label_dist.up_counts,
            num_classes,
        )
        class_weights_down = compute_class_weights_from_counts(
            train_label_dist.down_counts,
            num_classes,
        )

        logger.info(
            "Class weights computed for two-head outputs: up_weights=%s, down_weights=%s",
            {c: round(w, 4) for c, w in sorted(class_weights_up.items())},
            {c: round(w, 4) for c, w in sorted(class_weights_down.items())},
        )

        # Log class weights to MLFlow if available
        try:
            import mlflow  # type: ignore[import]
        except Exception:  # noqa: BLE001
            pass
        else:
            try:
                for c, w in class_weights_up.items():
                    mlflow.log_metric(f"class_weight_up_{c}", float(w))
                for c, w in class_weights_down.items():
                    mlflow.log_metric(f"class_weight_down_{c}", float(w))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log class weights to MLFlow: %s", exc)

    # Build or load model based on fine-tuning configuration.
    # When distributed training is enabled, model must be built/compiled
    # inside strategy.scope() so variables are mirrored across replicas.
    strategy_scope = dist_ctx.scope() if dist_ctx is not None else contextlib.nullcontext()

    fine_tuning_cfg = training_cfg["fine_tuning"]
    fine_tuning_enabled = bool(fine_tuning_cfg["enabled"])

    with strategy_scope:
        if fine_tuning_enabled:
            from .fine_tuning import (
                FineTuningError,
                load_model_from_registry,
                load_model_from_run,
                prepare_fine_tuning,
            )

            use_registry = bool(fine_tuning_cfg["use_model_registry"])

            if use_registry:
                registry_name = fine_tuning_cfg["registry_name"]
                if not registry_name:
                    raise ConfigError(
                        "training.fine_tuning.registry_name is required when use_model_registry is true"
                    )
                stage = str(fine_tuning_cfg["base_model_stage"])
                logger.info(
                    "Fine-tuning enabled: loading model from registry. name=%s, stage=%s",
                    registry_name,
                    stage,
                )
                try:
                    model = load_model_from_registry(registry_name, stage=stage)
                except FineTuningError as exc:
                    raise ConfigError(f"Failed to load base model for fine-tuning: {exc}") from exc
            else:
                run_id = fine_tuning_cfg["base_model_run_id"]
                if not run_id:
                    raise ConfigError(
                        "training.fine_tuning.base_model_run_id is required when fine_tuning.enabled is true "
                        "and use_model_registry is false"
                    )
                logger.info("Fine-tuning enabled: loading model from MLflow run. run_id=%s", run_id)
                try:
                    model = load_model_from_run(run_id)
                except FineTuningError as exc:
                    raise ConfigError(f"Failed to load base model for fine-tuning: {exc}") from exc

            # Prepare the model for fine-tuning (freeze layers, adjust LR)
            try:
                model = prepare_fine_tuning(
                    config,
                    model,
                    input_shape=input_shape,
                    long_term_input_dim=long_term_input_dim,
                )
            except FineTuningError as exc:
                raise ConfigError(f"Failed to prepare model for fine-tuning: {exc}") from exc

            logger.info(
                "Model prepared for fine-tuning: freeze_layers=%s, lr_factor=%s",
                fine_tuning_cfg["freeze_layers"],
                fine_tuning_cfg["learning_rate_factor"],
            )
        else:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model

            model = build_cnn_lstm_model(
                config,
                input_shape=input_shape,
                long_term_input_dim=long_term_input_dim,
            )

    if long_term_features is not None:
        try:
            input_count = len(getattr(model, "inputs", []))
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"Failed to inspect model inputs for long-term features: {exc}") from exc
        if input_count != 2:
            raise ConfigError(
                "Long-term features are enabled but model does not expose two inputs. "
                "Disable model.long_term or rebuild the base model with dual inputs."
            )
    else:
        try:
            input_count = len(getattr(model, "inputs", []))
        except Exception:
            input_count = 1
        if input_count == 2:
            raise ConfigError(
                "Model expects long-term inputs but model.long_term is disabled. "
                "Enable model.long_term and rebuild the snapshot dataset."
            )

    # Log model complexity metrics to MLFlow if available.
    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for model complexity logging: %s", exc)
    else:
        try:
            total_params = int(model.count_params())
            trainable_params = int(
                sum(int(np.prod(w.shape)) for w in getattr(model, "trainable_weights", []))
            )
            non_trainable_params = int(
                sum(int(np.prod(w.shape)) for w in getattr(model, "non_trainable_weights", []))
            )

            approx_flops = float(2 * total_params)

            metrics = {
                "model_total_params": float(total_params),
                "model_trainable_params": float(trainable_params),
                "model_non_trainable_params": float(non_trainable_params),
                "model_approx_flops": approx_flops,
            }

            for name, value in metrics.items():
                try:
                    mlflow.log_metric(name, float(value))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to log MLFlow model complexity metric %s: %s",
                        name,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute or log model complexity metrics: %s", exc)

    callbacks = create_callbacks(config)

    num_classes = int(output_cfg["num_classes"])

    # Helper to create a fresh generator (needed by tf.data.Dataset.from_generator)
    def _make_train_gen() -> Iterator[Tuple[Any, ...]]:
        gen, _ = build_training_generator(
            dataset=snapshot_dataset,
            start_index=0,
            end_index=effective_train_n,
            batch_size=generator_batch_size,
            num_classes=num_classes,
            normalization=train_stats,
            sample_weight_cfg=training_cfg["sample_weighting"],
            mask_start=mask_start,
            mask_count=mask_count,
            class_weights_up=class_weights_up,
            class_weights_down=class_weights_down,
        )
        if long_term_features is not None:
            gen = wrap_generator_with_long_term(
                gen, long_term_features, start_index=0, end_index=effective_train_n,
            )
        return gen

    # Compute steps from a throwaway call (same as build_training_generator returns)
    _, train_steps = build_training_generator(
        dataset=snapshot_dataset,
        start_index=0,
        end_index=effective_train_n,
        batch_size=generator_batch_size,
        num_classes=num_classes,
        normalization=train_stats,
        sample_weight_cfg=training_cfg["sample_weighting"],
        mask_start=mask_start,
        mask_count=mask_count,
        class_weights_up=class_weights_up,
        class_weights_down=class_weights_down,
    )

    if dist_ctx is not None:
        global_batch_size = generator_batch_size
        train_data: Any = wrap_generator_as_dataset(
            generator_factory=_make_train_gen,
            input_shape=input_shape,
            num_classes=num_classes,
            long_term_dim=long_term_input_dim,
            global_batch_size=global_batch_size,
            steps_per_epoch=train_steps,
            distributed_ctx=dist_ctx,
        )
        logger.info(
            "Distributed training: per_replica_batch=%d, global_batch=%d, replicas=%d",
            batch_size,
            global_batch_size,
            dist_ctx.num_replicas,
        )
    else:
        train_data = _make_train_gen()

    writer = None
    try:
        from observability.run_state import get_run_state_writer

        writer = get_run_state_writer()
    except Exception:
        writer = None

    if writer is not None and epochs > 0 and train_steps > 0:
        try:
            from observability.training_progress import create_training_progress_callback

            progress_cb = create_training_progress_callback(writer, epochs=epochs, steps_per_epoch=train_steps)
        except Exception:
            progress_cb = None
        if progress_cb is not None:
            callbacks.append(progress_cb)

    fit_kwargs: Dict[str, Any] = {
        "x": train_data,
        "epochs": epochs,
        "steps_per_epoch": train_steps,
        "callbacks": callbacks,
        "verbose": 1,
    }

    if val_count > 0:
        def _make_val_gen() -> Iterator[Tuple[Any, ...]]:
            gen, _ = build_training_generator(
                dataset=snapshot_dataset,
                start_index=val_start,
                end_index=val_end,
                batch_size=generator_batch_size,
                num_classes=num_classes,
                normalization=val_stats,
                sample_weight_cfg=None,
                mask_start=mask_start,
                mask_count=mask_count,
                class_weights_up=None,
                class_weights_down=None,
            )
            if long_term_features is not None:
                gen = wrap_generator_with_long_term(
                    gen, long_term_features, start_index=val_start, end_index=val_end,
                )
            return gen

        _, val_steps = build_training_generator(
            dataset=snapshot_dataset,
            start_index=val_start,
            end_index=val_end,
            batch_size=generator_batch_size,
            num_classes=num_classes,
            normalization=val_stats,
            sample_weight_cfg=None,
            mask_start=mask_start,
            mask_count=mask_count,
            class_weights_up=None,
            class_weights_down=None,
        )

        if dist_ctx is not None:
            val_data: Any = wrap_generator_as_dataset(
                generator_factory=_make_val_gen,
                input_shape=input_shape,
                num_classes=num_classes,
                long_term_dim=long_term_input_dim,
                global_batch_size=generator_batch_size,
                steps_per_epoch=val_steps,
                distributed_ctx=dist_ctx,
            )
        else:
            val_data = _make_val_gen()

        fit_kwargs["validation_data"] = val_data
        fit_kwargs["validation_steps"] = val_steps

    history = model.fit(**fit_kwargs)

    hpo_metric_value = _extract_hpo_metric_from_history(config, history)
    if hpo_metric_value is not None:
        config["_hpo_last_metric"] = hpo_metric_value

    final_loss = None
    if hasattr(history, "history") and "loss" in history.history:
        loss_values = history.history.get("loss") or []
        if loss_values:
            final_loss = loss_values[-1]

    logger.info(
        "Snapshot training completed. effective_train_n=%s, final_loss=%s",
        effective_train_n,
        final_loss,
    )

    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for snapshot training metrics: %s", exc)
    else:
        if hasattr(history, "history") and isinstance(history.history, dict):
            for metric_name, values in history.history.items():
                try:
                    series = list(values)  # type: ignore[arg-type]
                except TypeError:
                    continue
                for step, value in enumerate(series):
                    try:
                        mlflow.log_metric(metric_name, float(value), step=step)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to log MLFlow metric %s at step %s: %s",
                            metric_name,
                            step,
                            exc,
                        )

        try:
            mlflow.log_param("snapshot_dir", context.snapshot_dir)
            mlflow.log_param("snapshot_config_hash", context.config_hash)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log snapshot metadata to MLFlow: %s", exc)

    # Conditionally log the trained model to MLFlow using the modern Keras format.
    mlflow_cfg = config["mlflow"]
    artifact_logging_cfg = mlflow_cfg["artifact_logging"]
    log_trained_model = bool(artifact_logging_cfg["trained_model"])

    if log_trained_model:
        try:
            import mlflow  # type: ignore[import]
            mlflow_tf = __import__("mlflow.tensorflow", fromlist=["log_model"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to import MLFlow TensorFlow integration for model logging: %s",
                exc,
            )
        else:
            signature = None
            try:
                from mlflow.models import infer_signature  # type: ignore[import]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to import MLFlow infer_signature for model logging: %s",
                    exc,
                )
            else:
                try:
                    sample_n = effective_train_n
                    if sample_n > batch_size:
                        sample_n = batch_size
                    sample_gen, _ = build_training_generator(
                        dataset=snapshot_dataset,
                        start_index=0,
                        end_index=sample_n,
                        batch_size=sample_n,
                        num_classes=int(output_cfg["num_classes"]),
                        normalization=train_stats,
                        sample_weight_cfg=None,
                        mask_start=mask_start,
                        mask_count=mask_count,
                    )
                    if long_term_features is not None:
                        sample_gen = wrap_generator_with_long_term(
                            sample_gen,
                            long_term_features,
                            start_index=0,
                            end_index=sample_n,
                        )
                    batch = next(iter(sample_gen))
                    x_sample = batch[0]
                    y_sample = batch[1]
                    signature = infer_signature(x_sample, model.predict(x_sample))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to infer MLFlow model signature from snapshot training data: %s",
                        exc,
                    )

            logger.info("Logging trained model to MLFlow using mlflow.tensorflow.log_model.")
            try:
                if signature is not None:
                    mlflow_tf.log_model(model, "model", signature=signature)
                else:
                    mlflow_tf.log_model(model, "model")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log trained model to MLFlow: %s", exc)

            model_registry_cfg = mlflow_cfg["model_registry"]
            register_enabled = bool(model_registry_cfg["register_model"])

            if register_enabled:
                try:
                    model_name_pattern = model_registry_cfg["model_name_pattern"]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "MLFlow model registry is enabled but mlflow.model_registry.model_name_pattern is missing or invalid: %s",
                        exc,
                    )
                else:
                    try:
                        data_cfg = config["data"]
                        asset_pairs_cfg = data_cfg["asset_pairs"]
                        target_asset = str(asset_pairs_cfg["target_asset"])
                        architecture_name = str(model_cfg["architecture"])

                        model_name = model_name_pattern.format(
                            asset=target_asset,
                            model=architecture_name,
                        )

                        try:
                            register_model(model, model_name)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to register model '%s' in MLFlow model registry: %s",
                                model_name,
                                exc,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to prepare model name for MLFlow model registry: %s",
                            exc,
                        )

    return model


def run_training_pipeline(config: Dict[str, Any], data_object: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Execute the training pipeline."""

    snapshot_cfg = config["snapshot"]
    if bool(snapshot_cfg["enabled"]):
        return _run_snapshot_training_pipeline(config)

    raise ConfigError(
        "Legacy in-memory training pipeline is disabled. Set snapshot.enabled=true to use the snapshot pipeline."
    )


def pre_build_snapshots(config: Dict[str, Any]) -> None:
    """Pre-build all snapshot datasets required by the current config.

    When sequential training is enabled, this iterates over every time window
    and calls :func:`prepare_snapshot_dataset` for each so the ``.npz`` chunk
    files exist on disk before any parallel HPO worker is spawned. It also
    precomputes trial-invariant artifacts (normalization stats, class-weight
    label stats, long-term features) once per snapshot so HPO workers reuse
    cached data artifacts instead of recomputing them per trial. For
    non-sequential configs a single snapshot is built.

    This is intentionally a **no-op** for already-cached snapshots (the
    underlying :func:`prepare_snapshot_dataset` short-circuits when the
    manifest is marked complete and all chunk files are present).
    """

    if not bool(config.get("snapshot", {}).get("enabled", False)):
        return

    windows = _resolve_sequential_windows(config)
    if windows is not None and len(windows) > 1:
        for idx, (window_start, window_end) in enumerate(windows):
            window_config = copy.deepcopy(config)
            window_config["data"]["time_range"]["start_date"] = window_start
            window_config["data"]["time_range"]["end_date"] = window_end
            logger.info(
                "Pre-building snapshot for sequential window %s/%s: %s -> %s",
                idx + 1,
                len(windows),
                window_start,
                window_end,
            )
            snapshot_dataset = prepare_snapshot_dataset(window_config)
            _precompute_trial_invariant_snapshot_artifacts(window_config, snapshot_dataset)
    else:
        logger.info("Pre-building snapshot for single training window.")
        snapshot_dataset = prepare_snapshot_dataset(config)
        _precompute_trial_invariant_snapshot_artifacts(config, snapshot_dataset)

    logger.info("Snapshot pre-build complete; all chunks cached on disk.")


__all__ = ["pre_build_snapshots", "run_training_pipeline"]
