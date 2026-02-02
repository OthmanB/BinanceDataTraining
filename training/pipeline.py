"""Training pipeline skeleton.

In later phases this module will:
- Transform DataObject into model-ready tensors
- Build and compile models
- Run training, validation, and evaluation
- Integrate with MLFlow for experiment tracking

Phase 3 only logs that the training pipeline has been invoked.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import os

import numpy as np

from utils.config_loader import ConfigError
from preprocessing.train_test_split import chronological_split_indices
from preprocessing.snapshot_sequence_builder import (
    build_top_of_book_sequence_tensor,
    build_hybrid_depth_sequence_tensor,
)
from preprocessing.normalizer import create_normalizer_from_config
from preprocessing.feature_engineering import FeatureEngineer
from mlflow_integration.model_registry import register_model
from .dataset_cache import compute_dataset_hash, cache_dataset_to_npz
from .snapshot_dataset import (
    NormalizationStats,
    build_training_generator,
    compute_normalization_stats,
    load_normalization_stats,
    prepare_snapshot_dataset,
    save_normalization_stats,
)
from .snapshot_store import load_or_create_manifest, resolve_snapshot_context, save_manifest
from .callbacks import create_callbacks
from .class_weights import compute_class_weights


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


def _enforce_production_sample_cap_snapshot(config: Dict[str, Any], n_samples: int) -> None:
    run_mode_cfg = config.get("run_mode", {})
    mode = str(run_mode_cfg.get("mode"))
    if mode != "production":
        return

    training_cfg = config.get("training", {})
    debug_max_samples = int(training_cfg.get("debug_max_samples", 0))
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

    if os.path.exists(stats_path):
        stats = load_normalization_stats(stats_path)
    else:
        stats = compute_normalization_stats(dataset, start_index, end_index, method)
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


def _run_snapshot_training_pipeline(config: Dict[str, Any]) -> Optional[Any]:
    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])

    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    configured_val_split = float(training_cfg["validation_split"])
    if abs(configured_val_split - validation_ratio) > 1e-6:
        raise ValueError(
            "training.validation_split must match preprocessing.train_test_split.validation_ratio in this phase",
        )

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError(
            "Only model.output.type='two_head_intensity' is supported in snapshot training",
        )

    class_weights_cfg = training_cfg.get("class_weights", {})
    if isinstance(class_weights_cfg, dict) and class_weights_cfg.get("compute_from_train"):
        raise ConfigError(
            "training.class_weights.compute_from_train is not supported for multi-output models in this phase. "
            "Disable it or implement per-output sample weighting before enabling.",
        )

    snapshot_dataset = prepare_snapshot_dataset(config)
    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Snapshot training skipped: snapshot dataset has no samples.")
        return None

    _enforce_production_sample_cap_snapshot(config, n_samples)

    train_end, val_end, _ = _compute_split_boundaries(
        n_samples,
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if train_end <= 0:
        logger.info("Snapshot training skipped: no training samples available after split.")
        return None

    effective_train_n = min(train_end, debug_max_samples)
    if effective_train_n <= 0:
        logger.info("Snapshot training skipped: debug_max_samples=%s", debug_max_samples)
        return None

    val_start = train_end
    val_end = min(val_end, n_samples)
    val_count = max(0, val_end - val_start)

    if not snapshot_dataset.chunks:
        logger.info("Snapshot training skipped: no chunk files found.")
        return None

    first_chunk = snapshot_dataset.chunks[0]
    with np.load(first_chunk.file_path, mmap_mode="r") as npz:
        x_shape = npz["x"].shape
    if len(x_shape) != 5:
        raise ValueError("Snapshot input tensors must have rank 5")
    input_shape = tuple(int(d) for d in x_shape[1:])

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

    if fit_on_train_only:
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

    from models.cnn_lstm_multiclass import build_cnn_lstm_model

    model = build_cnn_lstm_model(config, input_shape=input_shape)

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

    train_gen, train_steps = build_training_generator(
        dataset=snapshot_dataset,
        start_index=0,
        end_index=effective_train_n,
        batch_size=batch_size,
        num_classes=int(output_cfg["num_classes"]),
        normalization=train_stats,
        sample_weight_cfg=training_cfg.get("sample_weighting"),
    )

    fit_kwargs: Dict[str, Any] = {
        "x": train_gen,
        "epochs": epochs,
        "steps_per_epoch": train_steps,
        "callbacks": callbacks,
        "verbose": 1,
    }

    if val_count > 0:
        val_gen, val_steps = build_training_generator(
            dataset=snapshot_dataset,
            start_index=val_start,
            end_index=val_end,
            batch_size=batch_size,
            num_classes=int(output_cfg["num_classes"]),
            normalization=val_stats,
            sample_weight_cfg=None,
        )
        fit_kwargs["validation_data"] = val_gen
        fit_kwargs["validation_steps"] = val_steps

    history = model.fit(**fit_kwargs)

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
    try:
        mlflow_cfg = config.get("mlflow", {})
        artifact_logging_cfg = mlflow_cfg.get("artifact_logging", {})
        log_trained_model = bool(artifact_logging_cfg.get("trained_model"))
    except Exception:  # noqa: BLE001
        log_trained_model = False

    if log_trained_model:
        try:
            import mlflow  # type: ignore[import]
            import mlflow.tensorflow  # type: ignore[import]
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
                    mlflow.tensorflow.log_model(model, "model", signature=signature)  # type: ignore[attr-defined]
                else:
                    mlflow.tensorflow.log_model(model, "model")  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log trained model to MLFlow: %s", exc)

            try:
                model_registry_cfg = mlflow_cfg.get("model_registry", {})
                register_enabled = bool(model_registry_cfg.get("register_model"))
            except Exception:  # noqa: BLE001
                register_enabled = False

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
    """Execute the training pipeline (Phase 3 skeleton).

    No actual model training is performed yet.
    """

    snapshot_cfg = config.get("snapshot", {})
    if isinstance(snapshot_cfg, dict) and snapshot_cfg.get("enabled"):
        return _run_snapshot_training_pipeline(config)

    if data_object is None:
        raise ValueError("data_object is required when snapshot.enabled is false")

    metadata = data_object["metadata"]
    n_samples = int(metadata["num_samples"])

    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    missing_snapshot_strategy = str(training_cfg["missing_snapshot_strategy"])

    if missing_snapshot_strategy not in ("fail", "skip", "synthetic"):
        raise ValueError(
            "training.missing_snapshot_strategy must be one of 'fail', 'skip', or 'synthetic'",
        )

    if n_samples <= 0:
        logger.info(
            "Training pipeline invoked (Phase 3 minimal). num_samples=0, skipping training.",
        )
        return None

    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    configured_val_split = float(training_cfg["validation_split"])
    if abs(configured_val_split - validation_ratio) > 1e-6:
        raise ValueError(
            "training.validation_split must match preprocessing.train_test_split.validation_ratio in this phase",
        )

    train_idx, val_idx, _ = chronological_split_indices(
        n_samples,
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if not train_idx:
        logger.info(
            "Training pipeline invoked (Phase 3 minimal). no training samples available after chronological split.",
        )
        return None

    effective_train_n = min(len(train_idx), debug_max_samples)
    if effective_train_n <= 0:
        logger.info(
            "Training pipeline invoked (Phase 3 minimal). debug_max_samples=%s resulted in no training samples.",
            debug_max_samples,
        )
        return None

    train_indices = train_idx[:effective_train_n]
    val_indices = val_idx

    logger.info(
        "Training pipeline invoked (Phase 3 minimal). num_samples=%s, train=%s, val=%s, debug_max_samples=%s, effective_train_n=%s",
        n_samples,
        len(train_idx),
        len(val_idx),
        debug_max_samples,
        effective_train_n,
    )

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError(
            "Only model.output.type='two_head_intensity' is supported in this phase of the training pipeline",
        )

    class_weights_cfg = training_cfg.get("class_weights", {})
    if isinstance(class_weights_cfg, dict) and class_weights_cfg.get("compute_from_train"):
        raise ConfigError(
            "training.class_weights.compute_from_train is not supported for multi-output models in this phase. "
            "Disable it or implement per-output sample weighting before enabling.",
        )

    cnn_cfg = model_cfg["cnn"]
    kernel_sizes = cnn_cfg["kernel_sizes"]
    pool_sizes = cnn_cfg["pool_sizes"]

    if not isinstance(kernel_sizes, list) or not kernel_sizes:
        raise ValueError("model.cnn.kernel_sizes must be a non-empty list in configuration")
    if not isinstance(pool_sizes, list) or not pool_sizes:
        raise ValueError("model.cnn.pool_sizes must be a non-empty list in configuration")

    heights = [int(k[0]) for k in kernel_sizes]
    widths = [int(k[1]) for k in kernel_sizes]

    pool_heights = [int(p[0]) for p in pool_sizes]
    pool_widths = [int(p[1]) for p in pool_sizes]

    min_height = 1
    for ph in pool_heights:
        min_height *= ph

    min_width = 1
    for pw in pool_widths:
        min_width *= pw

    height = max(max(heights), min_height)
    width = max(max(widths), min_width)
    channels = 1

    num_classes = int(output_cfg["num_classes"])

    snapshot_features: List[Any]
    snapshot_features = []
    target_asset = None

    try:
        data_cfg = config["data"]
        asset_pairs_cfg = data_cfg["asset_pairs"]
        target_asset = str(asset_pairs_cfg["target_asset"])
        order_books = data_object["order_books"]
        target_book = order_books.get(target_asset, {})
        snapshot_features = target_book.get("snapshot_features") or []
        snapshot_depth_data = target_book.get("snapshot_depth_data") or []
    except KeyError:
        snapshot_features = []
        snapshot_depth_data = []

    x_train = None
    x_val = None

    if snapshot_features:
        logger.info(
            "Using snapshot_features for training inputs (target_asset=%s). available_snapshots=%s",
            target_asset,
            len(snapshot_features),
        )

        anchor_indices = metadata.get("anchor_indices")
        if anchor_indices is None:
            raise ValueError(
                "metadata.anchor_indices must be populated by the preprocessing pipeline when snapshot_features are present",
            )

        # Check representation type to decide which tensor builder to use
        order_book_cfg = data_cfg.get("order_book", {})
        representation = str(order_book_cfg.get("representation", "top_of_book"))

        if representation == "hybrid":
            # Use hybrid depth tensor builder
            if not snapshot_depth_data:
                raise ValueError(
                    "data.order_book.representation is 'hybrid' but snapshot_depth_data is empty. "
                    "Ensure the preprocessing pipeline collected full depth data.",
                )

            logger.info(
                "Building hybrid depth tensors (representation='hybrid'). depth_snapshots=%s",
                len(snapshot_depth_data),
            )

            x_train = build_hybrid_depth_sequence_tensor(
                config=config,
                snapshot_depth_data=snapshot_depth_data,
                anchor_indices=list(anchor_indices),
                sample_indices=train_indices,
            )

            if val_indices:
                x_val = build_hybrid_depth_sequence_tensor(
                    config=config,
                    snapshot_depth_data=snapshot_depth_data,
                    anchor_indices=list(anchor_indices),
                    sample_indices=val_indices,
                )

        else:
            # Use top-of-book tensor builder (default)
            x_train = build_top_of_book_sequence_tensor(
                config=config,
                snapshot_features=snapshot_features,
                anchor_indices=list(anchor_indices),
                sample_indices=train_indices,
                height=height,
                width=width,
                channels=channels,
            )

            if val_indices:
                x_val = build_top_of_book_sequence_tensor(
                    config=config,
                    snapshot_features=snapshot_features,
                    anchor_indices=list(anchor_indices),
                    sample_indices=val_indices,
                    height=height,
                    width=width,
                    channels=channels,
                )

    else:
        if missing_snapshot_strategy == "fail":
            raise ValueError(
                "No snapshot_features available for training inputs for target asset; "
                "set training.missing_snapshot_strategy to 'skip' or 'synthetic' to change this behavior.",
            )

        if missing_snapshot_strategy == "skip":
            logger.info(
                "No snapshot_features available for training inputs; skipping training stage because "
                "training.missing_snapshot_strategy='skip'.",
            )
            return None

        data_cfg = config["data"]
        time_range_cfg = data_cfg["time_range"]
        cadence_seconds = int(time_range_cfg["cadence_seconds"])
        if cadence_seconds <= 0:
            raise ValueError("data.time_range.cadence_seconds must be positive")

        targets_cfg = config["targets"]
        visible_window_seconds = int(targets_cfg["visible_window_seconds"])
        if visible_window_seconds <= 0:
            raise ValueError("targets.visible_window_seconds must be positive")
        if visible_window_seconds % cadence_seconds != 0:
            raise ValueError(
                "targets.visible_window_seconds must be an integer multiple of data.time_range.cadence_seconds",
            )

        window_steps = visible_window_seconds // cadence_seconds
        if window_steps <= 0:
            raise ValueError(
                "Derived visible window length in steps must be at least one snapshot; "
                f"visible_window_seconds={visible_window_seconds}, cadence_seconds={cadence_seconds}",
            )

        logger.info(
            "No snapshot_features available for training inputs; using synthetic inputs because "
            "training.missing_snapshot_strategy='synthetic'.",
        )
        x_train = np.random.randn(effective_train_n, window_steps, height, width, channels).astype("float32")
        if val_indices:
            x_val = np.random.randn(len(val_indices), window_steps, height, width, channels).astype("float32")

    if x_train is None:
        raise ValueError("Training inputs could not be constructed; x_train is None")

    # Optionally integrate feature engineering derived features into the input.
    # This computes momentum features using anchor indices and mid_prices/volumes,
    # then broadcasts and concatenates along the channel dimension.
    fe_cfg = config["preprocessing"].get("feature_engineering", {})
    if isinstance(fe_cfg, dict) and fe_cfg.get("enabled"):
        try:
            feature_engineer = FeatureEngineer(config)

            # Retrieve precomputed order book features and volume proxy from transformer
            snapshot_derived_features = target_book.get("snapshot_derived_features")
            volume_proxy = target_book.get("volume_proxy")
            mid_prices_list = target_book.get("mid_prices")

            if snapshot_derived_features and volume_proxy and mid_prices_list:
                mid_prices_arr = np.asarray(mid_prices_list, dtype="float64")
                volumes_arr = np.asarray(volume_proxy, dtype="float64")
                anchor_indices_list = list(anchor_indices)
                cadence_seconds = int(data_cfg["time_range"]["cadence_seconds"])

                # Compute all features for all samples
                all_features = feature_engineer.compute_all_features(
                    snapshot_depth_data=snapshot_depth_data,
                    mid_prices=mid_prices_arr,
                    anchor_indices=anchor_indices_list,
                    cadence_seconds=cadence_seconds,
                )

                if all_features is not None and all_features.shape[0] > 0:
                    n_features = all_features.shape[1]

                    # Extract features for train and val indices
                    fe_train = all_features[train_indices].astype("float32")
                    fe_val = None
                    if val_indices:
                        fe_val = all_features[val_indices].astype("float32")

                    # Broadcast across time and spatial dimensions and concatenate
                    if x_train.ndim == 5:
                        _, t_steps, h_dim, w_dim, _ = x_train.shape
                        fe_train_exp = fe_train[:, None, None, None, :]
                        fe_train_broadcast = np.broadcast_to(
                            fe_train_exp,
                            (fe_train.shape[0], t_steps, h_dim, w_dim, n_features),
                        )
                        x_train = np.concatenate(
                            [x_train, fe_train_broadcast.astype("float32")], axis=-1
                        )

                        if x_val is not None and fe_val is not None:
                            fe_val_exp = fe_val[:, None, None, None, :]
                            fe_val_broadcast = np.broadcast_to(
                                fe_val_exp,
                                (fe_val.shape[0], t_steps, h_dim, w_dim, n_features),
                            )
                            x_val = np.concatenate(
                                [x_val, fe_val_broadcast.astype("float32")], axis=-1
                            )

                        logger.info(
                            "Integrated feature engineering features into training inputs: "
                            "n_features=%s, x_train.shape=%s",
                            n_features,
                            x_train.shape,
                        )
            else:
                logger.info(
                    "Feature engineering skipped: missing snapshot_derived_features, "
                    "volume_proxy, or mid_prices from preprocessing."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Feature engineering integration failed: %s. Continuing without derived features.",
                exc,
            )

    # Optionally integrate temporal features into the input channels according
    # to the model.input_representation.temporal_features configuration.
    ir_cfg = model_cfg.get("input_representation")
    if ir_cfg is None:
        raise ValueError("model.input_representation must be defined in configuration")
    tf_cfg = ir_cfg.get("temporal_features")
    if tf_cfg is None:
        raise ValueError("model.input_representation.temporal_features must be defined in configuration")
    integration_mode = str(tf_cfg["integration_mode"])
    use_local = bool(tf_cfg["use_local_features"])
    use_global = bool(tf_cfg["use_global_features"])

    if integration_mode not in ("none", "concat_channels"):
        raise ValueError(
            "model.input_representation.temporal_features.integration_mode must be one of 'none' or 'concat_channels'; "
            f"got {integration_mode!r}",
        )

    if integration_mode == "concat_channels":
        temporal_features = data_object.get("temporal_features")
        if not isinstance(temporal_features, dict):
            raise ValueError(
                "data_object.temporal_features must be a dict when temporal feature integration is enabled; "
                f"got {type(temporal_features)!r}",
            )

        local_arr = temporal_features.get("local")
        global_arr = temporal_features.get("global")

        if use_local:
            if local_arr is None:
                raise ValueError(
                    "Temporal feature integration is configured to use_local_features=True but "
                    "data_object.temporal_features.local is missing.",
                )
        if use_global:
            if global_arr is None:
                raise ValueError(
                    "Temporal feature integration is configured to use_global_features=True but "
                    "data_object.temporal_features.global is missing.",
                )

        feature_matrices = []
        if use_local and local_arr is not None:
            local_np = np.asarray(local_arr, dtype="float32")
            feature_matrices.append(local_np)
        if use_global and global_arr is not None:
            global_np = np.asarray(global_arr, dtype="float32")
            feature_matrices.append(global_np)

        if feature_matrices:
            tf_all = np.concatenate(feature_matrices, axis=1)
            if tf_all.shape[0] != n_samples:
                raise ValueError(
                    "Temporal feature matrices must have one row per sample; "
                    f"got tf_all.shape={tf_all.shape}, num_samples={n_samples}",
                )

            tf_train = tf_all[train_indices]
            tf_train = np.asarray(tf_train, dtype="float32")
            if val_indices:
                tf_val = tf_all[val_indices]
                tf_val = np.asarray(tf_val, dtype="float32")
            else:
                tf_val = None

            # Broadcast temporal features across time and spatial dimensions and
            # concatenate them along the channel axis.
            if x_train.ndim != 5:
                raise ValueError(
                    "Training input tensor must have rank 5 before temporal feature integration; "
                    f"got x_train.ndim={x_train.ndim}, shape={x_train.shape!r}",
                )

            _, t_steps, h_dim, w_dim, _ = x_train.shape
            tf_train_exp = tf_train[:, None, None, None, :]
            tf_train_broadcast = np.broadcast_to(
                tf_train_exp,
                (tf_train.shape[0], t_steps, h_dim, w_dim, tf_train.shape[1]),
            )
            x_train = np.concatenate([x_train, tf_train_broadcast.astype("float32")], axis=-1)

            if x_val is not None and tf_val is not None:
                tf_val_exp = tf_val[:, None, None, None, :]
                tf_val_broadcast = np.broadcast_to(
                    tf_val_exp,
                    (tf_val.shape[0], t_steps, h_dim, w_dim, tf_val.shape[1]),
                )
                x_val = np.concatenate([x_val, tf_val_broadcast.astype("float32")], axis=-1)

            logger.info(
                "Integrated temporal features into training inputs via concat_channels: "
                "n_samples=%s, feature_dim=%s",
                n_samples,
                tf_all.shape[1],
            )

    # Apply normalization to the input tensors based on preprocessing.normalization config.
    # Normalization statistics are computed on training data only when fit_on_train_only is true.
    normalization_cfg = config["preprocessing"]["normalization"]
    fit_on_train_only = bool(normalization_cfg["fit_on_train_only"])

    normalizer = create_normalizer_from_config(config)

    if fit_on_train_only:
        # Fit normalizer on training data only, then transform both train and val
        x_train = normalizer.fit_transform(x_train)
        if x_val is not None:
            x_val = normalizer.transform(x_val)
        logger.info(
            "Applied normalization (method=%s, fit_on_train_only=True): x_train.shape=%s",
            normalizer.method,
            x_train.shape,
        )
    else:
        # Fit and transform training data, fit and transform validation data separately
        x_train = normalizer.fit_transform(x_train)
        if x_val is not None:
            # Create a new normalizer for validation data
            val_normalizer = create_normalizer_from_config(config)
            x_val = val_normalizer.fit_transform(x_val)
        logger.info(
            "Applied normalization (method=%s, fit_on_train_only=False): x_train.shape=%s",
            normalizer.method,
            x_train.shape,
        )

    # Infer the model input shape from the constructed training tensor. This
    # must have rank 5: (N, T, H, W, C), so input_shape=(T, H, W, C).
    if x_train.ndim != 5:
        raise ValueError(
            "Training input tensor must have shape (N, T, H, W, C); "
            f"got x_train.ndim={x_train.ndim}, shape={x_train.shape!r}",
        )
    input_shape = tuple(int(d) for d in x_train.shape[1:])

    # Labels are built during preprocessing and stored in the DataObject.
    targets = data_object.get("targets")
    if targets is None:
        raise ValueError("data_object.targets must be populated by the preprocessing pipeline")

    labels_up_list = targets.get("labels_up_intensity")
    labels_down_list = targets.get("labels_down_intensity")
    if labels_up_list is None or labels_down_list is None:
        raise ValueError(
            "data_object.targets.labels_up_intensity and labels_down_intensity must be populated by the preprocessing pipeline",
        )

    if len(labels_up_list) < n_samples or len(labels_down_list) < n_samples:
        raise ValueError(
            "Intensity label arrays must have length at least metadata.num_samples; "
            f"got labels_up={len(labels_up_list)}, labels_down={len(labels_down_list)}, num_samples={n_samples}",
        )

    labels_up_arr = np.asarray(labels_up_list[:n_samples], dtype="int64")
    labels_down_arr = np.asarray(labels_down_list[:n_samples], dtype="int64")

    if labels_up_arr.min() < 0 or labels_up_arr.max() >= num_classes:
        raise ValueError(
            "labels_up_intensity values must be in the range [0, num_classes-1]; "
            f"observed min={labels_up_arr.min()}, max={labels_up_arr.max()}, num_classes={num_classes}",
        )
    if labels_down_arr.min() < 0 or labels_down_arr.max() >= num_classes:
        raise ValueError(
            "labels_down_intensity values must be in the range [0, num_classes-1]; "
            f"observed min={labels_down_arr.min()}, max={labels_down_arr.max()}, num_classes={num_classes}",
        )

    y_up_train = np.eye(num_classes, dtype="float32")[labels_up_arr[train_indices]]
    y_down_train = np.eye(num_classes, dtype="float32")[labels_down_arr[train_indices]]
    y_train = [y_up_train, y_down_train]

    y_val = None
    if val_indices:
        y_up_val = np.eye(num_classes, dtype="float32")[labels_up_arr[val_indices]]
        y_down_val = np.eye(num_classes, dtype="float32")[labels_down_arr[val_indices]]
        y_val = [y_up_val, y_down_val]

    # Compute class weights for handling imbalanced labels.
    # Weights are computed on training indices only to avoid data leakage.
    class_weight_dict = None
    class_weights_cfg = training_cfg.get("class_weights", {})
    if isinstance(class_weights_cfg, dict) and class_weights_cfg.get("compute_from_train"):
        train_labels_up = labels_up_arr[train_indices]
        train_labels_down = labels_down_arr[train_indices]

        up_weights = compute_class_weights(train_labels_up, num_classes)
        down_weights = compute_class_weights(train_labels_down, num_classes)

        class_weight_dict = {
            "up_intensity": up_weights,
            "down_intensity": down_weights,
        }

        logger.info(
            "Class weights computed from training data: up_weights=%s, down_weights=%s",
            {c: round(w, 4) for c, w in sorted(up_weights.items())},
            {c: round(w, 4) for c, w in sorted(down_weights.items())},
        )

    # Optional time-weighted sampling using exponential decay based on sample age
    # in days, configured via training.sample_weighting.
    sample_weight_train = None
    try:
        sw_cfg = training_cfg["sample_weighting"]
    except KeyError:
        sw_cfg = None

    if isinstance(sw_cfg, dict):
        sw_enabled = bool(sw_cfg["enabled"])
        if sw_enabled:
            method = str(sw_cfg["method"])
            if method != "exponential_decay":
                raise ValueError(
                    "training.sample_weighting.method must be 'exponential_decay' when enabled; "
                    f"got {method!r}",
                )

            apply_to = str(sw_cfg["apply_to"])
            if apply_to != "loss_function":
                raise ValueError(
                    "training.sample_weighting.apply_to must be 'loss_function' when sample weighting is enabled; "
                    f"got {apply_to!r}",
                )

            half_life_days = int(sw_cfg["half_life_days"])
            if half_life_days <= 0:
                raise ValueError("training.sample_weighting.half_life_days must be a positive integer when enabled")

            anchor_indices_all = metadata.get("anchor_indices") or []
            if not anchor_indices_all:
                raise ValueError(
                    "training.sample_weighting.enabled is true but metadata.anchor_indices is missing or empty; "
                    "temporal feature preprocessing must populate anchor_indices before training.",
                )

            if len(anchor_indices_all) != n_samples:
                raise ValueError(
                    "Length of metadata.anchor_indices must match metadata.num_samples when sample weighting is enabled; "
                    f"got len(anchor_indices)={len(anchor_indices_all)}, num_samples={n_samples}",
                )

            data_cfg = config["data"]
            asset_pairs_cfg = data_cfg["asset_pairs"]
            target_asset_sw = str(asset_pairs_cfg["target_asset"])

            order_books_sw = data_object.get("order_books", {})
            target_book_sw = order_books_sw.get(target_asset_sw, {})
            snapshot_timestamps_sw = target_book_sw.get("snapshot_timestamps") or []

            if not snapshot_timestamps_sw:
                raise ValueError(
                    "training.sample_weighting.enabled is true but order_books[target_asset].snapshot_timestamps is missing "
                    "or empty; snapshot_timestamps must be populated before training.",
                )

            ts_array = np.asarray(snapshot_timestamps_sw, dtype="datetime64[D]")
            if ts_array.ndim != 1:
                raise ValueError(
                    "order_books[target_asset].snapshot_timestamps must be a one-dimensional sequence when sample weighting is enabled",
                )

            anchor_arr = np.asarray(anchor_indices_all, dtype="int64")
            if anchor_arr.ndim != 1:
                raise ValueError("metadata.anchor_indices must be a one-dimensional list of integers when sample weighting is enabled")

            if anchor_arr.min() < 0 or anchor_arr.max() >= ts_array.shape[0]:
                raise ValueError(
                    "metadata.anchor_indices must reference valid snapshot indices when sample weighting is enabled; "
                    f"got min={anchor_arr.min()}, max={anchor_arr.max()}, num_snapshots={ts_array.shape[0]}",
                )

            anchor_ts = ts_array[anchor_arr]
            days = anchor_ts.astype("datetime64[D]").astype("int64")
            current_day = int(days.max())
            age_days = (current_day - days).astype("float64")

            decay_const = np.log(2.0) / float(half_life_days)
            weights_all = np.exp(-age_days * decay_const).astype("float32")

            sample_weight_train = weights_all[train_indices]
            if sample_weight_train.shape[0] != effective_train_n:
                raise ValueError(
                    "Sample weight vector length must match effective_train_n; "
                    f"got sample_weight_train.shape[0]={sample_weight_train.shape[0]}, effective_train_n={effective_train_n}",
                )

            logger.info(
                "Sample weighting enabled (method=exponential_decay, half_life_days=%s). "
                "train_weight_stats=(min=%s, max=%s, mean=%s, std=%s)",
                half_life_days,
                float(sample_weight_train.min()),
                float(sample_weight_train.max()),
                float(sample_weight_train.mean()),
                float(sample_weight_train.std()),
            )

            try:
                import mlflow  # type: ignore[import]
            except Exception:  # noqa: BLE001
                pass
            else:
                try:
                    mlflow.log_metric("sample_weight_min", float(sample_weight_train.min()))
                    mlflow.log_metric("sample_weight_max", float(sample_weight_train.max()))
                    mlflow.log_metric("sample_weight_mean", float(sample_weight_train.mean()))
                    mlflow.log_metric("sample_weight_std", float(sample_weight_train.std()))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to log sample weight diagnostics to MLFlow: %s", exc)

    dataset_hash = compute_dataset_hash(x_train, y_train, x_val, y_val)
    dataset_cache_path = cache_dataset_to_npz(
        config=config,
        dataset_hash=dataset_hash,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for dataset hash logging: %s", exc)
    else:
        try:
            training_cfg = config["training"]
            cache_cfg = training_cfg["dataset_cache"]
            dataset_version = str(cache_cfg["version"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read training.dataset_cache.version for dataset logging: %s", exc)
        else:
            try:
                mlflow.log_param("dataset_version", dataset_version)
                mlflow.log_param("dataset_hash", dataset_hash)
                if dataset_cache_path is not None:
                    mlflow.log_param("dataset_cache_path", dataset_cache_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log dataset hash/version parameters to MLFlow: %s", exc)

    from models.cnn_lstm_multiclass import build_cnn_lstm_model

    model = build_cnn_lstm_model(config, input_shape=input_shape)

    # Log model complexity metrics (parameter counts and an approximate FLOPs
    # estimate) to MLFlow if it is available.
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

            # Simple approximate FLOPs estimate proportional to parameter
            # count; this is intended as a coarse complexity indicator.
            approx_flops = float(2 * total_params)

            complexity_metrics = {
                "model_total_params": float(total_params),
                "model_trainable_params": float(trainable_params),
                "model_non_trainable_params": float(non_trainable_params),
                "model_approx_flops": approx_flops,
            }

            for name, value in complexity_metrics.items():
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

    fit_kwargs = {
        "x": x_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "verbose": 1,
    }

    if sample_weight_train is not None:
        fit_kwargs["sample_weight"] = [sample_weight_train, sample_weight_train]

    if class_weight_dict is not None:
        fit_kwargs["class_weight"] = class_weight_dict

    if x_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (x_val, y_val)

    history = model.fit(**fit_kwargs)

    final_loss = None
    if hasattr(history, "history") and "loss" in history.history:
        loss_values = history.history.get("loss") or []
        if loss_values:
            final_loss = loss_values[-1]

    hpo_metric_value = None
    try:
        hpo_cfg = config.get("hyperparameter_optimization", {})
        if isinstance(hpo_cfg, dict) and hpo_cfg.get("enabled"):
            metric_name = str(hpo_cfg["metric"])
            if hasattr(history, "history") and isinstance(history.history, dict):
                series = history.history.get(metric_name)
                if series:
                    try:
                        hpo_metric_value = float(series[-1])
                    except (TypeError, ValueError):  # noqa: BLE001
                        hpo_metric_value = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to compute hyperparameter optimization metric from training history: %s", exc)

    if hpo_metric_value is not None:
        metadata["last_hpo_metric"] = hpo_metric_value
        data_object["metadata"] = metadata

    logger.info(
        "Training pipeline completed (Phase 3 minimal). effective_train_n=%s, final_loss=%s",
        effective_train_n,
        final_loss,
    )

    # Explicitly log training metrics to MLFlow, approximating the behavior of
    # TensorFlow autologging but using the History object returned by Keras.
    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for metric logging: %s", exc)
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

    # Conditionally log the trained model to MLFlow using the modern Keras format.
    try:
        mlflow_cfg = config.get("mlflow", {})
        artifact_logging_cfg = mlflow_cfg.get("artifact_logging", {})
        log_trained_model = bool(artifact_logging_cfg.get("trained_model"))
    except Exception:  # noqa: BLE001
        log_trained_model = False

    if log_trained_model:
        try:
            import mlflow  # type: ignore[import]
            import mlflow.tensorflow  # type: ignore[import]
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
                    x_sample = x_train[:sample_n]
                    y_sample = model.predict(x_sample)
                    signature = infer_signature(x_sample, y_sample)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to infer MLFlow model signature from training data: %s",
                        exc,
                    )

            logger.info("Logging trained model to MLFlow using mlflow.tensorflow.log_model.")
            try:
                if signature is not None:
                    mlflow.tensorflow.log_model(model, "model", signature=signature)  # type: ignore[attr-defined]
                else:
                    mlflow.tensorflow.log_model(model, "model")  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to log trained model to MLFlow: %s", exc)

            # Optionally register the model in the MLFlow model registry using
            # the configuration-driven model name pattern.
            try:
                model_registry_cfg = mlflow_cfg.get("model_registry", {})
                register_enabled = bool(model_registry_cfg.get("register_model"))
            except Exception:  # noqa: BLE001
                register_enabled = False

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
                        model_cfg = config["model"]
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


__all__ = ["run_training_pipeline"]
