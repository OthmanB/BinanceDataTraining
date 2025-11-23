"""Training pipeline skeleton.

In later phases this module will:
- Transform DataObject into model-ready tensors
- Build and compile models
- Run training, validation, and evaluation
- Integrate with MLFlow for experiment tracking

Phase 3 only logs that the training pipeline has been invoked.
"""

from typing import Any, Dict, List
import logging

import numpy as np

from preprocessing.train_test_split import chronological_split_indices
from preprocessing.snapshot_sequence_builder import build_top_of_book_sequence_tensor
from mlflow_integration.model_registry import register_model
from .callbacks import create_callbacks


logger = logging.getLogger(__name__)


def run_training_pipeline(config: Dict[str, Any], data_object: Dict[str, Any]) -> None:
    """Execute the training pipeline (Phase 3 skeleton).

    No actual model training is performed yet.
    """

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
    except KeyError:
        snapshot_features = []

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

    if x_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (x_val, y_val)

    history = model.fit(**fit_kwargs)

    final_loss = None
    if hasattr(history, "history") and "loss" in history.history:
        loss_values = history.history.get("loss") or []
        if loss_values:
            final_loss = loss_values[-1]

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
