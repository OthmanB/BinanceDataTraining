"""Training pipeline skeleton.

In later phases this module will:
- Transform DataObject into model-ready tensors
- Build and compile models
- Run training, validation, and evaluation
- Integrate with MLFlow for experiment tracking

Phase 3 only logs that the training pipeline has been invoked.
"""

from typing import Any, Dict
import logging

import numpy as np

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

    if n_samples <= 0:
        logger.info(
            "Training pipeline invoked (Phase 3 minimal). num_samples=0, skipping training.",
        )
        return None

    effective_n = min(n_samples, debug_max_samples)

    logger.info(
        "Training pipeline invoked (Phase 3 minimal). num_samples=%s, debug_max_samples=%s, effective_n=%s",
        n_samples,
        debug_max_samples,
        effective_n,
    )

    model_cfg = config["model"]
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

    # Minimal spatial grid compatible with both the largest kernel and the
    # cumulative effect of all pooling layers.
    min_height = 1
    for ph in pool_heights:
        min_height *= ph

    min_width = 1
    for pw in pool_widths:
        min_width *= pw

    height = max(max(heights), min_height)
    width = max(max(widths), min_width)
    channels = 1

    input_shape = (height, width, channels)

    num_classes = int(model_cfg["output"]["num_classes"])

    # Attempt to build inputs from snapshot-level top-of-book features for the
    # target asset.
    use_real_inputs = False
    x_train = np.zeros((effective_n, height, width, channels), dtype="float32")

    try:
        data_cfg = config["data"]
        asset_pairs_cfg = data_cfg["asset_pairs"]
        target_asset = str(asset_pairs_cfg["target_asset"])
        order_books = data_object["order_books"]
        target_book = order_books.get(target_asset, {})
        snapshot_features = target_book.get("snapshot_features") or []

        if snapshot_features:
            use_real_inputs = True
            logger.info(
                "Using snapshot_features for training inputs (target_asset=%s). available_snapshots=%s",
                target_asset,
                len(snapshot_features),
            )

            # Map best bid/ask prices and quantities for each snapshot into a
            # simple 2x2 patch in the top-left corner of the spatial grid;
            # remaining cells stay zero.
            for i in range(effective_n):
                if i >= len(snapshot_features):
                    break

                features_i = snapshot_features[i]
                if not isinstance(features_i, (list, tuple)) or len(features_i) < 4:
                    continue

                bid_price, bid_qty, ask_price, ask_qty = features_i[:4]

                try:
                    bid_price_f = float(bid_price)
                    bid_qty_f = float(bid_qty)
                    ask_price_f = float(ask_price)
                    ask_qty_f = float(ask_qty)
                except (TypeError, ValueError):
                    # Leave this sample as zeros if parsing fails.
                    continue

                x_train[i, 0, 0, 0] = bid_price_f
                if width > 1:
                    x_train[i, 0, 1, 0] = bid_qty_f
                if height > 1:
                    x_train[i, 1, 0, 0] = ask_price_f
                if height > 1 and width > 1:
                    x_train[i, 1, 1, 0] = ask_qty_f

    except KeyError:
        use_real_inputs = False

    if not use_real_inputs:
        logger.info("Falling back to synthetic inputs for training.")
        x_train = np.random.randn(effective_n, height, width, channels).astype("float32")

    # Labels are built during preprocessing and stored in the DataObject.
    targets = data_object.get("targets")
    if targets is None:
        raise ValueError("data_object.targets must be populated by the preprocessing pipeline")

    labels_list = targets.get("labels")
    if labels_list is None:
        raise ValueError("data_object.targets.labels must be populated by the preprocessing pipeline")

    if len(labels_list) < effective_n:
        raise ValueError(
            "data_object.targets.labels length must be at least effective_n; "
            f"got labels={len(labels_list)}, effective_n={effective_n}",
        )

    labels_arr = np.asarray(labels_list[:effective_n], dtype="int64")

    if labels_arr.min() < 0 or labels_arr.max() >= num_classes:
        raise ValueError(
            "Target label values must be in the range [0, num_classes-1]; "
            f"observed min={labels_arr.min()}, max={labels_arr.max()}, num_classes={num_classes}",
        )

    y_train = np.eye(num_classes, dtype="float32")[labels_arr]

    from models.cnn_lstm_multiclass import build_cnn_lstm_model

    model = build_cnn_lstm_model(config, input_shape=input_shape)
    callbacks = create_callbacks(config)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=float(training_cfg["validation_split"]),
        callbacks=callbacks,
        verbose=1,
    )

    final_loss = None
    if hasattr(history, "history") and "loss" in history.history:
        loss_values = history.history.get("loss") or []
        if loss_values:
            final_loss = loss_values[-1]

    logger.info(
        "Training pipeline completed (Phase 3 minimal). effective_n=%s, final_loss=%s",
        effective_n,
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
                    sample_n = effective_n
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

    return model


__all__ = ["run_training_pipeline"]
