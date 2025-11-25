"""Canonical CNN+LSTM model architecture (skeleton).

This module defines a builder for the multi-class CNN+LSTM model described in the
technical specifications. The actual model is not used in Phase 3 yet, but the
builder is provided for completeness and future integration.
"""

from typing import Any, Dict, Tuple
import logging


logger = logging.getLogger(__name__)


def build_cnn_lstm_model(config: Dict[str, Any], input_shape: Tuple[int, ...]):
    """Build a CNN+LSTM Keras model according to the configuration.

    Parameters
    ----------
    config:
        Global configuration dictionary loaded from YAML.
    input_shape:
        Shape of the main input tensor (excluding batch dimension).

    Notes
    -----
    - TensorFlow/Keras is imported lazily inside this function to avoid import
      errors at startup if the dependency is not installed yet.
    - This function is not invoked in Phase 3; it will be wired into the
      training pipeline in later phases.
    """

    model_cfg = config.get("model", {})

    try:
        import tensorflow as tf  # type: ignore[import]
        from tensorflow import keras  # type: ignore[import]
        from tensorflow.keras import layers  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TensorFlow is required to build the CNN+LSTM model but could not be imported."
        ) from exc

    if len(input_shape) != 4:
        raise ValueError(
            "build_cnn_lstm_model expects input_shape=(T, H, W, C); "
            f"got input_shape={input_shape!r}",
        )

    cnn_cfg = model_cfg.get("cnn", {})
    lstm_cfg = model_cfg.get("lstm", {})
    dense_cfg = model_cfg.get("dense", {})
    output_cfg = model_cfg.get("output", {})

    num_layers = int(cnn_cfg.get("num_layers"))
    filters = cnn_cfg.get("filters")
    kernel_sizes = cnn_cfg.get("kernel_sizes")
    pool_sizes = cnn_cfg.get("pool_sizes")
    activation = cnn_cfg.get("activation")
    dropout_rates = cnn_cfg.get("dropout_rates")

    if not (isinstance(filters, list) and isinstance(kernel_sizes, list) and isinstance(pool_sizes, list)):
        raise ValueError("cnn.filters, cnn.kernel_sizes, and cnn.pool_sizes must be lists in config.model.cnn")

    inputs = keras.Input(shape=input_shape, name="main_input")
    x = inputs

    # Apply the convolutional block independently to each temporal slice using
    # TimeDistributed wrappers, so that spatial microstructure is encoded per
    # snapshot while weights are shared across time.
    for i in range(num_layers):
        f = int(filters[i])
        k = kernel_sizes[i]
        p = pool_sizes[i]
        dr = float(dropout_rates[i]) if dropout_rates and i < len(dropout_rates) else 0.0

        x = layers.TimeDistributed(
            layers.Conv2D(filters=f, kernel_size=tuple(k), activation=activation, padding="same"),
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=tuple(p)))(x)
        if dr > 0:
            x = layers.TimeDistributed(layers.Dropout(dr))(x)

    # Flatten spatial dimensions within each temporal slice to obtain a
    # sequence of frame-level embeddings of shape (T, D).
    x = layers.TimeDistributed(layers.Flatten())(x)

    lstm_units = int(lstm_cfg.get("units"))
    lstm_dropout = float(lstm_cfg.get("dropout", 0.0))
    lstm_recurrent_dropout = float(lstm_cfg.get("recurrent_dropout", 0.0))

    x = layers.LSTM(
        lstm_units,
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout,
        return_sequences=False,
    )(x)

    dense_layers = dense_cfg.get("layers", []) or []
    dense_dropout_rates = dense_cfg.get("dropout_rates", []) or []

    for i, units in enumerate(dense_layers):
        x = layers.Dense(int(units), activation="relu")(x)
        dr = float(dense_dropout_rates[i]) if i < len(dense_dropout_rates) else 0.0
        if dr > 0:
            x = layers.Dropout(dr)(x)

    output_type = str(output_cfg.get("type"))
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in this model builder")

    num_classes = int(output_cfg.get("num_classes"))
    output_activation = output_cfg["activation"]

    up_head = layers.Dense(num_classes, activation=output_activation, name="up_intensity")(x)
    down_head = layers.Dense(num_classes, activation=output_activation, name="down_intensity")(x)

    model = keras.Model(inputs=inputs, outputs=[up_head, down_head], name="cnn_lstm_two_head_intensity")

    compilation_cfg = model_cfg.get("compilation", {})
    optimizer_name = compilation_cfg.get("optimizer")
    learning_rate = float(compilation_cfg.get("learning_rate"))
    loss = compilation_cfg.get("loss")
    metrics_cfg = compilation_cfg.get("metrics")

    optimizer = keras.optimizers.get({"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})

    def _build_metrics_for_head(metric_specs):
        if isinstance(metric_specs, (list, tuple)):
            metrics_list = list(metric_specs)
        else:
            metrics_list = [metric_specs]

        metric_objects = []
        for m in metrics_list:
            if isinstance(m, str):
                name_lower = m.lower()
                if name_lower in {"accuracy", "acc", "categorical_accuracy"}:
                    metric_objects.append(keras.metrics.CategoricalAccuracy(name=m))
                elif name_lower == "precision":
                    metric_objects.append(keras.metrics.Precision(name=m))
                elif name_lower == "recall":
                    metric_objects.append(keras.metrics.Recall(name=m))
                else:
                    metric_objects.append(keras.metrics.get(m))
            else:
                metric_objects.append(keras.metrics.get(m))

        return metric_objects

    metrics = None
    if isinstance(metrics_cfg, dict):
        metrics = metrics_cfg
    elif metrics_cfg is None:
        metrics = None
    else:
        metrics = {
            "up_intensity": _build_metrics_for_head(metrics_cfg),
            "down_intensity": _build_metrics_for_head(metrics_cfg),
        }

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(
        "CNN+LSTM model built and compiled: name=%s, num_classes=%d", model.name, num_classes
    )

    return model


__all__ = ["build_cnn_lstm_model"]
