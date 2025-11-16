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

    for i in range(num_layers):
        f = int(filters[i])
        k = kernel_sizes[i]
        p = pool_sizes[i]
        dr = float(dropout_rates[i]) if dropout_rates and i < len(dropout_rates) else 0.0

        x = layers.Conv2D(filters=f, kernel_size=tuple(k), activation=activation, padding="same")(x)
        x = layers.MaxPooling2D(pool_size=tuple(p))(x)
        if dr > 0:
            x = layers.Dropout(dr)(x)

    # Flatten spatial dimensions before LSTM; exact reshaping will be refined later
    x = layers.Reshape((-1, x.shape[-1]))(x)

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

    num_classes = int(output_cfg.get("num_classes"))
    output_activation = output_cfg.get("activation", "softmax")

    outputs = layers.Dense(num_classes, activation=output_activation, name="class_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_multiclass")

    compilation_cfg = model_cfg.get("compilation", {})
    optimizer_name = compilation_cfg.get("optimizer")
    learning_rate = float(compilation_cfg.get("learning_rate"))
    loss = compilation_cfg.get("loss")
    metrics = compilation_cfg.get("metrics", []) or []

    optimizer = keras.optimizers.get({"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(
        "CNN+LSTM model built and compiled: name=%s, num_classes=%d", model.name, num_classes
    )

    return model


__all__ = ["build_cnn_lstm_model"]
