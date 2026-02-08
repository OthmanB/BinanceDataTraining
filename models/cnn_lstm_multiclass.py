"""Canonical CNN+LSTM model architecture with dual-channel support.

This module defines a builder for the multi-class CNN+LSTM model described in the
technical specifications. Supports both single-input (short-term only) and
dual-input (short-term + long-term context) architectures.

The dual-channel architecture (TD-019) adds a secondary input branch for
long-term market context features (7/30/90-day summary statistics), which
are merged with the short-term CNN+LSTM features before the output heads.

Architecture:
    Short-Term Input (T, H, W, C) → CNN+LSTM → (lstm_units,)
                                                    ↓
    [Optional Long-Term Input (n_features,)] → Dense → Concatenate → Dense → Two-Head Output
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging


logger = logging.getLogger(__name__)


def build_cnn_lstm_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, ...],
    long_term_input_dim: Optional[int] = None,
):
    """Build a CNN+LSTM Keras model according to the configuration.

    Supports dual-channel architecture when model.long_term.enabled is True.

    Parameters
    ----------
    config:
        Global configuration dictionary loaded from YAML.
    input_shape:
        Shape of the main (short-term) input tensor (excluding batch dimension).
        Expected: (T, H, W, C) where T is time steps, H/W are spatial dims.
    long_term_input_dim:
        Dimension of the long-term input vector. If None and long_term is enabled,
        it will be read from config.model.long_term.input_dim. Pass 0 to force
        single-input mode even when config has long_term enabled.

    Returns
    -------
    keras.Model:
        Compiled Keras model with either:
        - Single input: (batch, T, H, W, C) when long_term disabled
        - Dual inputs: [(batch, T, H, W, C), (batch, lt_dim)] when long_term enabled

    Notes
    -----
    - TensorFlow/Keras is imported lazily inside this function to avoid import
      errors at startup if the dependency is not installed yet.
    - When dual-input mode is enabled, the model expects a list of two inputs
      during training and inference.
    """

    model_cfg = config["model"]

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

    cnn_cfg = model_cfg["cnn"]
    lstm_cfg = model_cfg["lstm"]
    dense_cfg = model_cfg["dense"]
    output_cfg = model_cfg["output"]
    long_term_cfg = model_cfg["long_term"]

    # Determine if long-term branch is enabled
    long_term_enabled = bool(long_term_cfg["enabled"])
    if long_term_input_dim == 0:
        # Explicit override to disable long-term
        long_term_enabled = False
    elif long_term_input_dim is None and long_term_enabled:
        # Read from config
        input_dim_cfg = long_term_cfg["input_dim"]
        long_term_input_dim = int(input_dim_cfg) if input_dim_cfg is not None else 0
        if long_term_input_dim <= 0:
            # Auto-compute from windows and features
            windows = long_term_cfg["windows_days"]
            features = long_term_cfg["features"]
            long_term_input_dim = len(windows) * len(features)
    
    if long_term_enabled and (long_term_input_dim is None or long_term_input_dim <= 0):
        raise ValueError(
            "model.long_term.enabled is True but input_dim could not be determined. "
            "Set model.long_term.input_dim or ensure windows_days and features are configured."
        )

    num_layers = int(cnn_cfg["num_layers"])
    filters = cnn_cfg["filters"]
    kernel_sizes = cnn_cfg["kernel_sizes"]
    pool_sizes = cnn_cfg["pool_sizes"]
    activation = cnn_cfg["activation"]
    dropout_rates = cnn_cfg["dropout_rates"]

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

    lstm_units = int(lstm_cfg["units"])
    lstm_dropout = float(lstm_cfg["dropout"])
    lstm_recurrent_dropout = float(lstm_cfg["recurrent_dropout"])

    x = layers.LSTM(
        lstm_units,
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout,
        return_sequences=False,
    )(x)

    # Store short-term branch output before dense layers
    short_term_output = x

    # Build long-term branch if enabled
    long_term_input = None
    if long_term_enabled:
        long_term_input = keras.Input(
            shape=(long_term_input_dim,), name="long_term_input"
        )

        # Long-term dense layers (configurable, default to single 32-unit layer)
        lt_dense_cfg = long_term_cfg["dense"]
        lt_dense_layers = lt_dense_cfg["layers"]
        lt_dropout_rates = lt_dense_cfg["dropout_rates"]

        y = long_term_input
        for i, units in enumerate(lt_dense_layers):
            y = layers.Dense(int(units), activation="relu", name=f"lt_dense_{i}")(y)
            dr = float(lt_dropout_rates[i]) if i < len(lt_dropout_rates) else 0.0
            if dr > 0:
                y = layers.Dropout(dr, name=f"lt_dropout_{i}")(y)

        # Merge short-term and long-term branches
        x = layers.Concatenate(name="merge_branches")([short_term_output, y])

        logger.info(
            "Long-term branch added: input_dim=%d, dense_layers=%s",
            long_term_input_dim,
            lt_dense_layers,
        )

    dense_layers = dense_cfg["layers"]
    dense_dropout_rates = dense_cfg["dropout_rates"]

    for i, units in enumerate(dense_layers):
        x = layers.Dense(int(units), activation="relu")(x)
        dr = float(dense_dropout_rates[i]) if i < len(dense_dropout_rates) else 0.0
        if dr > 0:
            x = layers.Dropout(dr)(x)

    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in this model builder")

    num_classes = int(output_cfg["num_classes"])
    output_activation = output_cfg["activation"]

    up_head = layers.Dense(num_classes, activation=output_activation, name="up_intensity", dtype="float32")(x)
    down_head = layers.Dense(num_classes, activation=output_activation, name="down_intensity", dtype="float32")(x)

    # Build model with appropriate inputs
    if long_term_enabled and long_term_input is not None:
        all_inputs = [inputs, long_term_input]
        model_name = "cnn_lstm_dual_channel_two_head"
    else:
        all_inputs = inputs
        model_name = "cnn_lstm_two_head_intensity"

    model = keras.Model(inputs=all_inputs, outputs=[up_head, down_head], name=model_name)

    compilation_cfg = model_cfg["compilation"]
    optimizer_name = compilation_cfg["optimizer"]
    learning_rate = float(compilation_cfg["learning_rate"])
    loss = compilation_cfg["loss"]
    metrics_cfg = compilation_cfg["metrics"]

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
        "CNN+LSTM model built and compiled: name=%s, num_classes=%d, long_term_enabled=%s",
        model.name, num_classes, long_term_enabled
    )

    return model


__all__ = ["build_cnn_lstm_model"]
