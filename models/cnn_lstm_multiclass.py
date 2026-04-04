"""Canonical CNN+LSTM model architecture with dual-channel support.

This module defines a builder for the multi-class CNN+LSTM model described in the
technical specifications. Supports both single-input (short-term only) and
dual-input (short-term + long-term context) architectures.

The dual-channel architecture (TD-019) adds a secondary input branch for
long-term market context features (7/30/90-day summary statistics), which
are merged with the short-term CNN+LSTM features before the output heads.

Architecture:
    Short-Term Input (T, H, W, C) → CNN stack → LSTM stack → (lstm_units,)
                                                                   ↓
    [Optional Long-Term Input (n_features,)] → Conv1D+Dense → Concatenate → Dense → Two-Head Output

Configuration format (list-of-dicts):
    model.cnn.layers:  [{filters, kernel_size, pool_size, normalization, dropout}, ...]
    model.lstm.layers: [{units, dropout, recurrent_dropout, post_dropout}, ...]
    model.dense.layers: [{units, dropout}, ...]
    model.long_term.architecture.conv1d.layers: [{filters, kernel_size, pool_size, normalization, dropout}, ...]
    model.long_term.architecture.dense.layers:  [{units, dropout}, ...]
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging


logger = logging.getLogger(__name__)


def build_metrics_for_head(metric_specs: Any, keras: Any) -> List[Any]:
    if isinstance(metric_specs, (list, tuple)):
        metrics_list = list(metric_specs)
    else:
        metrics_list = [metric_specs]

    metric_objects = []
    for metric in metrics_list:
        if isinstance(metric, str):
            name_lower = metric.lower()
            if name_lower in {"accuracy", "acc", "categorical_accuracy"}:
                metric_objects.append(keras.metrics.CategoricalAccuracy(name=metric))
            elif name_lower == "precision":
                metric_objects.append(keras.metrics.Precision(name=metric))
            elif name_lower == "recall":
                metric_objects.append(keras.metrics.Recall(name=metric))
            else:
                metric_objects.append(keras.metrics.get(metric))
        else:
            metric_objects.append(keras.metrics.get(metric))

    return metric_objects


def _add_normalization_layer(x: Any, norm_type: Optional[str], filters: int, layers: Any,
                             *, time_distributed: bool = False, name_prefix: str = "") -> Any:
    """Add a normalization layer if requested.

    Parameters
    ----------
    x:
        Input tensor.
    norm_type:
        One of ``None``, ``"batch"``, ``"group"``, ``"layer"``.
    filters:
        Number of filters/channels (used for GroupNormalization groups).
    layers:
        ``tensorflow.keras.layers`` module.
    time_distributed:
        If True, wrap the normalization layer in ``TimeDistributed``.
    name_prefix:
        Optional name prefix for the layer.

    Returns
    -------
    Tensor with normalization applied, or unchanged if *norm_type* is None.
    """
    if norm_type is None:
        return x

    norm_type_lower = str(norm_type).lower()
    name = f"{name_prefix}_norm" if name_prefix else None

    if norm_type_lower == "batch":
        norm_layer = layers.BatchNormalization(name=name)
    elif norm_type_lower == "group":
        num_groups = min(32, filters)
        norm_layer = layers.GroupNormalization(groups=num_groups, name=name)
    elif norm_type_lower == "layer":
        norm_layer = layers.LayerNormalization(name=name)
    else:
        raise ValueError(
            f"Unsupported normalization type: {norm_type!r}. "
            "Must be null, 'batch', 'group', or 'layer'."
        )

    if time_distributed:
        return layers.TimeDistributed(norm_layer, name=name)(x)
    return norm_layer(x)


def _build_cnn_stack(x: Any, cnn_cfg: Dict[str, Any], layers: Any) -> Any:
    """Build the TimeDistributed CNN stack from ``model.cnn`` config.

    Parameters
    ----------
    x:
        Input tensor of shape ``(batch, T, H, W, C)``.
    cnn_cfg:
        ``model.cnn`` configuration dict with ``activation`` and ``layers`` list.
    layers:
        ``tensorflow.keras.layers`` module.

    Returns
    -------
    Tensor after CNN stack and ``TimeDistributed(Flatten())``.
    """
    cnn_layers = cnn_cfg["layers"]
    if not isinstance(cnn_layers, list):
        raise ValueError("model.cnn.layers must be a list of layer dicts")

    activation = str(cnn_cfg.get("activation", "relu"))

    for i, layer_cfg in enumerate(cnn_layers):
        f = int(layer_cfg["filters"])
        k = layer_cfg["kernel_size"]
        norm = layer_cfg.get("normalization")
        pool = layer_cfg.get("pool_size")
        dr = float(layer_cfg.get("dropout", 0.0))

        x = layers.TimeDistributed(
            layers.Conv2D(filters=f, kernel_size=tuple(k), activation=activation, padding="same"),
            name=f"cnn_{i}_conv2d",
        )(x)

        x = _add_normalization_layer(
            x, norm, f, layers, time_distributed=True, name_prefix=f"cnn_{i}",
        )

        if pool is not None:
            x = layers.TimeDistributed(
                layers.MaxPooling2D(pool_size=tuple(pool)),
                name=f"cnn_{i}_pool",
            )(x)

        if dr > 0:
            x = layers.TimeDistributed(layers.Dropout(dr), name=f"cnn_{i}_dropout")(x)

    # Flatten spatial dimensions within each temporal slice → (T, D)
    x = layers.TimeDistributed(layers.Flatten(), name="cnn_flatten")(x)
    return x


def _build_lstm_stack(x: Any, lstm_cfg: Dict[str, Any], layers: Any) -> Any:
    """Build the LSTM stack from ``model.lstm`` config.

    Parameters
    ----------
    x:
        Input tensor of shape ``(batch, T, D)`` from CNN flatten.
    lstm_cfg:
        ``model.lstm`` configuration dict with ``layers`` list.
    layers:
        ``tensorflow.keras.layers`` module.

    Returns
    -------
    Tensor after LSTM stack. Shape ``(batch, units)`` from the final LSTM.
    """
    lstm_layers = lstm_cfg["layers"]
    if not isinstance(lstm_layers, list) or len(lstm_layers) == 0:
        raise ValueError("model.lstm.layers must be a non-empty list of layer dicts")

    num_lstm = len(lstm_layers)

    for i, layer_cfg in enumerate(lstm_layers):
        units = int(layer_cfg["units"])
        dropout = float(layer_cfg.get("dropout", 0.0))
        recurrent_dropout = float(layer_cfg.get("recurrent_dropout", 0.0))
        post_dropout = float(layer_cfg.get("post_dropout", 0.0))
        is_last = (i == num_lstm - 1)

        x = layers.LSTM(
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=not is_last,
            name=f"lstm_{i}",
        )(x)

        if post_dropout > 0:
            x = layers.Dropout(post_dropout, name=f"lstm_{i}_post_dropout")(x)

    return x


def _build_dense_stack(x: Any, dense_cfg: Dict[str, Any], layers: Any,
                       *, name_prefix: str = "dense") -> Any:
    """Build a dense stack from a ``layers`` list-of-dicts config.

    Parameters
    ----------
    x:
        Input tensor.
    dense_cfg:
        Dict with ``layers`` key containing list of ``{units, dropout}`` dicts.
    layers:
        ``tensorflow.keras.layers`` module.
    name_prefix:
        Prefix for layer names.

    Returns
    -------
    Tensor after dense stack.
    """
    dense_layers = dense_cfg["layers"]
    if not isinstance(dense_layers, list):
        raise ValueError(f"{name_prefix}.layers must be a list of layer dicts")

    for i, layer_cfg in enumerate(dense_layers):
        units = int(layer_cfg["units"])
        dr = float(layer_cfg.get("dropout", 0.0))

        x = layers.Dense(units, activation="relu", name=f"{name_prefix}_{i}")(x)
        if dr > 0:
            x = layers.Dropout(dr, name=f"{name_prefix}_{i}_dropout")(x)

    return x


def _build_long_term_branch(long_term_input: Any, long_term_cfg: Dict[str, Any],
                            long_term_input_dim: int, layers: Any) -> Any:
    """Build the long-term context branch.

    Supports optional Conv1D layers followed by Dense layers.

    Parameters
    ----------
    long_term_input:
        Keras Input tensor of shape ``(batch, lt_dim)``.
    long_term_cfg:
        ``model.long_term`` configuration dict.
    long_term_input_dim:
        Total dimension of the long-term feature vector.
    layers:
        ``tensorflow.keras.layers`` module.

    Returns
    -------
    Tensor output of the long-term branch.
    """
    arch_cfg = long_term_cfg.get("architecture", {})
    conv1d_cfg = arch_cfg.get("conv1d", {})
    dense_cfg = arch_cfg.get("dense", {})

    # If no architecture key, fall back to legacy dense-only format
    if not arch_cfg and "dense" in long_term_cfg:
        dense_cfg = long_term_cfg["dense"]

    y = long_term_input

    # Optional Conv1D layers
    conv1d_layers = conv1d_cfg.get("layers", [])
    if conv1d_layers:
        windows = long_term_cfg["windows_days"]
        features = long_term_cfg["features"]
        num_windows = len(windows)
        num_features = len(features)

        # Reshape (batch, lt_dim) → (batch, num_windows, num_features)
        y = layers.Reshape((num_windows, num_features), name="lt_reshape")(y)

        conv1d_activation = str(conv1d_cfg.get("activation", "relu"))

        for i, layer_cfg in enumerate(conv1d_layers):
            f = int(layer_cfg["filters"])
            k = int(layer_cfg.get("kernel_size", 3))
            norm = layer_cfg.get("normalization")
            pool = layer_cfg.get("pool_size")
            dr = float(layer_cfg.get("dropout", 0.0))

            y = layers.Conv1D(
                filters=f, kernel_size=k, activation=conv1d_activation,
                padding="same", name=f"lt_conv1d_{i}",
            )(y)

            y = _add_normalization_layer(
                y, norm, f, layers, time_distributed=False, name_prefix=f"lt_conv1d_{i}",
            )

            if pool is not None:
                y = layers.MaxPooling1D(pool_size=int(pool), name=f"lt_conv1d_{i}_pool")(y)

            if dr > 0:
                y = layers.Dropout(dr, name=f"lt_conv1d_{i}_dropout")(y)

        y = layers.Flatten(name="lt_conv1d_flatten")(y)

    # Dense layers
    lt_dense_layers = dense_cfg.get("layers", [])
    if lt_dense_layers:
        for i, layer_cfg in enumerate(lt_dense_layers):
            # Support both dict format {units, dropout} and legacy int format
            if isinstance(layer_cfg, dict):
                units = int(layer_cfg["units"])
                dr = float(layer_cfg.get("dropout", 0.0))
            else:
                units = int(layer_cfg)
                lt_dropout_rates = dense_cfg.get("dropout_rates", [])
                dr = float(lt_dropout_rates[i]) if i < len(lt_dropout_rates) else 0.0

            y = layers.Dense(units, activation="relu", name=f"lt_dense_{i}")(y)
            if dr > 0:
                y = layers.Dropout(dr, name=f"lt_dropout_{i}")(y)

    return y


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

    # --- Short-term branch: CNN → LSTM ---
    inputs = keras.Input(shape=input_shape, name="main_input")
    x = _build_cnn_stack(inputs, cnn_cfg, layers)
    x = _build_lstm_stack(x, lstm_cfg, layers)

    short_term_output = x

    # --- Long-term branch (optional) ---
    long_term_input = None
    if long_term_enabled:
        long_term_input = keras.Input(
            shape=(long_term_input_dim,), name="long_term_input"
        )

        y = _build_long_term_branch(
            long_term_input, long_term_cfg, long_term_input_dim, layers,
        )

        x = layers.Concatenate(name="merge_branches")([short_term_output, y])

        logger.info(
            "Long-term branch added: input_dim=%d, architecture=%s",
            long_term_input_dim,
            list(long_term_cfg.get("architecture", {}).keys()),
        )

    # --- Shared dense head ---
    x = _build_dense_stack(x, dense_cfg, layers, name_prefix="dense")

    # --- Output heads ---
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in this model builder")

    num_classes = int(output_cfg["num_classes"])
    output_activation = output_cfg["activation"]

    up_head = layers.Dense(num_classes, activation=output_activation, name="up_intensity", dtype="float32")(x)
    down_head = layers.Dense(num_classes, activation=output_activation, name="down_intensity", dtype="float32")(x)

    # --- Assemble model ---
    if long_term_enabled and long_term_input is not None:
        all_inputs = [inputs, long_term_input]
        model_name = "cnn_lstm_dual_channel_two_head"
    else:
        all_inputs = inputs
        model_name = "cnn_lstm_two_head_intensity"

    model = keras.Model(inputs=all_inputs, outputs=[up_head, down_head], name=model_name)

    # --- Compilation ---
    compilation_cfg = model_cfg["compilation"]
    optimizer_name = compilation_cfg["optimizer"]
    learning_rate = float(compilation_cfg["learning_rate"])
    loss = compilation_cfg["loss"]
    metrics_cfg = compilation_cfg["metrics"]

    optimizer = keras.optimizers.get({"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})

    metrics = None
    if isinstance(metrics_cfg, dict):
        metrics = metrics_cfg
    elif metrics_cfg is None:
        metrics = None
    else:
        metrics = {
            "up_intensity": build_metrics_for_head(metrics_cfg, keras),
            "down_intensity": build_metrics_for_head(metrics_cfg, keras),
        }

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    cnn_count = len(cnn_cfg["layers"])
    lstm_count = len(lstm_cfg["layers"])
    logger.info(
        "CNN+LSTM model built and compiled: name=%s, num_classes=%d, "
        "cnn_layers=%d, lstm_layers=%d, long_term_enabled=%s",
        model.name, num_classes, cnn_count, lstm_count, long_term_enabled,
    )

    return model


__all__ = ["build_cnn_lstm_model", "build_metrics_for_head"]
