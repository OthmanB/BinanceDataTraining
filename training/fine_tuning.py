"""Fine-tuning utilities for loading and adapting pre-trained models.

This module provides:
- Model loading from MLflow (by run_id or from model registry)
- Layer freezing by pattern (none, cnn, cnn_lstm, all_but_output)
- Learning rate adjustment for fine-tuning
- Configuration validation for fine-tuning compatibility

TD-009: Fine-Tuning Support
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import os
import re

import numpy as np


logger = logging.getLogger(__name__)


# Valid freeze patterns for layer freezing
VALID_FREEZE_PATTERNS = frozenset({"none", "cnn", "cnn_lstm", "all_but_output"})

# Default allowed config overrides when fine-tuning
DEFAULT_ALLOWED_OVERRIDES = frozenset({
    "data.time_range.start_date",
    "data.time_range.end_date",
    "training.epochs",
    "training.batch_size",
    "training.fine_tuning",
    "training.debug_max_samples",
    "mlflow.experiment_name",
})


class FineTuningError(Exception):
    """Raised when fine-tuning operations fail."""

    pass


def load_model_from_run(run_id: str, artifact_path: str = "model") -> Any:
    """Load a Keras model from an MLflow run's artifacts.

    Parameters
    ----------
    run_id
        The MLflow run ID containing the model artifact.
    artifact_path
        The artifact path where the model was logged (default: "model").

    Returns
    -------
    keras.Model
        The loaded Keras model.

    Raises
    ------
    FineTuningError
        If the model cannot be loaded.
    """
    if not run_id:
        raise FineTuningError("run_id is required to load a model from MLflow run")

    try:
        import mlflow  # type: ignore[import]
        import mlflow.tensorflow  # type: ignore[import]
    except ImportError as exc:
        raise FineTuningError(
            "MLflow and mlflow.tensorflow are required for fine-tuning but could not be imported"
        ) from exc

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Loading model from MLflow run: run_id=%s, model_uri=%s", run_id, model_uri)

    try:
        model = mlflow.tensorflow.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        raise FineTuningError(
            f"Failed to load model from MLflow run '{run_id}' at path '{artifact_path}': {exc}"
        ) from exc

    logger.info(
        "Model loaded successfully from MLflow run: run_id=%s, model_name=%s",
        run_id,
        getattr(model, "name", "unknown"),
    )
    return model


def load_model_from_registry(
    name: str,
    stage: str = "Production",
    version: Optional[int] = None,
) -> Any:
    """Load a Keras model from the MLflow Model Registry.

    Parameters
    ----------
    name
        The registered model name.
    stage
        The model stage to load (e.g., "Production", "Staging", "None").
        Ignored if version is specified.
    version
        Optional specific version number to load. If provided, stage is ignored.

    Returns
    -------
    keras.Model
        The loaded Keras model.

    Raises
    ------
    FineTuningError
        If the model cannot be loaded.
    """
    if not name:
        raise FineTuningError("Model registry name is required to load from registry")

    try:
        import mlflow  # type: ignore[import]
        import mlflow.tensorflow  # type: ignore[import]
    except ImportError as exc:
        raise FineTuningError(
            "MLflow and mlflow.tensorflow are required for fine-tuning but could not be imported"
        ) from exc

    if version is not None:
        model_uri = f"models:/{name}/{version}"
        logger.info(
            "Loading model from MLflow registry: name=%s, version=%s, model_uri=%s",
            name,
            version,
            model_uri,
        )
    else:
        model_uri = f"models:/{name}/{stage}"
        logger.info(
            "Loading model from MLflow registry: name=%s, stage=%s, model_uri=%s",
            name,
            stage,
            model_uri,
        )

    try:
        model = mlflow.tensorflow.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        raise FineTuningError(
            f"Failed to load model from MLflow registry '{name}': {exc}"
        ) from exc

    logger.info(
        "Model loaded successfully from MLflow registry: name=%s, model_name=%s",
        name,
        getattr(model, "name", "unknown"),
    )
    return model


def _get_layer_type(layer: Any) -> str:
    """Determine the category of a Keras layer for freezing purposes.

    Returns one of: "cnn", "lstm", "dense", "output", "other"
    """
    layer_name = layer.name.lower() if hasattr(layer, "name") else ""
    class_name = layer.__class__.__name__

    # Check for output heads (by name convention)
    if "intensity" in layer_name or "output" in layer_name:
        return "output"

    # Check TimeDistributed wrappers
    if class_name == "TimeDistributed":
        inner_layer = getattr(layer, "layer", None)
        if inner_layer is not None:
            inner_class = inner_layer.__class__.__name__
            if inner_class in ("Conv2D", "MaxPooling2D", "Dropout", "Flatten"):
                return "cnn"

    # Check direct layer types
    if class_name in ("Conv2D", "MaxPooling2D", "Conv1D", "MaxPooling1D"):
        return "cnn"
    if class_name in ("LSTM", "GRU", "SimpleRNN", "Bidirectional"):
        return "lstm"
    if class_name == "Dense":
        # Check if it's an output layer by name
        if "intensity" in layer_name or "output" in layer_name:
            return "output"
        return "dense"

    return "other"


def freeze_layers(model: Any, pattern: str) -> Tuple[int, int]:
    """Freeze layers in a model based on the specified pattern.

    Parameters
    ----------
    model
        The Keras model to modify (in place).
    pattern
        Freezing pattern:
        - "none": All layers remain trainable
        - "cnn": Freeze CNN layers (Conv2D, MaxPooling2D in TimeDistributed)
        - "cnn_lstm": Freeze CNN and LSTM layers
        - "all_but_output": Freeze all layers except output Dense heads

    Returns
    -------
    Tuple[int, int]
        (frozen_count, trainable_count) - number of layers in each state.

    Raises
    ------
    FineTuningError
        If the pattern is not valid.
    """
    if pattern not in VALID_FREEZE_PATTERNS:
        raise FineTuningError(
            f"Invalid freeze_layers pattern: '{pattern}'. "
            f"Valid patterns are: {sorted(VALID_FREEZE_PATTERNS)}"
        )

    if pattern == "none":
        # Ensure all layers are trainable
        for layer in model.layers:
            layer.trainable = True
        trainable_count = len(model.layers)
        logger.info(
            "Layer freezing pattern='none': all %d layers remain trainable",
            trainable_count,
        )
        return 0, trainable_count

    # Determine which layer types to freeze based on pattern
    freeze_types: Set[str] = set()
    if pattern == "cnn":
        freeze_types = {"cnn"}
    elif pattern == "cnn_lstm":
        freeze_types = {"cnn", "lstm"}
    elif pattern == "all_but_output":
        freeze_types = {"cnn", "lstm", "dense", "other"}

    frozen_count = 0
    trainable_count = 0

    for layer in model.layers:
        layer_type = _get_layer_type(layer)

        if layer_type in freeze_types:
            layer.trainable = False
            frozen_count += 1
        else:
            layer.trainable = True
            trainable_count += 1

    logger.info(
        "Layer freezing pattern='%s': frozen=%d, trainable=%d",
        pattern,
        frozen_count,
        trainable_count,
    )

    return frozen_count, trainable_count


def adjust_learning_rate(model: Any, factor: float) -> float:
    """Adjust the learning rate of a compiled model's optimizer.

    Parameters
    ----------
    model
        The Keras model with a compiled optimizer.
    factor
        Multiplicative factor for the learning rate (e.g., 0.1 to reduce by 10x).

    Returns
    -------
    float
        The new learning rate after adjustment.

    Raises
    ------
    FineTuningError
        If the model has no optimizer or learning rate cannot be adjusted.
    """
    if factor <= 0:
        raise FineTuningError(
            f"learning_rate_factor must be positive, got {factor}"
        )

    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise FineTuningError(
            "Model has no optimizer; cannot adjust learning rate. "
            "Ensure the model was compiled before calling adjust_learning_rate."
        )

    # Get current learning rate
    lr_attr = getattr(optimizer, "learning_rate", None)
    if lr_attr is None:
        lr_attr = getattr(optimizer, "lr", None)

    if lr_attr is None:
        raise FineTuningError(
            "Optimizer has no 'learning_rate' or 'lr' attribute; "
            "cannot adjust learning rate."
        )

    # Handle both TensorFlow Variable and plain float
    try:
        # Try numpy() method first (TensorFlow Variable or mock with numpy())
        if hasattr(lr_attr, "numpy"):
            current_lr = float(lr_attr.numpy())
        elif hasattr(lr_attr, "__call__"):
            # Learning rate schedule - get initial value
            current_lr = float(lr_attr(0))
        else:
            current_lr = float(lr_attr)
    except Exception as exc:  # noqa: BLE001
        raise FineTuningError(
            f"Failed to read current learning rate from optimizer: {exc}"
        ) from exc

    new_lr = current_lr * factor

    # Set the new learning rate
    try:
        if hasattr(optimizer, "learning_rate"):
            optimizer.learning_rate.assign(new_lr)
        else:
            optimizer.lr.assign(new_lr)
    except Exception as exc:  # noqa: BLE001
        raise FineTuningError(
            f"Failed to set new learning rate {new_lr}: {exc}"
        ) from exc

    logger.info(
        "Learning rate adjusted: previous=%s, factor=%s, new=%s",
        current_lr,
        factor,
        new_lr,
    )

    return new_lr


def validate_input_shape_compatibility(
    model: Any,
    expected_input_shape: Tuple[int, ...],
) -> None:
    """Validate that a model's input shape matches the expected shape.

    Parameters
    ----------
    model
        The Keras model to validate.
    expected_input_shape
        Expected input shape (excluding batch dimension).

    Raises
    ------
    FineTuningError
        If shapes are incompatible.
    """
    model_input_shape = getattr(model, "input_shape", None)
    if model_input_shape is None:
        raise FineTuningError(
            "Model has no input_shape attribute; cannot validate compatibility"
        )

    # Model input_shape includes batch dimension as None
    # e.g., (None, T, H, W, C) - we compare [1:] with expected_input_shape
    if isinstance(model_input_shape, tuple):
        model_shape_no_batch = model_input_shape[1:]
    else:
        # Handle multi-input models (list of shapes)
        raise FineTuningError(
            "Multi-input models are not supported for fine-tuning shape validation"
        )

    if model_shape_no_batch != expected_input_shape:
        raise FineTuningError(
            f"Model input shape {model_shape_no_batch} does not match "
            f"expected shape {expected_input_shape}. "
            "Ensure the base model was trained with compatible data configuration."
        )

    logger.info(
        "Input shape validation passed: model_shape=%s, expected=%s",
        model_shape_no_batch,
        expected_input_shape,
    )


def validate_output_compatibility(
    model: Any,
    expected_num_classes: int,
    expected_output_type: str = "two_head_intensity",
) -> None:
    """Validate that a model's output configuration matches expectations.

    Parameters
    ----------
    model
        The Keras model to validate.
    expected_num_classes
        Expected number of classes per output head.
    expected_output_type
        Expected output type (currently only "two_head_intensity" is supported).

    Raises
    ------
    FineTuningError
        If output configuration is incompatible.
    """
    if expected_output_type != "two_head_intensity":
        raise FineTuningError(
            f"Only 'two_head_intensity' output type is supported; got '{expected_output_type}'"
        )

    # Check that model has two outputs
    output_shapes = getattr(model, "output_shape", None)
    if output_shapes is None:
        raise FineTuningError("Model has no output_shape attribute")

    if not isinstance(output_shapes, list) or len(output_shapes) != 2:
        raise FineTuningError(
            f"Expected two output heads for 'two_head_intensity', "
            f"but model has {len(output_shapes) if isinstance(output_shapes, list) else 1} outputs"
        )

    # Validate each head has the expected number of classes
    for i, shape in enumerate(output_shapes):
        if len(shape) != 2:
            raise FineTuningError(
                f"Output head {i} has unexpected shape {shape}; expected (None, num_classes)"
            )
        head_num_classes = shape[1]
        if head_num_classes != expected_num_classes:
            raise FineTuningError(
                f"Output head {i} has {head_num_classes} classes; expected {expected_num_classes}"
            )

    logger.info(
        "Output compatibility validated: num_outputs=%d, num_classes=%d",
        len(output_shapes),
        expected_num_classes,
    )


def prepare_fine_tuning(
    config: Dict[str, Any],
    model: Any,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Prepare a loaded model for fine-tuning by applying freezing and LR adjustment.

    This function orchestrates the fine-tuning preparation:
    1. Validates model compatibility with current config
    2. Applies layer freezing based on config
    3. Adjusts learning rate based on config
    4. Recompiles the model if needed

    Parameters
    ----------
    config
        The training configuration dictionary.
    model
        The pre-loaded Keras model.
    input_shape
        Optional expected input shape for validation (excluding batch dim).

    Returns
    -------
    keras.Model
        The prepared model ready for fine-tuning.

    Raises
    ------
    FineTuningError
        If preparation fails.
    """
    training_cfg = config.get("training", {})
    fine_tuning_cfg = training_cfg.get("fine_tuning", {})
    model_cfg = config.get("model", {})
    output_cfg = model_cfg.get("output", {})

    # Extract fine-tuning parameters
    freeze_pattern = str(fine_tuning_cfg.get("freeze_layers", "none"))
    lr_factor = float(fine_tuning_cfg.get("learning_rate_factor", 0.1))

    # Validate input shape if provided
    if input_shape is not None:
        validate_input_shape_compatibility(model, input_shape)

    # Validate output configuration
    num_classes = int(output_cfg.get("num_classes", 4))
    output_type = str(output_cfg.get("type", "two_head_intensity"))
    validate_output_compatibility(model, num_classes, output_type)

    # Apply layer freezing
    frozen_count, trainable_count = freeze_layers(model, freeze_pattern)

    # Log model parameter counts after freezing
    try:
        total_params = int(model.count_params())
        trainable_params = sum(
            int(np.prod(w.shape)) for w in getattr(model, "trainable_weights", [])
        )
        non_trainable_params = sum(
            int(np.prod(w.shape)) for w in getattr(model, "non_trainable_weights", [])
        )

        logger.info(
            "Model parameter counts after freezing: total=%d, trainable=%d, frozen=%d",
            total_params,
            trainable_params,
            non_trainable_params,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to compute parameter counts after freezing: %s", exc)

    # Adjust learning rate
    if lr_factor != 1.0:
        try:
            new_lr = adjust_learning_rate(model, lr_factor)
            logger.info("Fine-tuning learning rate set to: %s", new_lr)
        except FineTuningError as exc:
            logger.warning(
                "Could not adjust learning rate (model may need recompilation): %s",
                exc,
            )

    # Log MLflow metrics if available
    try:
        import mlflow  # type: ignore[import]

        if mlflow.active_run() is not None:
            mlflow.log_param("fine_tuning_enabled", True)
            mlflow.log_param("fine_tuning_freeze_pattern", freeze_pattern)
            mlflow.log_param("fine_tuning_lr_factor", lr_factor)
            mlflow.log_metric("fine_tuning_frozen_layers", frozen_count)
            mlflow.log_metric("fine_tuning_trainable_layers", trainable_count)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to log fine-tuning params to MLflow: %s", exc)

    logger.info(
        "Model prepared for fine-tuning: freeze_pattern=%s, lr_factor=%s",
        freeze_pattern,
        lr_factor,
    )

    return model


def get_fine_tuning_summary(model: Any) -> Dict[str, Any]:
    """Get a summary of model layer states for fine-tuning diagnostics.

    Parameters
    ----------
    model
        The Keras model.

    Returns
    -------
    Dict[str, Any]
        Summary including layer counts by type and trainability.
    """
    summary: Dict[str, Any] = {
        "total_layers": 0,
        "trainable_layers": 0,
        "frozen_layers": 0,
        "layers_by_type": {},
        "trainable_by_type": {},
    }

    for layer in model.layers:
        layer_type = _get_layer_type(layer)
        summary["total_layers"] += 1

        if layer.trainable:
            summary["trainable_layers"] += 1
        else:
            summary["frozen_layers"] += 1

        # Count by type
        summary["layers_by_type"][layer_type] = (
            summary["layers_by_type"].get(layer_type, 0) + 1
        )

        if layer.trainable:
            summary["trainable_by_type"][layer_type] = (
                summary["trainable_by_type"].get(layer_type, 0) + 1
            )

    return summary


__all__ = [
    "FineTuningError",
    "VALID_FREEZE_PATTERNS",
    "adjust_learning_rate",
    "freeze_layers",
    "get_fine_tuning_summary",
    "load_model_from_registry",
    "load_model_from_run",
    "prepare_fine_tuning",
    "validate_input_shape_compatibility",
    "validate_output_compatibility",
]
