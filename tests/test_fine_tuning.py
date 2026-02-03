"""Unit tests for fine-tuning utilities.

Tests cover:
- Model loading from MLflow (mocked)
- Layer freezing patterns
- Learning rate adjustment
- Input/output shape validation
- Fine-tuning preparation orchestration
"""

import importlib
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Import the module for reloading
import training.fine_tuning as fine_tuning_module

from training.fine_tuning import (
    FineTuningError,
    VALID_FREEZE_PATTERNS,
    adjust_learning_rate,
    freeze_layers,
    get_fine_tuning_summary,
    load_model_from_registry,
    load_model_from_run,
    prepare_fine_tuning,
    validate_input_shape_compatibility,
    validate_output_compatibility,
    _get_layer_type,
)


class MockLayer:
    """Mock Keras layer for testing."""

    def __init__(
        self,
        name: str,
        class_name: str = "Dense",
        trainable: bool = True,
        inner_layer: Optional["MockLayer"] = None,
    ):
        self.name = name
        self.trainable = trainable
        self._inner_layer = inner_layer
        self._class_name = class_name

    @property
    def __class__(self):
        class FakeClass:
            pass

        FakeClass.__name__ = self._class_name
        return FakeClass

    @property
    def layer(self) -> Optional["MockLayer"]:
        """For TimeDistributed wrapper."""
        return self._inner_layer


class MockVariable:
    """Mock TensorFlow Variable for learning rate."""

    def __init__(self, value: float):
        self._value = value

    def numpy(self) -> float:
        return self._value

    def assign(self, new_value: float) -> None:
        self._value = new_value


class MockOptimizer:
    """Mock Keras optimizer for testing."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = MockVariable(learning_rate)


class MockModel:
    """Mock Keras model for testing."""

    def __init__(
        self,
        layers: Optional[List[MockLayer]] = None,
        input_shape: Tuple = (None, 10, 20, 20, 4),
        output_shape: Optional[List[Tuple]] = None,
        optimizer: Optional[MockOptimizer] = None,
    ):
        self.layers = layers or []
        self.input_shape = input_shape
        self.output_shape = output_shape or [(None, 4), (None, 4)]
        self.optimizer = optimizer
        self.name = "mock_model"
        self.trainable_weights = []
        self.non_trainable_weights = []

    def count_params(self) -> int:
        return 10000


# =============================================================================
# Test _get_layer_type
# =============================================================================


class TestGetLayerType:
    """Tests for _get_layer_type helper."""

    def test_dense_layer(self) -> None:
        layer = MockLayer("dense_1", class_name="Dense")
        assert _get_layer_type(layer) == "dense"

    def test_output_layer_by_name(self) -> None:
        layer = MockLayer("up_intensity", class_name="Dense")
        assert _get_layer_type(layer) == "output"

    def test_output_layer_down_intensity(self) -> None:
        layer = MockLayer("down_intensity", class_name="Dense")
        assert _get_layer_type(layer) == "output"

    def test_lstm_layer(self) -> None:
        layer = MockLayer("lstm_1", class_name="LSTM")
        assert _get_layer_type(layer) == "lstm"

    def test_gru_layer(self) -> None:
        layer = MockLayer("gru_1", class_name="GRU")
        assert _get_layer_type(layer) == "lstm"

    def test_conv2d_layer(self) -> None:
        layer = MockLayer("conv2d_1", class_name="Conv2D")
        assert _get_layer_type(layer) == "cnn"

    def test_time_distributed_conv2d(self) -> None:
        inner = MockLayer("conv2d", class_name="Conv2D")
        layer = MockLayer("time_distributed_1", class_name="TimeDistributed", inner_layer=inner)
        assert _get_layer_type(layer) == "cnn"

    def test_time_distributed_max_pooling(self) -> None:
        inner = MockLayer("max_pooling2d", class_name="MaxPooling2D")
        layer = MockLayer("time_distributed_2", class_name="TimeDistributed", inner_layer=inner)
        assert _get_layer_type(layer) == "cnn"

    def test_time_distributed_flatten(self) -> None:
        inner = MockLayer("flatten", class_name="Flatten")
        layer = MockLayer("time_distributed_3", class_name="TimeDistributed", inner_layer=inner)
        assert _get_layer_type(layer) == "cnn"

    def test_unknown_layer(self) -> None:
        layer = MockLayer("custom_layer", class_name="CustomLayer")
        assert _get_layer_type(layer) == "other"


# =============================================================================
# Test freeze_layers
# =============================================================================


class TestFreezeLayers:
    """Tests for freeze_layers function."""

    def _create_typical_model(self) -> MockModel:
        """Create a mock model with typical CNN+LSTM+Dense+Output layers."""
        layers = [
            MockLayer("input_1", class_name="InputLayer"),
            MockLayer(
                "time_distributed_conv_1",
                class_name="TimeDistributed",
                inner_layer=MockLayer("conv2d", class_name="Conv2D"),
            ),
            MockLayer(
                "time_distributed_pool_1",
                class_name="TimeDistributed",
                inner_layer=MockLayer("max_pooling2d", class_name="MaxPooling2D"),
            ),
            MockLayer(
                "time_distributed_flatten",
                class_name="TimeDistributed",
                inner_layer=MockLayer("flatten", class_name="Flatten"),
            ),
            MockLayer("lstm_1", class_name="LSTM"),
            MockLayer("dense_1", class_name="Dense"),
            MockLayer("up_intensity", class_name="Dense"),
            MockLayer("down_intensity", class_name="Dense"),
        ]
        return MockModel(layers=layers)

    def test_freeze_none(self) -> None:
        model = self._create_typical_model()
        frozen, trainable = freeze_layers(model, "none")

        assert frozen == 0
        assert trainable == 8
        for layer in model.layers:
            assert layer.trainable is True

    def test_freeze_cnn(self) -> None:
        model = self._create_typical_model()
        frozen, trainable = freeze_layers(model, "cnn")

        # CNN layers: time_distributed_conv, time_distributed_pool, time_distributed_flatten = 3
        assert frozen == 3
        assert trainable == 5

        # Verify specific layers
        assert model.layers[1].trainable is False  # conv
        assert model.layers[2].trainable is False  # pool
        assert model.layers[3].trainable is False  # flatten
        assert model.layers[4].trainable is True   # lstm
        assert model.layers[5].trainable is True   # dense
        assert model.layers[6].trainable is True   # up_intensity
        assert model.layers[7].trainable is True   # down_intensity

    def test_freeze_cnn_lstm(self) -> None:
        model = self._create_typical_model()
        frozen, trainable = freeze_layers(model, "cnn_lstm")

        # CNN layers (3) + LSTM (1) = 4
        assert frozen == 4
        assert trainable == 4

        assert model.layers[1].trainable is False  # conv
        assert model.layers[2].trainable is False  # pool
        assert model.layers[3].trainable is False  # flatten
        assert model.layers[4].trainable is False  # lstm
        assert model.layers[5].trainable is True   # dense
        assert model.layers[6].trainable is True   # up_intensity
        assert model.layers[7].trainable is True   # down_intensity

    def test_freeze_all_but_output(self) -> None:
        model = self._create_typical_model()
        frozen, trainable = freeze_layers(model, "all_but_output")

        # All except output heads: 8 - 2 = 6
        assert frozen == 6
        assert trainable == 2

        assert model.layers[1].trainable is False  # conv
        assert model.layers[2].trainable is False  # pool
        assert model.layers[3].trainable is False  # flatten
        assert model.layers[4].trainable is False  # lstm
        assert model.layers[5].trainable is False  # dense
        assert model.layers[6].trainable is True   # up_intensity
        assert model.layers[7].trainable is True   # down_intensity

    def test_invalid_pattern_raises(self) -> None:
        model = self._create_typical_model()

        with pytest.raises(FineTuningError) as exc_info:
            freeze_layers(model, "invalid_pattern")

        assert "Invalid freeze_layers pattern" in str(exc_info.value)
        assert "invalid_pattern" in str(exc_info.value)

    def test_empty_model(self) -> None:
        model = MockModel(layers=[])
        frozen, trainable = freeze_layers(model, "none")

        assert frozen == 0
        assert trainable == 0


# =============================================================================
# Test adjust_learning_rate
# =============================================================================


class TestAdjustLearningRate:
    """Tests for adjust_learning_rate function."""

    def test_adjust_by_factor(self) -> None:
        optimizer = MockOptimizer(learning_rate=0.001)
        model = MockModel(optimizer=optimizer)

        new_lr = adjust_learning_rate(model, factor=0.1)

        assert new_lr == pytest.approx(0.0001)
        assert optimizer.learning_rate._value == pytest.approx(0.0001)

    def test_factor_of_one(self) -> None:
        optimizer = MockOptimizer(learning_rate=0.001)
        model = MockModel(optimizer=optimizer)

        new_lr = adjust_learning_rate(model, factor=1.0)

        assert new_lr == pytest.approx(0.001)

    def test_increase_learning_rate(self) -> None:
        optimizer = MockOptimizer(learning_rate=0.001)
        model = MockModel(optimizer=optimizer)

        new_lr = adjust_learning_rate(model, factor=2.0)

        assert new_lr == pytest.approx(0.002)

    def test_zero_factor_raises(self) -> None:
        optimizer = MockOptimizer(learning_rate=0.001)
        model = MockModel(optimizer=optimizer)

        with pytest.raises(FineTuningError) as exc_info:
            adjust_learning_rate(model, factor=0.0)

        assert "must be positive" in str(exc_info.value)

    def test_negative_factor_raises(self) -> None:
        optimizer = MockOptimizer(learning_rate=0.001)
        model = MockModel(optimizer=optimizer)

        with pytest.raises(FineTuningError) as exc_info:
            adjust_learning_rate(model, factor=-0.1)

        assert "must be positive" in str(exc_info.value)

    def test_no_optimizer_raises(self) -> None:
        model = MockModel(optimizer=None)

        with pytest.raises(FineTuningError) as exc_info:
            adjust_learning_rate(model, factor=0.1)

        assert "no optimizer" in str(exc_info.value)


# =============================================================================
# Test validate_input_shape_compatibility
# =============================================================================


class TestValidateInputShapeCompatibility:
    """Tests for validate_input_shape_compatibility function."""

    def test_matching_shapes(self) -> None:
        model = MockModel(input_shape=(None, 10, 20, 20, 4))

        # Should not raise
        validate_input_shape_compatibility(model, (10, 20, 20, 4))

    def test_mismatched_shapes(self) -> None:
        model = MockModel(input_shape=(None, 10, 20, 20, 4))

        with pytest.raises(FineTuningError) as exc_info:
            validate_input_shape_compatibility(model, (10, 30, 30, 4))

        assert "does not match" in str(exc_info.value)

    def test_mismatched_channels(self) -> None:
        model = MockModel(input_shape=(None, 10, 20, 20, 4))

        with pytest.raises(FineTuningError) as exc_info:
            validate_input_shape_compatibility(model, (10, 20, 20, 8))

        assert "does not match" in str(exc_info.value)


# =============================================================================
# Test validate_output_compatibility
# =============================================================================


class TestValidateOutputCompatibility:
    """Tests for validate_output_compatibility function."""

    def test_valid_two_head_output(self) -> None:
        model = MockModel(output_shape=[(None, 4), (None, 4)])

        # Should not raise
        validate_output_compatibility(model, expected_num_classes=4)

    def test_wrong_num_classes(self) -> None:
        model = MockModel(output_shape=[(None, 4), (None, 4)])

        with pytest.raises(FineTuningError) as exc_info:
            validate_output_compatibility(model, expected_num_classes=3)

        assert "3" in str(exc_info.value)

    def test_single_output_raises(self) -> None:
        model = MockModel(output_shape=(None, 4))

        with pytest.raises(FineTuningError) as exc_info:
            validate_output_compatibility(model, expected_num_classes=4)

        assert "two output heads" in str(exc_info.value)

    def test_unsupported_output_type(self) -> None:
        model = MockModel()

        with pytest.raises(FineTuningError) as exc_info:
            validate_output_compatibility(
                model,
                expected_num_classes=4,
                expected_output_type="single_head",
            )

        assert "single_head" in str(exc_info.value)


# =============================================================================
# Test load_model_from_run
# =============================================================================


class TestLoadModelFromRun:
    """Tests for load_model_from_run function."""

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(FineTuningError) as exc_info:
            load_model_from_run("")

        assert "run_id is required" in str(exc_info.value)

    def test_none_run_id_raises(self) -> None:
        with pytest.raises(FineTuningError) as exc_info:
            load_model_from_run(None)  # type: ignore[arg-type]

        assert "run_id is required" in str(exc_info.value)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_successful_load(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_model = MockModel()
        mock_mlflow_tf.load_model.return_value = mock_model
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        # Need to reimport to pick up the mock
        importlib.reload(fine_tuning_module)
        try:
            result = fine_tuning_module.load_model_from_run("abc123")
            assert result is mock_model
            mock_mlflow_tf.load_model.assert_called_once_with("runs:/abc123/model")
        finally:
            importlib.reload(fine_tuning_module)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_custom_artifact_path(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_model = MockModel()
        mock_mlflow_tf.load_model.return_value = mock_model
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            result = fine_tuning_module.load_model_from_run("abc123", artifact_path="custom_model")
            mock_mlflow_tf.load_model.assert_called_once_with("runs:/abc123/custom_model")
        finally:
            importlib.reload(fine_tuning_module)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_load_failure_raises(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_mlflow_tf.load_model.side_effect = Exception("Model not found")
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            with pytest.raises(fine_tuning_module.FineTuningError) as exc_info:
                fine_tuning_module.load_model_from_run("abc123")

            assert "Failed to load model" in str(exc_info.value)
            assert "abc123" in str(exc_info.value)
        finally:
            importlib.reload(fine_tuning_module)


# =============================================================================
# Test load_model_from_registry
# =============================================================================


class TestLoadModelFromRegistry:
    """Tests for load_model_from_registry function."""

    def test_empty_name_raises(self) -> None:
        # Re-import to get fresh references after any module reloads
        from training.fine_tuning import (
            FineTuningError as FTE,
            load_model_from_registry as load_from_registry,
        )
        with pytest.raises(FTE) as exc_info:
            load_from_registry("")

        assert "Model registry name is required" in str(exc_info.value)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_load_by_stage(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_model = MockModel()
        mock_mlflow_tf.load_model.return_value = mock_model
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            result = fine_tuning_module.load_model_from_registry("my_model", stage="Production")
            assert result is mock_model
            mock_mlflow_tf.load_model.assert_called_once_with("models:/my_model/Production")
        finally:
            importlib.reload(fine_tuning_module)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_load_by_version(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_model = MockModel()
        mock_mlflow_tf.load_model.return_value = mock_model
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            result = fine_tuning_module.load_model_from_registry("my_model", version=5)
            mock_mlflow_tf.load_model.assert_called_once_with("models:/my_model/5")
        finally:
            importlib.reload(fine_tuning_module)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_version_overrides_stage(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_model = MockModel()
        mock_mlflow_tf.load_model.return_value = mock_model
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            result = fine_tuning_module.load_model_from_registry("my_model", stage="Staging", version=3)
            # Version takes precedence over stage
            mock_mlflow_tf.load_model.assert_called_once_with("models:/my_model/3")
        finally:
            importlib.reload(fine_tuning_module)

    @patch.dict("sys.modules", {"mlflow": MagicMock(), "mlflow.tensorflow": MagicMock()})
    def test_load_failure_raises(self) -> None:
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow_tf = sys.modules["mlflow.tensorflow"]
        mock_mlflow_tf.load_model.side_effect = Exception("Registry error")
        mock_mlflow.tensorflow = mock_mlflow_tf  # type: ignore[attr-defined]

        importlib.reload(fine_tuning_module)
        try:
            with pytest.raises(fine_tuning_module.FineTuningError) as exc_info:
                fine_tuning_module.load_model_from_registry("my_model")
            assert "Failed to load model" in str(exc_info.value)
        finally:
            importlib.reload(fine_tuning_module)


# =============================================================================
# Test prepare_fine_tuning
# =============================================================================


class TestPrepareFinetuning:
    """Tests for prepare_fine_tuning orchestration function."""

    def _create_config(
        self,
        freeze_layers: str = "none",
        lr_factor: float = 0.1,
        num_classes: int = 4,
    ) -> Dict[str, Any]:
        return {
            "training": {
                "fine_tuning": {
                    "enabled": True,
                    "freeze_layers": freeze_layers,
                    "learning_rate_factor": lr_factor,
                },
            },
            "model": {
                "output": {
                    "num_classes": num_classes,
                    "type": "two_head_intensity",
                },
            },
        }

    def _create_model_with_layers(self) -> MockModel:
        layers = [
            MockLayer("input_1", class_name="InputLayer"),
            MockLayer(
                "time_distributed_conv_1",
                class_name="TimeDistributed",
                inner_layer=MockLayer("conv2d", class_name="Conv2D"),
            ),
            MockLayer("lstm_1", class_name="LSTM"),
            MockLayer("dense_1", class_name="Dense"),
            MockLayer("up_intensity", class_name="Dense"),
            MockLayer("down_intensity", class_name="Dense"),
        ]
        return MockModel(
            layers=layers,
            input_shape=(None, 10, 20, 20, 4),
            output_shape=[(None, 4), (None, 4)],
            optimizer=MockOptimizer(learning_rate=0.001),
        )

    def test_prepare_with_no_freezing(self) -> None:
        config = self._create_config(freeze_layers="none", lr_factor=0.1)
        model = self._create_model_with_layers()

        result = prepare_fine_tuning(config, model)

        assert result is model
        # All layers should be trainable
        for layer in model.layers:
            assert layer.trainable is True

    def test_prepare_with_cnn_freezing(self) -> None:
        config = self._create_config(freeze_layers="cnn", lr_factor=0.1)
        model = self._create_model_with_layers()

        result = prepare_fine_tuning(config, model)

        # CNN layers should be frozen
        assert model.layers[1].trainable is False  # time_distributed_conv
        assert model.layers[2].trainable is True   # lstm
        assert model.layers[4].trainable is True   # up_intensity

    def test_prepare_adjusts_learning_rate(self) -> None:
        config = self._create_config(freeze_layers="none", lr_factor=0.5)
        model = self._create_model_with_layers()

        prepare_fine_tuning(config, model)

        # LR should be adjusted: 0.001 * 0.5 = 0.0005
        assert model.optimizer.learning_rate._value == pytest.approx(0.0005)

    def test_prepare_validates_input_shape(self) -> None:
        # Re-import to get fresh references after any module reloads
        from training.fine_tuning import (
            FineTuningError as FTE,
            prepare_fine_tuning as prep_ft,
        )
        config = self._create_config()
        model = self._create_model_with_layers()
        model.input_shape = (None, 5, 10, 10, 2)  # Different shape

        with pytest.raises(FTE) as exc_info:
            prep_ft(config, model, input_shape=(10, 20, 20, 4))

        assert "does not match" in str(exc_info.value)

    def test_prepare_validates_output_classes(self) -> None:
        # Re-import to get fresh references after any module reloads
        from training.fine_tuning import (
            FineTuningError as FTE,
            prepare_fine_tuning as prep_ft,
        )
        config = self._create_config(num_classes=3)  # Different num_classes
        model = self._create_model_with_layers()  # Has 4 classes

        with pytest.raises(FTE) as exc_info:
            prep_ft(config, model)

        assert "3" in str(exc_info.value) or "classes" in str(exc_info.value)


# =============================================================================
# Test get_fine_tuning_summary
# =============================================================================


class TestGetFineTuningSummary:
    """Tests for get_fine_tuning_summary function."""

    def test_summary_all_trainable(self) -> None:
        layers = [
            MockLayer("conv", class_name="Conv2D", trainable=True),
            MockLayer("lstm", class_name="LSTM", trainable=True),
            MockLayer("dense", class_name="Dense", trainable=True),
        ]
        model = MockModel(layers=layers)

        summary = get_fine_tuning_summary(model)

        assert summary["total_layers"] == 3
        assert summary["trainable_layers"] == 3
        assert summary["frozen_layers"] == 0
        assert summary["layers_by_type"]["cnn"] == 1
        assert summary["layers_by_type"]["lstm"] == 1
        assert summary["layers_by_type"]["dense"] == 1

    def test_summary_mixed_trainability(self) -> None:
        layers = [
            MockLayer("conv", class_name="Conv2D", trainable=False),
            MockLayer("lstm", class_name="LSTM", trainable=False),
            MockLayer("dense", class_name="Dense", trainable=True),
            MockLayer("up_intensity", class_name="Dense", trainable=True),
        ]
        model = MockModel(layers=layers)

        summary = get_fine_tuning_summary(model)

        assert summary["total_layers"] == 4
        assert summary["trainable_layers"] == 2
        assert summary["frozen_layers"] == 2
        assert summary["trainable_by_type"].get("dense", 0) == 1
        assert summary["trainable_by_type"].get("output", 0) == 1

    def test_summary_empty_model(self) -> None:
        model = MockModel(layers=[])

        summary = get_fine_tuning_summary(model)

        assert summary["total_layers"] == 0
        assert summary["trainable_layers"] == 0
        assert summary["frozen_layers"] == 0


# =============================================================================
# Test VALID_FREEZE_PATTERNS constant
# =============================================================================


class TestValidFreezePatterns:
    """Tests for VALID_FREEZE_PATTERNS constant."""

    def test_contains_expected_patterns(self) -> None:
        assert "none" in VALID_FREEZE_PATTERNS
        assert "cnn" in VALID_FREEZE_PATTERNS
        assert "cnn_lstm" in VALID_FREEZE_PATTERNS
        assert "all_but_output" in VALID_FREEZE_PATTERNS

    def test_is_frozen_set(self) -> None:
        # Ensure it's immutable
        assert isinstance(VALID_FREEZE_PATTERNS, frozenset)


# =============================================================================
# Integration-style tests with real TensorFlow (if available)
# =============================================================================


class TestWithRealTensorFlow:
    """Integration tests using real TensorFlow/Keras models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_tensorflow(self):
        """Skip tests if TensorFlow is not available."""
        pytest.importorskip("tensorflow")

    def test_freeze_real_keras_model(self) -> None:
        """Test freeze_layers with a real Keras model."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Build a simple model similar to CNN+LSTM architecture
        inputs = keras.Input(shape=(10, 20, 20, 4))
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.LSTM(64)(x)
        x = layers.Dense(32, activation="relu")(x)
        up_out = layers.Dense(4, activation="softmax", name="up_intensity")(x)
        down_out = layers.Dense(4, activation="softmax", name="down_intensity")(x)

        model = keras.Model(inputs=inputs, outputs=[up_out, down_out])
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Test freezing CNN layers
        frozen, trainable = freeze_layers(model, "cnn")

        assert frozen > 0
        assert trainable > 0

        # Verify output heads are still trainable
        for layer in model.layers:
            if "intensity" in layer.name:
                assert layer.trainable is True

    def test_adjust_real_optimizer_lr(self) -> None:
        """Test adjust_learning_rate with a real Keras optimizer."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        inputs = keras.Input(shape=(10,))
        outputs = layers.Dense(4, activation="softmax")(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        initial_lr = 0.01
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss="categorical_crossentropy",
        )

        new_lr = adjust_learning_rate(model, factor=0.1)

        assert new_lr == pytest.approx(0.001, rel=1e-5)

    def test_validate_real_model_shapes(self) -> None:
        """Test shape validation with a real Keras model."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Re-import to get fresh references after any module reloads
        from training.fine_tuning import (
            FineTuningError as FTE,
            validate_input_shape_compatibility as validate_input,
            validate_output_compatibility as validate_output,
        )

        inputs = keras.Input(shape=(10, 20, 20, 4))
        x = layers.TimeDistributed(layers.Flatten())(inputs)
        x = layers.LSTM(32)(x)
        up_out = layers.Dense(4, activation="softmax", name="up_intensity")(x)
        down_out = layers.Dense(4, activation="softmax", name="down_intensity")(x)

        model = keras.Model(inputs=inputs, outputs=[up_out, down_out])

        # Should pass
        validate_input(model, (10, 20, 20, 4))
        validate_output(model, expected_num_classes=4)

        # Should fail on wrong shape
        with pytest.raises(FTE):
            validate_input(model, (10, 30, 30, 4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
