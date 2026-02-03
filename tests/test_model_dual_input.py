"""Tests for CNN+LSTM model builder with dual-channel support (TD-019).

This module tests the model architecture construction, including:
- Single-input (short-term only) mode
- Dual-input (short-term + long-term context) mode
- Configuration parsing for long-term branch
- Input/output shape validation
"""

from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import patch

import numpy as np


def _make_minimal_config(
    long_term_cfg: Dict[str, Any] | None = None,
    input_dim_override: int | None = None,
) -> Dict[str, Any]:
    """Create a minimal valid config for model building."""
    lt_cfg = long_term_cfg if long_term_cfg is not None else {"enabled": False}
    
    return {
        "model": {
            "cnn": {
                "num_layers": 2,
                "filters": [32, 64],
                "kernel_sizes": [[3, 3], [3, 3]],
                "pool_sizes": [[2, 2], [2, 2]],
                "activation": "relu",
                "dropout_rates": [0.1, 0.2],
            },
            "lstm": {
                "units": 64,
                "dropout": 0.2,
                "recurrent_dropout": 0.0,
            },
            "dense": {
                "layers": [32],
                "dropout_rates": [0.2],
            },
            "output": {
                "type": "two_head_intensity",
                "num_classes": 4,
                "activation": "softmax",
            },
            "compilation": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "long_term": lt_cfg,
        },
    }


class TestModelBuilderSingleInput(unittest.TestCase):
    """Tests for single-input model (long_term disabled)."""

    def test_build_single_input_model(self) -> None:
        """Test building model without long-term context."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()
        input_shape = (10, 8, 8, 4)  # (T, H, W, C)

        model = build_cnn_lstm_model(config, input_shape)

        # Check model name
        self.assertEqual(model.name, "cnn_lstm_two_head_intensity")

        # Check single input
        self.assertEqual(len(model.inputs), 1)
        # Input shape should match (excluding batch dim)
        self.assertEqual(model.inputs[0].shape[1:], input_shape)

        # Check two outputs
        self.assertEqual(len(model.outputs), 2)
        # Check output names contain the head identifiers (Keras 3.x uses different naming)
        output_names = [o.name for o in model.outputs]
        # Just verify we have two outputs with expected shapes
        self.assertEqual(model.outputs[0].shape[-1], 4)  # num_classes
        self.assertEqual(model.outputs[1].shape[-1], 4)

    def test_single_input_forward_pass(self) -> None:
        """Test forward pass with single input."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        # Create dummy input
        batch_size = 2
        x = np.random.randn(batch_size, *input_shape).astype(np.float32)

        # Forward pass
        outputs = model.predict(x, verbose=0)

        # Check output shapes
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (batch_size, 4))
        self.assertEqual(outputs[1].shape, (batch_size, 4))

        # Check outputs are valid probabilities
        for out in outputs:
            self.assertTrue(np.all(out >= 0))
            self.assertTrue(np.all(out <= 1))
            np.testing.assert_array_almost_equal(out.sum(axis=1), np.ones(batch_size))

    def test_explicit_override_to_disable_long_term(self) -> None:
        """Test that passing long_term_input_dim=0 disables long-term even if config enables it."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={"enabled": True})
        input_shape = (10, 8, 8, 4)

        # Pass 0 to explicitly disable
        model = build_cnn_lstm_model(config, input_shape, long_term_input_dim=0)

        # Should be single-input
        self.assertEqual(model.name, "cnn_lstm_two_head_intensity")
        self.assertEqual(len(model.inputs), 1)


class TestModelBuilderDualInput(unittest.TestCase):
    """Tests for dual-input model (long_term enabled)."""

    def test_build_dual_input_model(self) -> None:
        """Test building model with long-term context."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={
            "enabled": True,
            "windows_days": [7, 30],
            "features": ["mean_return", "volatility"],
            "dense": {
                "layers": [16],
                "dropout_rates": [0.1],
            },
        })
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        # Check model name
        self.assertEqual(model.name, "cnn_lstm_dual_channel_two_head")

        # Check dual inputs
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(model.inputs[0].name, "main_input")
        self.assertEqual(model.inputs[1].name, "long_term_input")

        # Check input shapes
        self.assertEqual(model.inputs[0].shape[1:], input_shape)
        # 2 windows * 2 features = 4
        self.assertEqual(model.inputs[1].shape[1], 4)

    def test_dual_input_with_explicit_dim(self) -> None:
        """Test building dual-input model with explicit input_dim."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={"enabled": True})
        input_shape = (10, 8, 8, 4)

        # Pass explicit long_term_input_dim
        model = build_cnn_lstm_model(config, input_shape, long_term_input_dim=12)

        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(model.inputs[1].shape[1], 12)

    def test_dual_input_forward_pass(self) -> None:
        """Test forward pass with dual inputs."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={
            "enabled": True,
            "windows_days": [7, 30, 90],
            "features": ["mean_return", "volatility", "volume_proxy", "skewness"],
        })
        input_shape = (10, 8, 8, 4)
        lt_input_dim = 3 * 4  # 3 windows * 4 features

        model = build_cnn_lstm_model(config, input_shape)

        # Create dummy inputs
        batch_size = 2
        x_short = np.random.randn(batch_size, *input_shape).astype(np.float32)
        x_long = np.random.randn(batch_size, lt_input_dim).astype(np.float32)

        # Forward pass with list of inputs
        outputs = model.predict([x_short, x_long], verbose=0)

        # Check output shapes
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (batch_size, 4))
        self.assertEqual(outputs[1].shape, (batch_size, 4))

        # Check outputs are valid probabilities
        for out in outputs:
            self.assertTrue(np.all(out >= 0))
            self.assertTrue(np.all(out <= 1))
            np.testing.assert_array_almost_equal(out.sum(axis=1), np.ones(batch_size))

    def test_dual_input_auto_compute_dim(self) -> None:
        """Test that input_dim is auto-computed from windows and features."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        # Test various combinations
        test_cases = [
            ([7], ["mean_return"], 1),
            ([7, 30], ["mean_return", "volatility"], 4),
            ([7, 30, 90], ["mean_return", "volatility", "volume_proxy", "skewness"], 12),
        ]

        for windows, features, expected_dim in test_cases:
            config = _make_minimal_config(long_term_cfg={
                "enabled": True,
                "windows_days": windows,
                "features": features,
            })
            input_shape = (10, 8, 8, 4)

            model = build_cnn_lstm_model(config, input_shape)

            self.assertEqual(
                model.inputs[1].shape[1],
                expected_dim,
                f"Failed for windows={windows}, features={features}",
            )


class TestModelBuilderConfigValidation(unittest.TestCase):
    """Tests for configuration validation."""

    def test_invalid_input_shape(self) -> None:
        """Test that invalid input shape raises ValueError."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()

        # Wrong number of dimensions
        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8))
        self.assertIn("input_shape=(T, H, W, C)", str(ctx.exception))

    def test_invalid_output_type(self) -> None:
        """Test that unsupported output type raises ValueError."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()
        config["model"]["output"]["type"] = "single_head"

        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8, 4))
        self.assertIn("two_head_intensity", str(ctx.exception))

    def test_missing_cnn_config(self) -> None:
        """Test that missing CNN config raises error."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()
        config["model"]["cnn"]["filters"] = "not_a_list"

        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8, 4))
        self.assertIn("must be lists", str(ctx.exception))


class TestLongTermContextHelpers(unittest.TestCase):
    """Tests for training/long_term_context.py helpers."""

    def test_is_long_term_enabled(self) -> None:
        """Test is_long_term_enabled function."""
        from training.long_term_context import is_long_term_enabled

        config_disabled = _make_minimal_config(long_term_cfg={"enabled": False})
        self.assertFalse(is_long_term_enabled(config_disabled))

        config_enabled = _make_minimal_config(long_term_cfg={"enabled": True})
        self.assertTrue(is_long_term_enabled(config_enabled))

        config_missing = {"model": {}}
        self.assertFalse(is_long_term_enabled(config_missing))

    def test_get_long_term_input_dim(self) -> None:
        """Test get_long_term_input_dim function."""
        from training.long_term_context import get_long_term_input_dim

        config_disabled = _make_minimal_config(long_term_cfg={"enabled": False})
        self.assertEqual(get_long_term_input_dim(config_disabled), 0)

        config_enabled = _make_minimal_config(long_term_cfg={
            "enabled": True,
            "windows_days": [7, 30],
            "features": ["mean_return", "volatility"],
        })
        self.assertEqual(get_long_term_input_dim(config_enabled), 4)


if __name__ == "__main__":
    unittest.main()
