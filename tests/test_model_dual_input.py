"""Tests for CNN+LSTM model builder with dual-channel support (TD-019).

This module tests the model architecture construction, including:
- Single-input (short-term only) mode
- Dual-input (short-term + long-term context) mode
- Multi-LSTM stacking
- CNN normalization layers
- Optional pooling per CNN layer
- Long-term Conv1D + Dense architecture
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
    cnn_layers: list | None = None,
    lstm_layers: list | None = None,
    dense_layers: list | None = None,
) -> Dict[str, Any]:
    """Create a minimal valid config for model building (new list-of-dicts format)."""
    lt_defaults: Dict[str, Any] = {
        "enabled": False,
        "windows_days": [7, 30, 90],
        "resolution_days": 1,
        "features": ["mean_return", "volatility", "volume_proxy", "skewness"],
        "summary_method": "mean",
        "ewma_halflife_days": 7.0,
        "input_dim": input_dim_override,
        "architecture": {
            "conv1d": {"activation": "relu", "layers": []},
            "dense": {"layers": [{"units": 32, "dropout": 0.2}]},
        },
    }
    lt_cfg = dict(lt_defaults)
    if long_term_cfg is not None:
        lt_cfg.update(long_term_cfg)
    if input_dim_override is not None:
        lt_cfg["input_dim"] = input_dim_override

    if cnn_layers is None:
        cnn_layers = [
            {"filters": 32, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": None, "dropout": 0.1},
            {"filters": 64, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": None, "dropout": 0.2},
        ]

    if lstm_layers is None:
        lstm_layers = [
            {"units": 64, "dropout": 0.2, "recurrent_dropout": 0.0, "post_dropout": 0.0},
        ]

    if dense_layers is None:
        dense_layers = [
            {"units": 32, "dropout": 0.2},
        ]

    return {
        "model": {
            "cnn": {
                "activation": "relu",
                "layers": cnn_layers,
            },
            "lstm": {
                "layers": lstm_layers,
            },
            "dense": {
                "layers": dense_layers,
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
            "architecture": {
                "conv1d": {"activation": "relu", "layers": []},
                "dense": {"layers": [{"units": 16, "dropout": 0.1}]},
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


class TestModelBuilderMultiLSTM(unittest.TestCase):
    """Tests for stacked (multi-layer) LSTM support."""

    def test_two_lstm_layers(self) -> None:
        """Test model with two stacked LSTM layers."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(lstm_layers=[
            {"units": 64, "dropout": 0.2, "recurrent_dropout": 0.0, "post_dropout": 0.25},
            {"units": 32, "dropout": 0.1, "recurrent_dropout": 0.0, "post_dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        # Verify both LSTM layers exist
        lstm_layer_names = [l.name for l in model.layers if "lstm" in l.name.lower() and "dropout" not in l.name]
        self.assertEqual(len(lstm_layer_names), 2)

    def test_three_lstm_forward_pass(self) -> None:
        """Test forward pass with three stacked LSTMs."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(lstm_layers=[
            {"units": 32, "dropout": 0.0, "recurrent_dropout": 0.0, "post_dropout": 0.1},
            {"units": 48, "dropout": 0.0, "recurrent_dropout": 0.0, "post_dropout": 0.1},
            {"units": 16, "dropout": 0.0, "recurrent_dropout": 0.0, "post_dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        batch_size = 2
        x = np.random.randn(batch_size, *input_shape).astype(np.float32)
        outputs = model.predict(x, verbose=0)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (batch_size, 4))


class TestModelBuilderNormalization(unittest.TestCase):
    """Tests for normalization layers in the CNN stack."""

    def test_batch_normalization(self) -> None:
        """Test CNN layers with batch normalization."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 32, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": "batch", "dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        bn_layers = [l for l in model.layers if "batch_normalization" in l.__class__.__name__.lower()]
        self.assertGreaterEqual(len(bn_layers), 1)

    def test_layer_normalization(self) -> None:
        """Test CNN layers with layer normalization."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 32, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": "layer", "dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        ln_layers = [l for l in model.layers if "layer_normalization" in l.__class__.__name__.lower()]
        self.assertGreaterEqual(len(ln_layers), 1)

    def test_no_normalization(self) -> None:
        """Test CNN layers with normalization=null (None)."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 32, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": None, "dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        norm_layers = [
            l for l in model.layers
            if any(n in l.__class__.__name__.lower() for n in ("normalization",))
        ]
        self.assertEqual(len(norm_layers), 0)


class TestModelBuilderOptionalPooling(unittest.TestCase):
    """Tests for optional pooling (pool_size=null skips pooling)."""

    def test_skip_pooling(self) -> None:
        """Test CNN layer with pool_size=null (no pooling)."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 32, "kernel_size": [3, 3], "pool_size": None, "normalization": None, "dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        pool_layers = [l for l in model.layers if "pooling" in l.__class__.__name__.lower()]
        self.assertEqual(len(pool_layers), 0)

    def test_mixed_pooling(self) -> None:
        """Test mix of layers with and without pooling."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 16, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": None, "dropout": 0.0},
            {"filters": 32, "kernel_size": [3, 3], "pool_size": None, "normalization": None, "dropout": 0.0},
            {"filters": 64, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": None, "dropout": 0.0},
        ])
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        # Should have exactly 2 pooling layers (layers 0 and 2)
        pool_layers = [l for l in model.layers if "pooling" in l.__class__.__name__.lower()]
        self.assertEqual(len(pool_layers), 2)


class TestModelBuilderLongTermConv1D(unittest.TestCase):
    """Tests for long-term branch with Conv1D + Dense architecture."""

    def test_long_term_conv1d_forward_pass(self) -> None:
        """Test forward pass with Conv1D on long-term branch."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={
            "enabled": True,
            "windows_days": [7, 30, 90],
            "features": ["mean_return", "volatility", "volume_proxy", "skewness"],
            "input_dim": None,
            "architecture": {
                "conv1d": {
                    "activation": "relu",
                    "layers": [
                        {"filters": 16, "kernel_size": 2, "pool_size": None, "normalization": None, "dropout": 0.0},
                    ],
                },
                "dense": {
                    "layers": [{"units": 16, "dropout": 0.1}],
                },
            },
        })
        input_shape = (10, 8, 8, 4)
        lt_dim = 3 * 4  # 3 windows * 4 features = 12

        model = build_cnn_lstm_model(config, input_shape)

        batch_size = 2
        x_short = np.random.randn(batch_size, *input_shape).astype(np.float32)
        x_long = np.random.randn(batch_size, lt_dim).astype(np.float32)

        outputs = model.predict([x_short, x_long], verbose=0)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (batch_size, 4))

        # Verify Conv1D layer exists in the model
        conv1d_layers = [l for l in model.layers if "conv1d" in l.__class__.__name__.lower()]
        self.assertGreaterEqual(len(conv1d_layers), 1)

    def test_long_term_dense_only(self) -> None:
        """Test long-term branch with empty Conv1D (dense-only)."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(long_term_cfg={
            "enabled": True,
            "windows_days": [7, 30],
            "features": ["mean_return", "volatility"],
            "architecture": {
                "conv1d": {"activation": "relu", "layers": []},
                "dense": {"layers": [{"units": 16, "dropout": 0.0}]},
            },
        })
        input_shape = (10, 8, 8, 4)

        model = build_cnn_lstm_model(config, input_shape)

        # No Conv1D layers should exist
        conv1d_layers = [l for l in model.layers if "conv1d" in l.__class__.__name__.lower()]
        self.assertEqual(len(conv1d_layers), 0)


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

    def test_invalid_cnn_layers_type(self) -> None:
        """Test that non-list CNN layers raises ValueError."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config()
        config["model"]["cnn"]["layers"] = "not_a_list"

        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8, 4))
        self.assertIn("must be a list", str(ctx.exception))

    def test_empty_lstm_layers_raises(self) -> None:
        """Test that empty LSTM layers list raises ValueError."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(lstm_layers=[])

        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8, 4))
        self.assertIn("non-empty list", str(ctx.exception))

    def test_invalid_normalization_type(self) -> None:
        """Test that invalid normalization type raises ValueError."""
        try:
            from models.cnn_lstm_multiclass import build_cnn_lstm_model
        except ImportError:
            self.skipTest("TensorFlow not available")

        config = _make_minimal_config(cnn_layers=[
            {"filters": 32, "kernel_size": [3, 3], "pool_size": [2, 2], "normalization": "invalid", "dropout": 0.0},
        ])

        with self.assertRaises(ValueError) as ctx:
            build_cnn_lstm_model(config, input_shape=(10, 8, 8, 4))
        self.assertIn("Unsupported normalization", str(ctx.exception))


class TestLongTermContextHelpers(unittest.TestCase):
    """Tests for training/long_term_context.py helpers."""

    def test_is_long_term_enabled(self) -> None:
        """Test is_long_term_enabled function."""
        from training.long_term_context import is_long_term_enabled

        config_disabled = _make_minimal_config(long_term_cfg={"enabled": False})
        self.assertFalse(is_long_term_enabled(config_disabled))

        config_enabled = _make_minimal_config(long_term_cfg={"enabled": True})
        self.assertTrue(is_long_term_enabled(config_enabled))

        with self.assertRaises(KeyError):
            is_long_term_enabled({"model": {}})

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
