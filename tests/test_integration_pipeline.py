"""Integration tests for TD-013: End-to-end snapshot pipeline validation.

These tests verify that the snapshot pipeline components work together
correctly across module boundaries. Tests use minimal fixtures to ensure
they run quickly while still validating real data flow.
"""

import os
import tempfile
import unittest
from datetime import datetime
from typing import Any, Dict

import numpy as np


def get_minimal_config() -> Dict[str, Any]:
    """Get a minimal but complete configuration for integration tests."""
    return {
        "run_mode": {"mode": "trial"},
        "data": {
            "source_type": "database",
            "connection": {
                "database_uri": "http://localhost:4000",
                "table_prefix": "orderbook_",
                "request_timeout_seconds": 30,
                "connect_timeout_seconds": 10,
                "max_retries": 3,
                "retry_backoff_factor": 0.5,
            },
            "multi_database": {
                "enabled": False,
                "strategy": "time_split",
                "connections": [],
            },
            "asset_pairs": {
                "target_asset": "BTCUSDT",
                "correlated_assets": [],
                "alignment": {
                    "method": "interpolate",
                    "missing_policy_small": "interpolate",
                    "missing_policy_large": "zero_pad",
                    "large_gap_seconds": 600,
                    "bucket_tolerance_seconds": 2,
                    "include_mask_channel": True,
                },
            },
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "cadence_seconds": 10,
            },
            "ingestion": {
                "chunk_hours": 12,
                "chunk_delay_seconds": 0.5,
                "max_concurrent_chunk_fetches": 1,
            },
            "order_book": {
                "depth_levels": 50,
                "representation": "hybrid",
                "hybrid": {
                    "raw_levels": 10,
                    "aggregated_bins": 10,
                    "bin_strategy": "equal_width",
                },
                "schema": {
                    "timestamp_column": "ts",
                    "bid_price_column": "bid_price",
                    "bid_quantity_column": "bid_quantity",
                    "ask_price_column": "ask_price",
                    "ask_quantity_column": "ask_quantity",
                    "batch_id_column": "batch_id",
                },
            },
            "temporal_features": {
                "local": ["minute_sin", "minute_cos"],
                "global": ["day_of_week"],
                "market_session": {"utc_offset_hours": 9, "sessions": []},
            },
            "validation": {
                "check_missing_data": True,
                "max_gap_seconds": 60,
                "fail_on_invalid": False,
            },
        },
        "targets": {
            "prediction_horizon_seconds": 1800,
            "visible_window_seconds": 60,
            "price_classes": {
                "definition_type": "boundaries",
                "boundaries": [-0.005, -0.001, 0.001, 0.005],
            },
            "labeling": {
                "scheme": "two_head_intensity",
                "use_midpoint": True,
                "handle_gaps": "skip",
            },
        },
        "preprocessing": {
            "normalization": {"method": "zscore", "per_asset": False, "fit_on_train_only": True},
            "feature_engineering": {
                "enabled": True,
                "order_book_features": ["bid_ask_spread", "mid_price"],
                "derived_features": [],
                "momentum_window_seconds": 10,
                "volume_proxy_method": "total_quantity",
                "edge_decay": {"enabled": False, "method": "exponential"},
            },
            "train_test_split": {
                "method": "chronological",
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "class_balancing": {"enabled": False, "method": "class_weights"},
        },
        "model": {
            "framework": "tensorflow",
            "backend": "keras",
            "architecture": "cnn_lstm",
            "input_representation": {
                "strategy": "hybrid",
                "temporal_features": {
                    "integration_mode": "append_channels",
                    "use_local_features": True,
                    "use_global_features": True,
                },
            },
            "cnn": {"num_layers": 2, "filters": [32, 64], "kernel_sizes": [(3, 3), (3, 3)], "pool_sizes": [(2, 2), (2, 2)]},
            "lstm": {"units": 32, "dropout": 0.1, "recurrent_dropout": 0.0},
            "dense": {"layers": [16], "dropout_rates": [0.1]},
            "output": {"type": "two_head_intensity", "num_classes": 4, "activation": "softmax"},
            "compilation": {"optimizer": "adam", "learning_rate": 0.001, "loss": "categorical_crossentropy"},
            "long_term": {
                "enabled": False,
                "windows_days": [7, 30, 90],
                "resolution_days": 1,
                "features": ["mean_return", "volatility", "volume_proxy", "skewness"],
                "summary_method": "mean",
                "ewma_halflife_days": 7.0,
                "input_dim": None,
                "dense": {"layers": [32], "dropout_rates": [0.2]},
            },
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "validation_split": 0.15,
            "debug_max_samples": 100,
            "missing_snapshot_strategy": "skip",
            "callbacks": {
                "early_stopping": {"enabled": False, "monitor": "val_loss", "patience": 5, "restore_best_weights": True},
                "reduce_lr": {"enabled": False, "monitor": "val_loss", "factor": 0.5, "patience": 3, "min_lr": 1e-6},
            },
            "class_weights": {"compute_from_train": False},
            "sample_weighting": {"enabled": False, "method": "recency", "half_life_days": 30, "apply_to": "both"},
            "fine_tuning": {
                "enabled": False,
                "base_model_run_id": None,
                "base_model_stage": "Production",
                "use_model_registry": False,
                "registry_name": None,
                "freeze_layers": "none",
                "learning_rate_factor": 0.1,
            },
            "dataset_cache": {"enabled": False, "directory": "cache", "filename_pattern": "dataset_{hash}.npz", "version": "v1"},
        },
        "snapshot": {
            "enabled": True,
            "directory": "snapshots",
            "root_name": "test_snapshots",
            "name": "test_run",
            "on_config_mismatch": "warn",
            "max_snapshots": 100,
        },
        "hyperparameter_optimization": {
            "enabled": False,
            "framework": "optuna",
            "n_trials": 5,
            "direction": "maximize",
            "metric": "val_accuracy",
            "search_space": {},
            "trial_model_logging": {"enabled": False},
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "integration-tests",
            "local_tmp_dir": "tmp/mlflow",
            "run_naming": {"pattern": "{asset}_{model}_{timestamp}"},
            "artifact_logging": {"trained_model": False, "model_architecture_plot": False, "training_plots": False, "confusion_matrix": False, "class_distribution": False, "feature_importance": False},
            "model_registry": {"register_model": False, "model_name_pattern": "{asset}_predictor"},
        },
        "evaluation": {
            "metrics": ["accuracy"],
            "calibration_analysis": {"enabled": False, "n_bins": 10},
            "post_hoc_calibration": {"enabled": False, "method": "temperature_scaling", "fit_on_validation": True, "temperature_bounds": {"min": 0.1, "max": 10.0}},
            "temporal_degradation": {"enabled": False, "num_windows": 5, "overlap_fraction": 0.25},
            "missing_snapshot_strategy": "skip",
            "backtesting": {"enabled": False, "initial_capital": 10000.0, "transaction_cost": 0.001, "signal_strategy": "net_intensity", "signal_threshold": 0.6, "intensity_threshold": 1, "position_sizing": "equal", "max_position_pct": 1.0},
        },
        "logging": {
            "level": "WARNING",
            "colored_output": True,
            "log_function_names": True,
            "colors": {"function_names": "cyan", "parameter_names": "green", "parameter_values": "yellow", "info": "green", "warning": "yellow", "error": "red", "debug": "blue"},
        },
        "security": {"environment_variables": [], "validation": {"check_env_vars_at_startup": False, "fail_if_missing": False}},
        "diagnostics": {
            "enabled": False,
            "sampling": {"method": "random", "num_samples": 100, "random_seed": 42},
            "spread_checks": {"suspicious_threshold_pct": 0.01, "high_spread_threshold_pct": 0.1, "max_high_spread_fraction": 0.1},
            "quantity_checks": {"enable_negative_checks": True, "enable_zero_stats": True},
            "outlier_checks": {"enabled": False, "z_score_threshold": 3.0, "min_nonzero_points": 10},
            "anomaly_export": {"enabled": False, "max_samples": 100},
            "visualization": {
                "enabled": False,
                "time_series": False,
                "histograms": False,
                "histogram_bins": 50,
                "heatmaps": {"enabled": False, "types": [], "num_time_bins": 50, "num_spread_bins": 50},
                "spread_clipping": {"enabled": False, "num_sigma": 3.0},
                "depth_heatmap": {"enabled": False, "num_price_bins": 50},
            },
            "label_checks": {"enabled": False, "num_examples": 5, "max_examples_per_figure": 5},
            "gap_checks": {"enabled": False, "large_gap_multiplier": 10.0, "very_large_gap_multiplier": 100.0},
        },
    }


class TestModelBuildIntegration(unittest.TestCase):
    """Integration tests for model building from config."""

    def test_build_model_from_config(self) -> None:
        """Build CNN+LSTM model from complete config."""
        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        input_shape = (6, 10, 4, 5)  # (T, H, W, C)

        model = build_cnn_lstm_model(config, input_shape)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 1)  # Single input mode

    def test_build_dual_input_model_from_config(self) -> None:
        """Build dual-input model when long_term is enabled."""
        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        config["model"]["long_term"]["enabled"] = True
        config["model"]["long_term"]["input_dim"] = 12  # 3 windows * 4 features

        input_shape = (6, 10, 4, 5)
        model = build_cnn_lstm_model(config, input_shape)

        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 2)  # Dual input mode

    def test_model_forward_pass_single_input(self) -> None:
        """Test forward pass with single input."""
        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        input_shape = (6, 10, 4, 5)

        model = build_cnn_lstm_model(config, input_shape)
        x = np.random.randn(2, *input_shape).astype(np.float32)
        outputs = model.predict(x, verbose=0)

        # Two-head output: direction (4 classes) and intensity (4 classes)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (2, 4))
        self.assertEqual(outputs[1].shape, (2, 4))

    def test_model_forward_pass_dual_input(self) -> None:
        """Test forward pass with dual inputs."""
        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        config["model"]["long_term"]["enabled"] = True
        long_term_dim = 12

        input_shape = (6, 10, 4, 5)
        model = build_cnn_lstm_model(config, input_shape, long_term_input_dim=long_term_dim)

        x_short = np.random.randn(2, *input_shape).astype(np.float32)
        x_long = np.random.randn(2, long_term_dim).astype(np.float32)
        outputs = model.predict([x_short, x_long], verbose=0)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (2, 4))
        self.assertEqual(outputs[1].shape, (2, 4))


class TestNormalizationIntegration(unittest.TestCase):
    """Integration tests for normalization pipeline."""

    def test_normalizer_creation_from_config(self) -> None:
        """Create normalizer from config."""
        from preprocessing.normalizer import create_normalizer_from_config

        config = get_minimal_config()
        # Use 'standard' which is a valid method
        config["preprocessing"]["normalization"]["method"] = "standard"
        normalizer = create_normalizer_from_config(config)
        self.assertIsNotNone(normalizer)

    def test_normalizer_fit_transform_roundtrip(self) -> None:
        """Normalizer fit and transform on sample data."""
        from preprocessing.normalizer import create_normalizer_from_config

        config = get_minimal_config()
        config["preprocessing"]["normalization"]["method"] = "standard"
        normalizer = create_normalizer_from_config(config)

        # Create sample 4D data (batch, height, width, channels)
        X_train = np.random.randn(100, 10, 4, 5).astype(np.float32) * 10 + 5

        normalizer.fit(X_train)
        X_normalized = normalizer.transform(X_train)

        # Normalized data should have mean near 0, std near 1
        self.assertTrue(np.allclose(X_normalized.mean(), 0, atol=0.1))
        self.assertTrue(np.allclose(X_normalized.std(), 1, atol=0.2))


class TestClassWeightsIntegration(unittest.TestCase):
    """Integration tests for class weights computation."""

    def test_compute_class_weights_from_counts(self) -> None:
        """Compute class weights from label counts."""
        from training.class_weights import compute_class_weights_from_counts

        # API expects Dict[int, int], not numpy arrays
        direction_counts = {0: 100, 1: 50, 2: 30, 3: 20}
        intensity_counts = {0: 80, 1: 60, 2: 40, 3: 20}

        dir_weights = compute_class_weights_from_counts(direction_counts, num_classes=4)
        int_weights = compute_class_weights_from_counts(intensity_counts, num_classes=4)

        self.assertEqual(len(dir_weights), 4)
        self.assertEqual(len(int_weights), 4)

        # Minority classes should have higher weights
        self.assertGreater(dir_weights[3], dir_weights[0])
        self.assertGreater(int_weights[3], int_weights[0])


class TestLongTermFeaturesIntegration(unittest.TestCase):
    """Integration tests for long-term feature computation."""

    def test_long_term_config_from_yaml(self) -> None:
        """Parse LongTermConfig from config dict."""
        from preprocessing.long_term_features import LongTermConfig

        config = get_minimal_config()
        config["model"]["long_term"]["enabled"] = True

        lt_config = LongTermConfig.from_config(config)
        self.assertTrue(lt_config.enabled)
        self.assertEqual(lt_config.windows_days, [7, 30, 90])
        self.assertEqual(len(lt_config.features), 4)
        self.assertEqual(lt_config.input_dim, 12)  # 3 windows * 4 features

    def test_compute_long_term_features_disabled(self) -> None:
        """When disabled, returns zero-dim features."""
        from preprocessing.long_term_features import LongTermConfig, compute_long_term_features

        config = get_minimal_config()
        # Keep long_term.enabled = False (disabled by default)

        # Create sample timestamps and prices
        n_samples = 50
        cadence_seconds = 10
        base_ts = np.datetime64("2024-01-01").astype("datetime64[s]").astype(np.int64)
        timestamps = np.array([base_ts + i * cadence_seconds for i in range(n_samples)], dtype=np.int64)
        mid_prices = np.random.randn(n_samples).astype(np.float32) * 100 + 50000
        anchor_timestamps = timestamps.copy()

        features = compute_long_term_features(
            config=config,
            mid_prices=mid_prices,
            timestamps=timestamps,
            anchor_timestamps=anchor_timestamps,
            cadence_seconds=cadence_seconds,
        )
        self.assertEqual(features.shape, (n_samples, 0))


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests for post-hoc calibration."""

    def test_temperature_calibration_roundtrip(self) -> None:
        """Temperature calibration fit and apply."""
        from evaluation.calibration import TemperatureScaler, apply_temperature_scaling, fit_temperature

        # Create mock softmax probabilities
        np.random.seed(42)
        n_samples = 100
        n_classes = 4

        # Simulate overconfident predictions
        raw_logits = np.random.randn(n_samples, n_classes).astype(np.float32) * 3
        probs = np.exp(raw_logits) / np.exp(raw_logits).sum(axis=1, keepdims=True)
        labels = np.random.randint(0, n_classes, n_samples)

        # Fit temperature
        scaler = fit_temperature(probs, labels)
        self.assertIsNotNone(scaler)
        self.assertGreater(scaler.temperature, 0)

        # Apply temperature scaling
        calibrated = apply_temperature_scaling(probs, scaler.temperature)
        self.assertEqual(calibrated.shape, probs.shape)

        # Probabilities should still sum to 1
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)


class TestBacktestingIntegration(unittest.TestCase):
    """Integration tests for backtesting framework."""

    def test_backtest_config_creation(self) -> None:
        """Create BacktestConfig directly."""
        from evaluation.backtesting import BacktestConfig

        bt_config = BacktestConfig(
            initial_capital=10000.0,
            transaction_cost_pct=0.001,
            signal_strategy="net_intensity",
            signal_threshold=0.6,
            intensity_threshold=1,
            position_sizing="equal",
            max_position_pct=1.0,
        )
        self.assertEqual(bt_config.initial_capital, 10000.0)
        self.assertEqual(bt_config.signal_strategy, "net_intensity")

    def test_backtest_simulation_basic(self) -> None:
        """Run basic backtest simulation."""
        from evaluation.backtesting import BacktestConfig, BacktestResult, run_backtest

        # Create mock full config with backtesting settings
        config = get_minimal_config()
        config["evaluation"]["backtesting"]["enabled"] = True

        # Create mock predictions (up/down intensity probs, not direction/intensity)
        np.random.seed(42)
        n_samples = 100
        n_classes = 4
        y_prob_up = np.random.rand(n_samples, n_classes).astype(np.float32)
        y_prob_up /= y_prob_up.sum(axis=1, keepdims=True)
        y_prob_down = np.random.rand(n_samples, n_classes).astype(np.float32)
        y_prob_down /= y_prob_down.sum(axis=1, keepdims=True)

        # Create mock prices with slight trend
        prices = (50000 + np.cumsum(np.random.randn(n_samples) * 10)).astype(np.float32)
        horizon_steps = 6  # 60 seconds / 10s cadence

        result = run_backtest(
            config=config,
            y_prob_up=y_prob_up,
            y_prob_down=y_prob_down,
            prices=prices,
            horizon_steps=horizon_steps,
        )

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(len(result.equity_curve), n_samples)


class TestFineTuningIntegration(unittest.TestCase):
    """Integration tests for fine-tuning support."""

    def test_freeze_layers(self) -> None:
        """Test layer freezing by pattern."""
        from training.fine_tuning import freeze_layers

        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        input_shape = (6, 10, 4, 5)
        model = build_cnn_lstm_model(config, input_shape)

        # Freeze CNN layers using the 'cnn' pattern
        freeze_layers(model, pattern="cnn")
        cnn_layers = [l for l in model.layers if "conv" in l.name.lower()]
        for layer in cnn_layers:
            self.assertFalse(layer.trainable, f"Layer {layer.name} should be frozen")

    def test_adjust_learning_rate(self) -> None:
        """Test learning rate adjustment for fine-tuning."""
        from training.fine_tuning import adjust_learning_rate

        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        input_shape = (6, 10, 4, 5)
        model = build_cnn_lstm_model(config, input_shape)

        original_lr = float(model.optimizer.learning_rate.numpy())
        adjust_learning_rate(model, factor=0.1)
        new_lr = float(model.optimizer.learning_rate.numpy())

        self.assertAlmostEqual(new_lr, original_lr * 0.1, places=8)


class TestEndToEndPipelineComponents(unittest.TestCase):
    """Integration tests verifying component interactions."""

    def test_config_logging_integration(self) -> None:
        """Test that logging setup works with full config."""
        from utils.colored_logging import setup_colored_logging

        config = get_minimal_config()
        logger = setup_colored_logging(config)
        self.assertIsNotNone(logger)

    def test_model_compilation_matches_config(self) -> None:
        """Verify model compilation uses config values."""
        from models.cnn_lstm_multiclass import build_cnn_lstm_model

        config = get_minimal_config()
        config["model"]["compilation"]["learning_rate"] = 0.005

        input_shape = (6, 10, 4, 5)
        model = build_cnn_lstm_model(config, input_shape)

        lr = float(model.optimizer.learning_rate.numpy())
        self.assertAlmostEqual(lr, 0.005, places=6)

    def test_normalization_stats_serialization(self) -> None:
        """Test normalization stats save/load cycle."""
        from training.snapshot_dataset import NormalizationStats, load_normalization_stats, save_normalization_stats

        stats = NormalizationStats(
            method="standard",
            mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            std=np.array([0.5, 1.0, 1.5], dtype=np.float32),
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            save_normalization_stats(temp_path, stats)
            loaded = load_normalization_stats(temp_path)

            self.assertEqual(loaded.method, stats.method)
            self.assertIsNotNone(loaded.mean)
            self.assertIsNotNone(loaded.std)
            np.testing.assert_array_equal(loaded.mean, stats.mean)  # type: ignore
            np.testing.assert_array_equal(loaded.std, stats.std)  # type: ignore
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
