import unittest
from unittest import mock

import numpy as np

from evaluation.evaluator import evaluate_model


class _DummyEvalModel:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.last_input_shape = None

    def predict(self, x, verbose=0):  # type: ignore[override]
        self.last_input_shape = tuple(x.shape)
        n = x.shape[0]
        up = np.zeros((n, self.num_classes), dtype="float32")
        down = np.zeros((n, self.num_classes), dtype="float32")
        up[:, 0] = 1.0
        down[:, 0] = 1.0
        return [up, down]


class TestEvaluator(unittest.TestCase):
    @mock.patch("evaluation.evaluator.mlflow", create=True)
    def test_evaluate_model_runs_with_synthetic_inputs(self, mock_mlflow) -> None:  # type: ignore[override]
        num_samples = 20
        num_classes = 4

        config = {
            "data": {
                "asset_pairs": {"target_asset": "BTCUSDT", "correlated_assets": []},
                "time_range": {"start_date": "2024-01-01", "end_date": "2024-01-10", "cadence_seconds": 10},
                "order_book": {"depth_levels": 10, "representation": "hybrid", "schema": {}},
                "temporal_features": {"local": [], "global": [], "market_session": {"utc_offset_hours": 0, "sessions": []}},
                "validation": {"check_missing_data": False, "max_gap_seconds": 60, "fail_on_invalid": False},
            },
            "targets": {
                "prediction_horizon_seconds": 1800,
                "visible_window_seconds": 3600,
                "price_classes": {"definition_type": "percentage", "boundaries": [2.0, 5.0, 10.0]},
                "labeling": {"scheme": "two_head_intensity", "use_midpoint": True, "handle_gaps": "interpolate"},
            },
            "preprocessing": {
                "normalization": {"method": "min_max", "per_asset": True, "fit_on_train_only": True},
                "feature_engineering": {
                    "enabled": False,
                    "order_book_features": [],
                    "derived_features": [],
                    "momentum_window_seconds": 300,
                    "volume_proxy_method": "top_of_book",
                    "edge_decay": {"enabled": False, "method": "linear"},
                },
                "train_test_split": {"method": "chronological", "train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15},
                "class_balancing": {"enabled": False, "method": "class_weights"},
            },
            "model": {
                "framework": "keras",
                "backend": "tensorflow",
                "architecture": "CNN_LSTM_MultiClass",
                "input_representation": {
                    "strategy": "stacked_channels",
                    "temporal_features": {"integration_mode": "none", "use_local_features": False, "use_global_features": False},
                },
                "cnn": {"num_layers": 1, "filters": [8], "kernel_sizes": [[1, 1]], "pooling": "max", "pool_sizes": [[1, 1]], "activation": "relu", "dropout_rates": [0.0]},
                "lstm": {"units": 4, "dropout": 0.0, "recurrent_dropout": 0.0},
                "dense": {"layers": [], "dropout_rates": []},
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
                "output": {"type": "two_head_intensity", "num_classes": num_classes, "activation": "softmax"},
                "compilation": {"optimizer": "adam", "learning_rate": 0.001, "loss": "categorical_crossentropy", "metrics": ["accuracy"]},
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "validation_split": 0.15,
                "debug_max_samples": 100,
                "missing_snapshot_strategy": "synthetic",
                "callbacks": {"early_stopping": {"enabled": False, "monitor": "val_loss", "patience": 1, "restore_best_weights": False}, "reduce_lr": {"enabled": False, "monitor": "val_loss", "factor": 0.5, "patience": 1, "min_lr": 1e-5}},
                "class_weights": {"compute_from_train": False},
                "sample_weighting": {"enabled": False, "method": "exponential_decay", "half_life_days": 90, "apply_to": "loss_function"},
                "fine_tuning": {
                    "enabled": False,
                    "base_model_run_id": None,
                    "base_model_stage": "Production",
                    "use_model_registry": False,
                    "registry_name": None,
                    "freeze_layers": "none",
                    "learning_rate_factor": 0.1,
                },
            },
            "evaluation": {
                "metrics": ["accuracy"],
                "calibration_analysis": {"enabled": False, "n_bins": 10},
                "post_hoc_calibration": {"enabled": False, "method": "temperature_scaling", "fit_on_validation": True, "min_samples": 500, "temperature_bounds": {"min": 0.1, "max": 10.0}},
                "temporal_degradation": {"enabled": False, "num_windows": 5, "overlap_fraction": 0.0, "log_per_window_metrics": True},
                "missing_snapshot_strategy": "synthetic",
                "backtesting": {"enabled": False, "horizon_steps": 5, "initial_capital": 10000.0, "transaction_cost": 0.001, "signal_strategy": "net_intensity", "signal_threshold": 0.6, "intensity_threshold": 1, "position_sizing": "equal", "max_position_pct": 1.0},
            },
            "mlflow": {
                "tracking_uri": "http://mlflow",
                "experiment_name": "test",
                "local_tmp_dir": "tmp",
                "run_naming": {"pattern": "test"},
                "artifact_logging": {"trained_model": False, "model_architecture_plot": False, "training_plots": False, "confusion_matrix": False, "class_distribution": False, "feature_importance": False},
                "model_registry": {"register_model": False, "model_name_pattern": "test"},
            },
            "diagnostics": {
                "enabled": False,
                "sampling": {"method": "uniform", "num_samples": 10, "random_seed": 42},
                "spread_checks": {"suspicious_threshold_pct": 1.0, "high_spread_threshold_pct": 5.0, "max_high_spread_fraction": 0.05},
                "quantity_checks": {"enable_negative_checks": False, "enable_zero_stats": False},
                "outlier_checks": {"enabled": False, "z_score_threshold": 3.0, "min_nonzero_points": 10},
                "anomaly_export": {"enabled": False, "max_samples": 10},
                "visualization": {
                    "enabled": False,
                    "time_series": False,
                    "histograms": False,
                    "histogram_bins": 10,
                    "heatmaps": {"enabled": False, "types": [], "num_time_bins": 10, "num_spread_bins": 10},
                    "spread_clipping": {"enabled": False, "num_sigma": 5.0},
                    "depth_heatmap": {"enabled": False, "num_price_bins": 10},
                },
                "label_checks": {"enabled": False, "num_examples": 1, "max_examples_per_figure": 1},
                "gap_checks": {"enabled": False, "large_gap_multiplier": 2.0, "very_large_gap_multiplier": 6.0},
            },
            "logging": {
                "level": "INFO",
                "colored_output": False,
                "log_function_names": False,
                "colors": {
                    "function_names": "cyan",
                    "parameter_names": "yellow",
                    "parameter_values": "green",
                    "info": "green",
                    "warning": "yellow",
                    "error": "red",
                    "debug": "blue",
                },
            },
            "security": {
                "environment_variables": [],
                "validation": {"check_env_vars_at_startup": False, "fail_if_missing": False},
            },
        }

        metadata = {
            "num_samples": num_samples,
            "anchor_indices": list(range(num_samples)),
        }

        labels = [0] * num_samples
        targets = {
            "labels_up_intensity": labels,
            "labels_down_intensity": labels,
        }

        data_object = {
            "metadata": metadata,
            "order_books": {},
            "temporal_features": {},
            "targets": targets,
            "external_data": {},
        }

        model = _DummyEvalModel(num_classes=num_classes)

        evaluate_model(config, model, data_object)

        self.assertIsNotNone(model.last_input_shape)
        self.assertEqual(model.last_input_shape[0], int(num_samples * 0.15))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
