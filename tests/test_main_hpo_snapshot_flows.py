"""Mocked integration tests for main snapshot+HPO control flow."""

from __future__ import annotations

import argparse
import logging
import types
import unittest
from unittest import mock

import main


def _base_config(mode: str) -> dict:
    return {
        "run_mode": {"mode": mode},
        "training": {
            "runtime": {"device": "cpu", "gpu_visible_devices": None},
            "sequential_training": {
                "enabled": True,
                "window_days": 1,
                "cleanup_completed_windows": False,
            },
        },
        "data": {
            "asset_pairs": {"target_asset": "BTCUSDT"},
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            },
        },
        "model": {"architecture": "cnn_lstm"},
        "mlflow": {"run_naming": {"pattern": "{asset}_{model}_{timestamp}"}},
        "snapshot": {"enabled": True},
        "hyperparameter_optimization": {"enabled": True},
    }


class TestMainSnapshotHPOFlows(unittest.TestCase):
    @mock.patch("main.end_run")
    @mock.patch("main.run_training_pipeline")
    @mock.patch("main._evaluate_snapshot_sequential")
    @mock.patch("main.run_hyperparameter_search")
    @mock.patch("main.start_run")
    @mock.patch("main.validate_environment")
    @mock.patch("main.setup_colored_logging")
    @mock.patch("main.load_config")
    @mock.patch("main._parse_args")
    def test_trial_snapshot_hpo_runs_search_and_skips_final_train_eval(
        self,
        mock_parse_args: mock.MagicMock,
        mock_load_config: mock.MagicMock,
        mock_setup_logger: mock.MagicMock,
        mock_validate_env: mock.MagicMock,
        mock_start_run: mock.MagicMock,
        mock_run_hpo: mock.MagicMock,
        mock_eval_seq: mock.MagicMock,
        mock_run_training: mock.MagicMock,
        mock_end_run: mock.MagicMock,
    ) -> None:
        config = _base_config("trial")

        mock_parse_args.return_value = argparse.Namespace(
            config="config/e2e_trial_13_snapshot_hpo_trial.yaml",
            schema="config/validation_schema.yaml",
        )
        mock_load_config.return_value = config
        mock_setup_logger.return_value = logging.getLogger("tests.main_trial")
        mock_start_run.return_value = types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))
        mock_run_hpo.return_value = config

        exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mock_validate_env.assert_called_once()
        mock_run_hpo.assert_called_once_with(config, None)
        mock_run_training.assert_not_called()
        mock_eval_seq.assert_not_called()
        mock_end_run.assert_called_once()

    @mock.patch("main.end_run")
    @mock.patch("main._evaluate_snapshot_sequential")
    @mock.patch("main.run_training_pipeline")
    @mock.patch("main.run_hyperparameter_search")
    @mock.patch("main.start_run")
    @mock.patch("main.validate_environment")
    @mock.patch("main.setup_colored_logging")
    @mock.patch("main.load_config")
    @mock.patch("main._parse_args")
    def test_production_snapshot_hpo_runs_search_train_and_eval(
        self,
        mock_parse_args: mock.MagicMock,
        mock_load_config: mock.MagicMock,
        mock_setup_logger: mock.MagicMock,
        mock_validate_env: mock.MagicMock,
        mock_start_run: mock.MagicMock,
        mock_run_hpo: mock.MagicMock,
        mock_run_training: mock.MagicMock,
        mock_eval_seq: mock.MagicMock,
        mock_end_run: mock.MagicMock,
    ) -> None:
        config = _base_config("production")
        best_config = _base_config("production")
        best_config["training"]["batch_size"] = 14
        model = object()

        mock_parse_args.return_value = argparse.Namespace(
            config="config/e2e_trial_14_snapshot_hpo_production.yaml",
            schema="config/validation_schema.yaml",
        )
        mock_load_config.return_value = config
        mock_setup_logger.return_value = logging.getLogger("tests.main_production")
        mock_start_run.return_value = types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-2"))
        mock_run_hpo.return_value = best_config
        mock_run_training.return_value = model

        exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mock_validate_env.assert_called_once()
        mock_run_hpo.assert_called_once_with(config, None)
        mock_run_training.assert_called_once_with(best_config, None)
        mock_eval_seq.assert_called_once()

        eval_args = mock_eval_seq.call_args.args
        self.assertIs(eval_args[0], best_config)
        self.assertIs(eval_args[1], model)
        self.assertIsInstance(eval_args[2], logging.Logger)
        mock_end_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
