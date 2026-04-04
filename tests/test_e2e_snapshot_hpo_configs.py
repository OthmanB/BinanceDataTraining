"""Configuration-level tests for snapshot + HPO e2e profiles."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from utils.config_loader import load_config


class TestE2ESnapshotHPOConfigs(unittest.TestCase):
    def _env(self) -> dict:
        return {
            "DATABASE_URI": "http://example-db",
            "DATABASE_URI_HIST": "http://example-db-hist",
            "DATABASE_URI_LIVE": "http://example-db-live",
            "MLFLOW_TRACKING_URI": "http://mlflow",
        }

    def test_trial_snapshot_hpo_config_loads(self) -> None:
        with mock.patch.dict(os.environ, self._env(), clear=False):
            cfg = load_config(
                config_path="config/e2e_trial_13_snapshot_hpo_trial.yaml",
                schema_path="config/validation_schema.yaml",
            )

        self.assertEqual(cfg["run_mode"]["mode"], "trial")
        self.assertTrue(bool(cfg["snapshot"]["enabled"]))
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["enabled"]))
        self.assertTrue(bool(cfg["training"]["sequential_training"]["enabled"]))

    def test_production_snapshot_hpo_config_loads(self) -> None:
        with mock.patch.dict(os.environ, self._env(), clear=False):
            cfg = load_config(
                config_path="config/e2e_trial_14_snapshot_hpo_production.yaml",
                schema_path="config/validation_schema.yaml",
            )

        self.assertEqual(cfg["run_mode"]["mode"], "production")
        self.assertTrue(bool(cfg["snapshot"]["enabled"]))
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["enabled"]))
        self.assertTrue(bool(cfg["training"]["sequential_training"]["enabled"]))

    def test_parallel_trial_snapshot_hpo_config_loads(self) -> None:
        with mock.patch.dict(os.environ, self._env(), clear=False):
            cfg = load_config(
                config_path="config/e2e_trial_15_parallel_hpo_trial.yaml",
                schema_path="config/validation_schema.yaml",
            )

        self.assertEqual(cfg["run_mode"]["mode"], "trial")
        self.assertTrue(bool(cfg["snapshot"]["enabled"]))
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["enabled"]))
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["parallel"]["enabled"]))
        self.assertEqual(
            cfg["hyperparameter_optimization"]["parallel"]["resources"],
            ["gpu:0", "gpu:1"],
        )
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["regime"]["enabled"]))

    def test_architecture_stress_trial_config_loads(self) -> None:
        with mock.patch.dict(os.environ, self._env(), clear=False):
            cfg = load_config(
                config_path="config/e2e_trial_16_architecture_regime_trial.yaml",
                schema_path="config/validation_schema.yaml",
            )

        self.assertEqual(cfg["run_mode"]["mode"], "trial")
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["parallel"]["enabled"]))
        self.assertEqual(
            cfg["hyperparameter_optimization"]["parallel"]["resources"],
            ["gpu:0", "gpu:1"],
        )
        self.assertEqual(
            cfg["hyperparameter_optimization"]["search_space"]["batch_size"],
            [24, 24],
        )
        self.assertTrue(bool(cfg["hyperparameter_optimization"]["regime"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
