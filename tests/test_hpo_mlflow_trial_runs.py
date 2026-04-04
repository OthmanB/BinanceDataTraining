"""Tests for MLflow run handling in parallel HPO.

These tests ensure that HPO trial workers create explicit MLflow runs (instead
of relying on MLflow's implicit auto-named run creation) and correctly link trial
runs to the supervising parent run.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class TestHPOMlflowTrialRuns(unittest.TestCase):
    def _build_mlflow_config(self, tracking_uri: str, experiment_name: str) -> dict:
        return {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "experiment_name": experiment_name,
            },
            "data": {"asset_pairs": {"target_asset": None}},
        }

    def test_trial_run_links_to_parent_when_no_active_run(self) -> None:
        from models.hyperparameter_tuning import (  # noqa: PLC0415
            _configure_mlflow_from_config,
            _start_hpo_trial_mlflow_run,
            _try_import_mlflow,
        )

        mlflow = _try_import_mlflow()
        if mlflow is None:
            raise unittest.SkipTest("MLflow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = Path(os.path.join(tmpdir, "mlruns")).as_uri()
            experiment_name = "unittest_hpo_trial_run_parent_link"
            config = self._build_mlflow_config(tracking_uri, experiment_name)

            mlflow = _configure_mlflow_from_config(config)
            self.assertIsNotNone(mlflow)

            parent_run_id: str
            with mlflow.start_run(run_name="parent"):
                parent_run_id = mlflow.active_run().info.run_id

            run = _start_hpo_trial_mlflow_run(
                config=config,
                study_name="study",
                trial_number=3,
                resource="gpu:0",
                parent_run_id=parent_run_id,
            )
            self.assertIsNotNone(run)
            try:
                active = mlflow.active_run()
                self.assertIsNotNone(active)

                run_id = active.info.run_id
                client = mlflow.tracking.MlflowClient()
                fetched = client.get_run(run_id)

                self.assertEqual(fetched.data.tags.get("run_type"), "hpo_trial")
                self.assertEqual(fetched.data.tags.get("hpo.study_name"), "study")
                self.assertEqual(fetched.data.tags.get("hpo.trial_number"), "3")
                self.assertEqual(fetched.data.tags.get("hpo.resource"), "gpu:0")
                self.assertEqual(fetched.data.tags.get("mlflow.parentRunId"), parent_run_id)
                self.assertEqual(
                    fetched.data.tags.get("mlflow.runName"),
                    "asset_hpo_trial_3_gpu_0",
                )
            finally:
                mlflow.end_run()

    def test_trial_run_is_nested_when_parent_run_active(self) -> None:
        from models.hyperparameter_tuning import (  # noqa: PLC0415
            _configure_mlflow_from_config,
            _start_hpo_trial_mlflow_run,
            _try_import_mlflow,
        )

        mlflow = _try_import_mlflow()
        if mlflow is None:
            raise unittest.SkipTest("MLflow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = Path(os.path.join(tmpdir, "mlruns")).as_uri()
            experiment_name = "unittest_hpo_trial_run_nested"
            config = self._build_mlflow_config(tracking_uri, experiment_name)
            mlflow = _configure_mlflow_from_config(config)
            self.assertIsNotNone(mlflow)

            with mlflow.start_run(run_name="parent"):
                parent_run_id = mlflow.active_run().info.run_id

                run = _start_hpo_trial_mlflow_run(
                    config=config,
                    study_name="study",
                    trial_number=1,
                    resource="cpu",
                    parent_run_id=None,
                )
                self.assertIsNotNone(run)
                try:
                    active = mlflow.active_run()
                    self.assertIsNotNone(active)
                    run_id = active.info.run_id
                    client = mlflow.tracking.MlflowClient()
                    fetched = client.get_run(run_id)

                    self.assertEqual(fetched.data.tags.get("mlflow.parentRunId"), parent_run_id)
                    self.assertEqual(fetched.data.tags.get("run_type"), "hpo_trial")
                finally:
                    mlflow.end_run()


if __name__ == "__main__":
    unittest.main()
