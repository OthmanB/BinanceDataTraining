"""Unit tests for mlflow_integration.model_registry.

These tests run without a real MLflow installation by providing fake modules.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest import mock


class _FakeClient:
    def __init__(self, listings: dict[tuple[str, str], list]) -> None:
        self._listings = listings

    def list_artifacts(self, run_id: str, path: str = "") -> list:
        return list(self._listings.get((str(run_id), str(path)), []))


class TestMlflowModelRegistry(unittest.TestCase):
    def test_register_model_skips_when_artifact_missing(self) -> None:
        from mlflow_integration.model_registry import register_model

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.register_model = mock.MagicMock()
        fake_mlflow.active_run = mock.MagicMock(
            return_value=SimpleNamespace(info=SimpleNamespace(run_id="run-1"), data=SimpleNamespace(tags={}))
        )

        fake_tracking = types.ModuleType("mlflow.tracking")
        fake_tracking.MlflowClient = mock.MagicMock(return_value=_FakeClient(listings={}))

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tracking": fake_tracking}):
            register_model("demo")

        fake_mlflow.register_model.assert_not_called()

    def test_register_model_registers_when_artifact_present(self) -> None:
        from mlflow_integration.model_registry import register_model

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.register_model = mock.MagicMock(return_value=SimpleNamespace(version="7"))
        fake_mlflow.active_run = mock.MagicMock(
            return_value=SimpleNamespace(info=SimpleNamespace(run_id="run-2"), data=SimpleNamespace(tags={}))
        )

        fake_tracking = types.ModuleType("mlflow.tracking")
        listings = {
            ("run-2", ""): [SimpleNamespace(path="model")],
        }
        fake_tracking.MlflowClient = mock.MagicMock(return_value=_FakeClient(listings=listings))

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tracking": fake_tracking}):
            register_model("demo")

        fake_mlflow.register_model.assert_called_once()
        args, kwargs = fake_mlflow.register_model.call_args
        self.assertEqual(kwargs.get("name"), "demo")
        self.assertEqual(kwargs.get("model_uri"), "runs:/run-2/model")


if __name__ == "__main__":
    unittest.main()
