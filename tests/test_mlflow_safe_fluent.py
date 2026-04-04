"""Unit tests for mlflow_integration.safe_fluent.

These tests run without a real MLflow installation by providing fake modules.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest import mock


class TestSafeFluent(unittest.TestCase):
    def test_log_keras_model_returns_false_without_active_run(self) -> None:
        from mlflow_integration.safe_fluent import log_keras_model_to_active_run

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.active_run = mock.MagicMock(return_value=None)

        fake_tf = types.ModuleType("mlflow.tensorflow")
        fake_tf.log_model = mock.MagicMock()

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tensorflow": fake_tf}):
            ok = log_keras_model_to_active_run(model=object())

        self.assertFalse(ok)
        fake_tf.log_model.assert_not_called()

    def test_log_keras_model_logs_when_active_run_exists(self) -> None:
        from mlflow_integration.safe_fluent import log_keras_model_to_active_run

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.active_run = mock.MagicMock(return_value=SimpleNamespace(info=SimpleNamespace(run_id="r")))

        fake_tf = types.ModuleType("mlflow.tensorflow")
        fake_tf.log_model = mock.MagicMock()

        sig = {"sig": 1}
        model = object()

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tensorflow": fake_tf}):
            ok = log_keras_model_to_active_run(model=model, signature=sig)

        self.assertTrue(ok)
        fake_tf.log_model.assert_called_once()
        _args, kwargs = fake_tf.log_model.call_args
        self.assertEqual(kwargs.get("signature"), sig)

    def test_register_skipped_when_log_model_fails(self) -> None:
        from mlflow_integration.safe_fluent import log_keras_model_and_register_if_enabled

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.active_run = mock.MagicMock(return_value=SimpleNamespace(info=SimpleNamespace(run_id="r")))

        fake_tf = types.ModuleType("mlflow.tensorflow")
        fake_tf.log_model = mock.MagicMock(side_effect=RuntimeError("boom"))

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tensorflow": fake_tf}):
            with mock.patch("mlflow_integration.model_registry.register_model") as reg:
                ok = log_keras_model_and_register_if_enabled(
                    model=object(),
                    model_name="demo",
                    register_enabled=True,
                )

        self.assertFalse(ok)
        reg.assert_not_called()

    def test_register_called_when_model_logged(self) -> None:
        from mlflow_integration.safe_fluent import log_keras_model_and_register_if_enabled

        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.active_run = mock.MagicMock(return_value=SimpleNamespace(info=SimpleNamespace(run_id="r")))

        fake_tf = types.ModuleType("mlflow.tensorflow")
        fake_tf.log_model = mock.MagicMock()

        with mock.patch.dict("sys.modules", {"mlflow": fake_mlflow, "mlflow.tensorflow": fake_tf}):
            with mock.patch("mlflow_integration.model_registry.register_model") as reg:
                ok = log_keras_model_and_register_if_enabled(
                    model=object(),
                    model_name="demo",
                    register_enabled=True,
                    artifact_path="model",
                )

        self.assertTrue(ok)
        reg.assert_called_once()
        _args, kwargs = reg.call_args
        self.assertEqual(_args[0], "demo")
        self.assertEqual(kwargs.get("artifact_path"), "model")


if __name__ == "__main__":
    unittest.main()
