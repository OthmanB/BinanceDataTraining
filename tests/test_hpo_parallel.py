"""Unit tests for parallel HPO helper utilities."""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from models.hyperparameter_tuning import (
    _allocate_trials_to_workers,
    _apply_worker_resource,
    _apply_worker_runtime_environment,
    _compute_next_batch_size,
    _compute_safe_batch_cap_from_memory,
    _evaluate_trial_objective,
    _is_resource_exhaustion_error,
    _resolve_failure_objective_value,
    _resolve_parallel_settings,
    _resolve_regime_settings,
    _resolve_worker_runtime_options,
)


class TestHPOParallelHelpers(unittest.TestCase):
    def test_allocate_trials_balanced(self) -> None:
        self.assertEqual(_allocate_trials_to_workers(7, 3), [3, 2, 2])

    def test_allocate_trials_handles_empty_inputs(self) -> None:
        self.assertEqual(_allocate_trials_to_workers(0, 3), [])
        self.assertEqual(_allocate_trials_to_workers(3, 0), [])

    def test_apply_worker_resource_cpu(self) -> None:
        config = {"training": {"runtime": {"device": "gpu", "gpu_visible_devices": "0"}}}
        updated = _apply_worker_resource(config, "cpu")
        self.assertEqual(updated["training"]["runtime"]["device"], "cpu")
        self.assertIsNone(updated["training"]["runtime"]["gpu_visible_devices"])

    def test_apply_worker_resource_gpu(self) -> None:
        config = {"training": {"runtime": {"device": "cpu", "gpu_visible_devices": None}}}
        updated = _apply_worker_resource(config, "gpu:1")
        self.assertEqual(updated["training"]["runtime"]["device"], "gpu")
        self.assertEqual(updated["training"]["runtime"]["gpu_visible_devices"], "1")

    def test_resolve_parallel_settings_requires_mapping(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings({})

    def test_resolve_parallel_settings_parses_values(self) -> None:
        hpo_cfg = {
            "parallel": {
                "enabled": True,
                "max_trials_per_worker_process": 2,
                "resources": ["GPU:0", "cpu", "  gpu:1  "],
                "storage_uri": "sqlite:///tmp/hpo.db",
                "study_name": "my_study",
            }
        }
        settings = _resolve_parallel_settings(hpo_cfg)
        self.assertTrue(bool(settings["enabled"]))
        self.assertEqual(settings["resources"], ["gpu:0", "cpu", "gpu:1"])
        self.assertEqual(settings["storage_uri"], "sqlite:///tmp/hpo.db")
        self.assertEqual(settings["study_name"], "my_study")
        self.assertEqual(int(settings["max_trials_per_worker_process"]), 2)
        self.assertTrue(bool(settings["rss_watchdog_enabled"]))
        self.assertEqual(float(settings["rss_watchdog_max_worker_rss_gb"]), 28.0)

    def test_resolve_parallel_settings_requires_positive_worker_cap(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_rss_threshold(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "rss_watchdog_max_worker_rss_gb": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_worker_runtime_options_parses_explicit_values(self) -> None:
        runtime_cfg = {
            "gpu_memory_growth": True,
            "gpu_allocator": "cuda_malloc_async",
            "gpu_init_lock_enabled": True,
            "gpu_init_stagger_seconds": 1.5,
        }
        options = _resolve_worker_runtime_options(runtime_cfg)
        self.assertTrue(bool(options["gpu_memory_growth"]))
        self.assertEqual(options["gpu_allocator"], "cuda_malloc_async")
        self.assertTrue(bool(options["gpu_init_lock_enabled"]))
        self.assertEqual(float(options["gpu_init_stagger_seconds"]), 1.5)

    def test_resolve_regime_settings_defaults_when_missing(self) -> None:
        settings = _resolve_regime_settings({})
        self.assertFalse(bool(settings["enabled"]))
        self.assertEqual(int(settings["max_retry_attempts"]), 2)
        self.assertEqual(str(settings["failure_policy"]), "prune")

    def test_resolve_regime_settings_parses_values(self) -> None:
        hpo_cfg = {
            "regime": {
                "enabled": True,
                "retry_on_oom": True,
                "max_retry_attempts": 3,
                "batch_backoff_factor": 0.6,
                "min_batch_size": 8,
                "failure_policy": "penalize",
                "failure_penalty_value": 42.0,
                "max_vram_fraction": 0.9,
                "vram_penalty_weight": 5.0,
                "low_utilization_penalty": {
                    "enabled": True,
                    "min_samples_per_second": 10.0,
                    "weight": 3.0,
                },
                "safe_envelope": {
                    "enabled": True,
                    "min_success_trials": 4,
                    "headroom_fraction": 0.95,
                    "persistence_enabled": True,
                },
            }
        }
        settings = _resolve_regime_settings(hpo_cfg)
        self.assertTrue(bool(settings["enabled"]))
        self.assertEqual(int(settings["max_retry_attempts"]), 3)
        self.assertEqual(str(settings["failure_policy"]), "penalize")
        self.assertEqual(float(settings["max_vram_fraction"]), 0.9)

    def test_resource_exhaustion_classifier(self) -> None:
        self.assertTrue(_is_resource_exhaustion_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")))
        self.assertTrue(_is_resource_exhaustion_error(RuntimeError("No DNN in stream executor")))
        self.assertFalse(_is_resource_exhaustion_error(RuntimeError("network timeout")))

    def test_compute_next_batch_size_respects_floor(self) -> None:
        self.assertEqual(_compute_next_batch_size(24, 0.5, 8), 12)
        self.assertEqual(_compute_next_batch_size(9, 0.9, 8), 8)

    def test_compute_safe_batch_cap_from_memory(self) -> None:
        payload = {
            "global": {
                "success_count": 3,
                "failure_count": 1,
                "max_success_batch": 24,
                "min_oom_batch": 28,
            },
            "by_signature": {
                "cnn1=64;cnn2=128;lstm=96": {
                    "success_count": 2,
                    "failure_count": 1,
                    "max_success_batch": 24,
                    "min_oom_batch": 26,
                }
            },
        }
        cap = _compute_safe_batch_cap_from_memory(
            payload,
            signature="cnn1=64;cnn2=128;lstm=96",
            min_batch_size=8,
            min_success_trials=2,
            headroom_fraction=0.95,
        )
        self.assertEqual(cap, 22)

    def test_failure_objective_value_by_direction(self) -> None:
        self.assertEqual(_resolve_failure_objective_value("minimize", 123.0), 123.0)
        self.assertEqual(_resolve_failure_objective_value("maximize", 123.0), -123.0)

    def test_apply_worker_runtime_environment_cpu_masks_gpus(self) -> None:
        base_config = {
            "training": {
                "runtime": {
                    "device": "gpu",
                    "gpu_visible_devices": "0",
                    "gpu_memory_growth": False,
                    "gpu_allocator": "default",
                    "gpu_init_lock_enabled": False,
                    "gpu_init_stagger_seconds": 0,
                }
            },
            "mlflow": {"local_tmp_dir": "/tmp"},
        }
        with mock.patch.dict("os.environ", {}, clear=True):
            _apply_worker_runtime_environment(base_config, "cpu")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def test_apply_worker_runtime_environment_gpu_sets_allocator_and_growth(self) -> None:
        class _FakeTFConfig:
            def __init__(self) -> None:
                self.device = types.SimpleNamespace(name="/physical_device:GPU:0")
                self.experimental = types.SimpleNamespace(set_memory_growth=self._set_memory_growth)
                self.calls = []

            def list_physical_devices(self, device_type: str):
                return [self.device] if device_type == "GPU" else []

            def _set_memory_growth(self, device: object, enabled: bool) -> None:
                self.calls.append((device, enabled))

        fake_tf_config = _FakeTFConfig()
        fake_tf = types.SimpleNamespace(config=fake_tf_config)
        base_config = {
            "training": {
                "runtime": {
                    "device": "gpu",
                    "gpu_visible_devices": "0",
                    "gpu_memory_growth": True,
                    "gpu_allocator": "cuda_malloc_async",
                    "gpu_init_lock_enabled": False,
                    "gpu_init_stagger_seconds": 0,
                }
            },
            "mlflow": {"local_tmp_dir": "/tmp"},
        }
        with mock.patch.dict("sys.modules", {"tensorflow": fake_tf}):
            with mock.patch.dict("os.environ", {}, clear=True):
                _apply_worker_runtime_environment(base_config, "gpu:0")
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0")
                self.assertEqual(os.environ.get("TF_GPU_ALLOCATOR"), "cuda_malloc_async")
        self.assertEqual(fake_tf_config.calls, [(fake_tf_config.device, True)])

    def test_evaluate_trial_objective_calls_cleanup_on_success(self) -> None:
        trial = types.SimpleNamespace(number=1, user_attrs={})

        def _set_user_attr(key: str, value: object) -> None:
            trial.user_attrs[key] = value

        trial.set_user_attr = _set_user_attr

        base_config = {
            "mlflow": {
                "local_tmp_dir": "/tmp",
                "artifact_logging": {"trained_model": True},
                "model_registry": {"register_model": True},
            },
            "training": {"runtime": {"device": "gpu", "gpu_visible_devices": "0"}},
            "model": {"cnn_lstm": {"filters": [64, 128], "lstm_units": 64}},
        }

        def _run_training_pipeline(cfg: dict, _data_obj: object) -> None:
            cfg["_hpo_last_metric"] = 0.123

        fake_pipeline = types.SimpleNamespace(run_training_pipeline=_run_training_pipeline)

        with mock.patch.dict("sys.modules", {"training.pipeline": fake_pipeline}):
            with mock.patch("models.hyperparameter_tuning._sample_hyperparameters", return_value={"batch_size": 16}):
                with mock.patch("models.hyperparameter_tuning._apply_hyperparameters", side_effect=lambda cfg, _p: cfg):
                    with mock.patch("models.hyperparameter_tuning._cleanup_trial_runtime") as cleanup_mock:
                        value = _evaluate_trial_objective(
                            base_config,
                            None,
                            {"search_space": {}},
                            "loss",
                            "minimize",
                            False,
                            trial,
                            resource=None,
                            study_name="test",
                        )
        self.assertEqual(value, 0.123)
        self.assertEqual(cleanup_mock.call_count, 1)

    def test_evaluate_trial_objective_calls_cleanup_on_error(self) -> None:
        trial = types.SimpleNamespace(number=2, user_attrs={})

        def _set_user_attr(key: str, value: object) -> None:
            trial.user_attrs[key] = value

        trial.set_user_attr = _set_user_attr

        base_config = {
            "mlflow": {
                "local_tmp_dir": "/tmp",
                "artifact_logging": {"trained_model": True},
                "model_registry": {"register_model": True},
            },
            "training": {"runtime": {"device": "gpu", "gpu_visible_devices": "0"}},
            "model": {"cnn_lstm": {"filters": [64, 128], "lstm_units": 64}},
        }

        def _run_training_pipeline(_cfg: dict, _data_obj: object) -> None:
            raise RuntimeError("boom")

        fake_pipeline = types.SimpleNamespace(run_training_pipeline=_run_training_pipeline)

        with mock.patch.dict("sys.modules", {"training.pipeline": fake_pipeline}):
            with mock.patch("models.hyperparameter_tuning._sample_hyperparameters", return_value={"batch_size": 16}):
                with mock.patch("models.hyperparameter_tuning._apply_hyperparameters", side_effect=lambda cfg, _p: cfg):
                    with mock.patch("models.hyperparameter_tuning._cleanup_trial_runtime") as cleanup_mock:
                        with self.assertRaises(RuntimeError):
                            _evaluate_trial_objective(
                                base_config,
                                None,
                                {"search_space": {}},
                                "loss",
                                "minimize",
                                False,
                                trial,
                                resource=None,
                                study_name="test",
                            )
        self.assertEqual(cleanup_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
