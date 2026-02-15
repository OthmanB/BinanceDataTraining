"""Unit tests for parallel HPO helper utilities."""

from __future__ import annotations

import os
import types
import unittest
import copy
from unittest import mock

from models.hyperparameter_tuning import (
    _allocate_trials_to_workers,
    _apply_hpo_resume_namespace,
    _apply_worker_resource,
    _apply_worker_runtime_environment,
    _compute_next_batch_size,
    _compute_safe_batch_cap_from_memory,
    _extract_trial_details,
    _evaluate_trial_objective,
    _is_resource_exhaustion_error,
    _record_trial_phase_memory_event,
    _select_wave_resources,
    _should_trigger_rss_watchdog,
    _resolve_failure_objective_value,
    _resolve_parallel_settings,
    _resolve_regime_settings,
    _resolve_worker_runtime_options,
    _update_adaptive_scheduler_state,
    run_hyperparameter_search,
)


class TestHPOParallelHelpers(unittest.TestCase):
    def test_run_hyperparameter_search_resets_study_with_stale_running_trials(self) -> None:
        class _FakeTrial:
            def __init__(self, state_name: str, number: int, value: float) -> None:
                self.state = types.SimpleNamespace(name=state_name)
                self.number = int(number)
                self.value = float(value)
                self.params = {"batch_size": 8}
                self.user_attrs = {"batch_size_effective": 8}
                self.datetime_start = None
                self.datetime_complete = None

        class _FakeStudy:
            def __init__(self, trials: list[object], best_trial: object) -> None:
                self.trials = trials
                self.best_trial = best_trial

        stale_running = _FakeStudy(
            trials=[_FakeTrial("RUNNING", 0, 0.0)],
            best_trial=_FakeTrial("RUNNING", 0, 0.0),
        )

        complete_a = _FakeTrial("COMPLETE", 1, 0.8)
        complete_b = _FakeTrial("COMPLETE", 2, 0.7)
        clean_study = _FakeStudy(trials=[complete_a, complete_b], best_trial=complete_b)

        create_calls: list[dict] = []
        delete_calls: list[dict] = []

        def _create_study(*, direction: str, study_name: str, storage: str, load_if_exists: bool = False):
            create_calls.append(
                {
                    "direction": direction,
                    "study_name": study_name,
                    "storage": storage,
                    "load_if_exists": bool(load_if_exists),
                }
            )
            if load_if_exists:
                return stale_running
            return clean_study

        def _delete_study(*, study_name: str, storage: str) -> None:
            delete_calls.append({"study_name": study_name, "storage": storage})

        fake_optuna = types.SimpleNamespace(
            create_study=_create_study,
            load_study=lambda **_kwargs: clean_study,
            delete_study=_delete_study,
        )

        config = {
            "snapshot": {"enabled": False},
            "mlflow": {"local_tmp_dir": "/tmp"},
            "hyperparameter_optimization": {
                "enabled": True,
                "framework": "optuna",
                "n_trials": 2,
                "direction": "minimize",
                "metric": "loss",
                "trial_model_logging": {"enabled": False},
                "search_space": {"batch_size": [8, 16]},
                "parallel": {
                    "enabled": True,
                    "max_trials_per_worker_process": 1,
                    "resources": ["gpu:0"],
                    "storage_uri": "sqlite:////tmp/test_hpo_stale_running.db",
                    "study_name": "test_hpo_stale_running",
                    "resume_study": False,
                },
            },
        }

        with mock.patch.dict("sys.modules", {"optuna": fake_optuna}):
            with mock.patch("models.hyperparameter_tuning._apply_hyperparameters", side_effect=lambda cfg, _p: copy.deepcopy(cfg)):
                with mock.patch(
                    "models.hyperparameter_tuning._resolve_best_params_for_final_training",
                    return_value=({"batch_size": 8}, 8, 8),
                ):
                    with mock.patch("observability.run_state.get_run_state_writer", return_value=None):
                        with mock.patch("models.hyperparameter_tuning._try_import_mlflow", return_value=None):
                            best_config = run_hyperparameter_search(config, data_object=None)

        self.assertIsInstance(best_config, dict)
        self.assertEqual(len(delete_calls), 1)
        self.assertEqual(delete_calls[0]["study_name"], "test_hpo_stale_running")
        self.assertEqual(delete_calls[0]["storage"], "sqlite:////tmp/test_hpo_stale_running.db")
        self.assertGreaterEqual(len(create_calls), 2)
        self.assertTrue(bool(create_calls[0]["load_if_exists"]))
        self.assertFalse(bool(create_calls[1]["load_if_exists"]))

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
                "rss_watchdog_startup_grace_seconds": 90,
                "rss_watchdog_require_trial_start": True,
                "rss_watchdog_startup_timeout_seconds": 2400,
                "rss_watchdog_single_worker_fallback_enabled": True,
                "rss_watchdog_max_restarts_before_single_worker": 3,
                "adaptive_scheduler_enabled": True,
                "adaptive_scheduler_min_workers": 1,
                "adaptive_scheduler_recovery_waves": 3,
                "adaptive_scheduler_min_trials_per_worker_process": 1,
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
        self.assertEqual(float(settings["rss_watchdog_startup_grace_seconds"]), 90.0)
        self.assertTrue(bool(settings["rss_watchdog_require_trial_start"]))
        self.assertEqual(float(settings["rss_watchdog_startup_timeout_seconds"]), 2400.0)
        self.assertTrue(bool(settings["rss_watchdog_single_worker_fallback_enabled"]))
        self.assertEqual(int(settings["rss_watchdog_max_restarts_before_single_worker"]), 3)
        self.assertTrue(bool(settings["adaptive_scheduler_enabled"]))
        self.assertEqual(int(settings["adaptive_scheduler_min_workers"]), 1)
        self.assertEqual(int(settings["adaptive_scheduler_recovery_waves"]), 3)
        self.assertEqual(int(settings["adaptive_scheduler_min_trials_per_worker_process"]), 1)

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

    def test_resolve_parallel_settings_rejects_invalid_startup_grace(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "rss_watchdog_startup_grace_seconds": -1,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_startup_timeout(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "rss_watchdog_startup_timeout_seconds": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_fallback_restart_threshold(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "rss_watchdog_max_restarts_before_single_worker": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_adaptive_min_workers(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "adaptive_scheduler_min_workers": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_adaptive_recovery_waves(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "adaptive_scheduler_recovery_waves": 0,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_resolve_parallel_settings_rejects_invalid_adaptive_min_trials_cap(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_parallel_settings(
                {
                    "parallel": {
                        "enabled": True,
                        "max_trials_per_worker_process": 2,
                        "adaptive_scheduler_min_trials_per_worker_process": 3,
                        "resources": ["gpu:0"],
                    }
                }
            )

    def test_should_trigger_rss_watchdog_honors_startup_grace(self) -> None:
        should_trigger, _, _ = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={123: 10_000},
            rss_watchdog_limit_bytes=1,
            wave_started_monotonic=100.0,
            now_monotonic=120.0,
            startup_grace_seconds=30.0,
            require_trial_start=False,
            trial_started_in_wave=False,
            startup_timeout_seconds=600.0,
        )
        self.assertFalse(should_trigger)

    def test_should_trigger_rss_watchdog_requires_trial_start_before_timeout(self) -> None:
        should_trigger, _, _ = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={123: 10_000},
            rss_watchdog_limit_bytes=1,
            wave_started_monotonic=100.0,
            now_monotonic=500.0,
            startup_grace_seconds=30.0,
            require_trial_start=True,
            trial_started_in_wave=False,
            startup_timeout_seconds=1_000.0,
        )
        self.assertFalse(should_trigger)

    def test_should_trigger_rss_watchdog_allows_timeout_without_trial_start(self) -> None:
        should_trigger, pid, rss = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={123: 10_000},
            rss_watchdog_limit_bytes=1,
            wave_started_monotonic=100.0,
            now_monotonic=1_500.0,
            startup_grace_seconds=30.0,
            require_trial_start=True,
            trial_started_in_wave=False,
            startup_timeout_seconds=1_000.0,
        )
        self.assertTrue(should_trigger)
        self.assertEqual(pid, 123)
        self.assertEqual(rss, 10_000)

    def test_should_trigger_rss_watchdog_with_started_trial(self) -> None:
        should_trigger, pid, rss = _should_trigger_rss_watchdog(
            rss_watchdog_enabled=True,
            rss_by_pid={123: 10_000},
            rss_watchdog_limit_bytes=1,
            wave_started_monotonic=100.0,
            now_monotonic=200.0,
            startup_grace_seconds=30.0,
            require_trial_start=True,
            trial_started_in_wave=True,
            startup_timeout_seconds=1_000.0,
        )
        self.assertTrue(should_trigger)
        self.assertEqual(pid, 123)
        self.assertEqual(rss, 10_000)

    def test_select_wave_resources_respects_single_worker_override(self) -> None:
        selected = _select_wave_resources(["gpu:0", "gpu:1"], remaining_trials=5, force_single_worker=True)
        self.assertEqual(selected, ["gpu:0"])

    def test_update_adaptive_scheduler_state_scales_down_on_watchdog(self) -> None:
        worker_cap, trial_cap, stable_waves = _update_adaptive_scheduler_state(
            current_worker_cap=3,
            current_trials_per_worker_cap=4,
            max_worker_cap=3,
            max_trials_per_worker_cap=4,
            min_worker_cap=1,
            min_trials_per_worker_cap=1,
            stable_waves=2,
            recovery_waves=2,
            wave_had_progress=True,
            worker_error_count=0,
            watchdog_triggered=True,
        )
        self.assertEqual(worker_cap, 2)
        self.assertEqual(trial_cap, 3)
        self.assertEqual(stable_waves, 0)

    def test_update_adaptive_scheduler_state_scales_down_on_no_progress(self) -> None:
        worker_cap, trial_cap, stable_waves = _update_adaptive_scheduler_state(
            current_worker_cap=2,
            current_trials_per_worker_cap=2,
            max_worker_cap=3,
            max_trials_per_worker_cap=4,
            min_worker_cap=1,
            min_trials_per_worker_cap=1,
            stable_waves=1,
            recovery_waves=3,
            wave_had_progress=False,
            worker_error_count=0,
            watchdog_triggered=False,
        )
        self.assertEqual(worker_cap, 1)
        self.assertEqual(trial_cap, 1)
        self.assertEqual(stable_waves, 0)

    def test_update_adaptive_scheduler_state_scales_up_after_stable_waves(self) -> None:
        worker_cap, trial_cap, stable_waves = _update_adaptive_scheduler_state(
            current_worker_cap=1,
            current_trials_per_worker_cap=1,
            max_worker_cap=3,
            max_trials_per_worker_cap=4,
            min_worker_cap=1,
            min_trials_per_worker_cap=1,
            stable_waves=1,
            recovery_waves=2,
            wave_had_progress=True,
            worker_error_count=0,
            watchdog_triggered=False,
        )
        self.assertEqual(worker_cap, 2)
        self.assertEqual(trial_cap, 2)
        self.assertEqual(stable_waves, 0)

    def test_record_trial_phase_memory_event_tracks_max_per_phase(self) -> None:
        trial = types.SimpleNamespace(user_attrs={})

        def _set_user_attr(key: str, value: object) -> None:
            trial.user_attrs[key] = value

        trial.set_user_attr = _set_user_attr

        with mock.patch("models.hyperparameter_tuning._read_process_rss_bytes", side_effect=[1024, 512, 2048]):
            with mock.patch("models.hyperparameter_tuning.time.time", side_effect=[10.0, 11.0, 12.0]):
                _record_trial_phase_memory_event(
                    trial,
                    phase="after_snapshot_load",
                    resource="gpu:0",
                    details={"n_samples": 100},
                )
                _record_trial_phase_memory_event(
                    trial,
                    phase="after_snapshot_load",
                    resource="gpu:0",
                    details={"n_samples": 100},
                )
                _record_trial_phase_memory_event(
                    trial,
                    phase="after_normalization_stats",
                    resource="gpu:0",
                    details={"val_count": 20},
                )

        events = trial.user_attrs.get("phase_memory_events")
        self.assertIsInstance(events, list)
        assert isinstance(events, list)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].get("phase"), "after_snapshot_load")
        self.assertEqual(events[2].get("phase"), "after_normalization_stats")

        max_by_phase = trial.user_attrs.get("phase_memory_max_by_phase")
        self.assertIsInstance(max_by_phase, dict)
        assert isinstance(max_by_phase, dict)
        self.assertEqual(int(float(max_by_phase["after_snapshot_load"])), 1024)
        self.assertEqual(int(float(max_by_phase["after_normalization_stats"])), 2048)

    def test_extract_trial_details_includes_phase_memory_fields(self) -> None:
        trial = types.SimpleNamespace(
            state=types.SimpleNamespace(name="COMPLETE"),
            number=3,
            value=0.5,
            params={"batch_size": 16},
            datetime_start=None,
            datetime_complete=None,
            user_attrs={
                "phase_memory_events": [
                    {
                        "phase": "after_snapshot_load",
                        "rss_bytes": 1024,
                        "timestamp": 100.0,
                    }
                ],
                "phase_memory_max_by_phase": {
                    "after_snapshot_load": 1024,
                },
            },
        )
        study = types.SimpleNamespace(trials=[trial])

        details = _extract_trial_details(study)

        self.assertEqual(len(details), 1)
        entry = details[0]
        self.assertIn("phase_memory_events", entry)
        self.assertIn("phase_memory_max_by_phase", entry)
        events = entry["phase_memory_events"]
        self.assertIsInstance(events, list)
        self.assertEqual(events[0].get("phase"), "after_snapshot_load")
        max_by_phase = entry["phase_memory_max_by_phase"]
        self.assertEqual(int(float(max_by_phase["after_snapshot_load"])), 1024)

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

    def test_apply_hpo_resume_namespace_sets_namespace_when_enabled(self) -> None:
        config = {
            "training": {
                "sequential_training": {
                    "enabled": True,
                    "resume_enabled": True,
                    "window_days": 1,
                }
            }
        }

        namespace = _apply_hpo_resume_namespace(
            config,
            study_name="demo_study",
            trial_number=12,
            resource="gpu:1",
            attempt=2,
        )

        assert namespace is not None
        self.assertIn("study=demo_study", namespace)
        self.assertIn("trial=12", namespace)
        self.assertIn("attempt=2", namespace)
        self.assertIn("resource=gpu:1", namespace)
        seq_cfg = config["training"]["sequential_training"]
        self.assertEqual(seq_cfg["resume_namespace"], namespace)

    def test_apply_hpo_resume_namespace_skips_when_resume_disabled(self) -> None:
        config = {
            "training": {
                "sequential_training": {
                    "enabled": True,
                    "resume_enabled": False,
                    "window_days": 1,
                }
            }
        }

        namespace = _apply_hpo_resume_namespace(
            config,
            study_name="demo_study",
            trial_number=7,
            resource="gpu:0",
            attempt=0,
        )

        self.assertIsNone(namespace)
        seq_cfg = config["training"]["sequential_training"]
        self.assertNotIn("resume_namespace", seq_cfg)

    def test_evaluate_trial_objective_sets_resume_namespace_for_training_pipeline(self) -> None:
        trial = types.SimpleNamespace(number=9, user_attrs={})

        def _set_user_attr(key: str, value: object) -> None:
            trial.user_attrs[key] = value

        trial.set_user_attr = _set_user_attr

        captured: dict = {}

        base_config = {
            "mlflow": {
                "local_tmp_dir": "/tmp",
                "artifact_logging": {"trained_model": True},
                "model_registry": {"register_model": True},
            },
            "training": {
                "runtime": {"device": "gpu", "gpu_visible_devices": "0"},
                "sequential_training": {
                    "enabled": True,
                    "window_days": 1,
                    "resume_enabled": True,
                },
            },
            "model": {"cnn_lstm": {"filters": [64, 128], "lstm_units": 64}},
        }

        def _run_training_pipeline(cfg: dict, _data_obj: object) -> None:
            phase_probe = cfg.get("_hpo_phase_memory_probe")
            if callable(phase_probe):
                phase_probe("after_snapshot_load", {"n_samples": 321})
            captured["trial_config"] = copy.deepcopy(cfg)
            cfg["_hpo_last_metric"] = 0.321

        fake_pipeline = types.SimpleNamespace(run_training_pipeline=_run_training_pipeline)

        with mock.patch.dict("sys.modules", {"training.pipeline": fake_pipeline}):
            with mock.patch("models.hyperparameter_tuning._sample_hyperparameters", return_value={"batch_size": 16}):
                with mock.patch("models.hyperparameter_tuning._apply_hyperparameters", side_effect=lambda cfg, _p: copy.deepcopy(cfg)):
                    with mock.patch("models.hyperparameter_tuning._cleanup_trial_runtime"):
                        value = _evaluate_trial_objective(
                            base_config,
                            None,
                            {"search_space": {}},
                            "loss",
                            "minimize",
                            False,
                            trial,
                            resource="gpu:0",
                            study_name="parallel_hpo",
                        )

        self.assertEqual(value, 0.321)
        trial_config = captured["trial_config"]
        seq_cfg = trial_config["training"]["sequential_training"]
        namespace = str(seq_cfg.get("resume_namespace", ""))
        self.assertTrue(namespace)
        self.assertIn("study=parallel_hpo", namespace)
        self.assertIn("trial=9", namespace)
        self.assertIn("attempt=0", namespace)
        self.assertIn("resource=gpu:0", namespace)
        self.assertEqual(trial.user_attrs.get("sequential_resume_namespace"), namespace)
        phase_max = trial.user_attrs.get("phase_memory_max_by_phase")
        self.assertIsInstance(phase_max, dict)
        assert isinstance(phase_max, dict)
        self.assertIn("after_snapshot_load", phase_max)

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
