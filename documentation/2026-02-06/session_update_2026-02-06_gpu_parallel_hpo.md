# Session Update — 2026-02-06 (GPU Enablement, Sequential Snapshot, Parallel HPO)

## Scope

This note records all changes made since the last commit during the current working session, including configuration, pipeline behavior, tests, and operational verification.

## Major Implementations

### 1) Sequential Snapshot Training/Evaluation Flow

- Added rolling sequential snapshot window support in training pipeline.
- Added optional cleanup after completed windows.
- Added retention option for cleanup (`cleanup_keep_last_windows`).
- Added resumability for interrupted sequential runs (`resume_enabled`) with:
  - persisted resume state (`.json`),
  - persisted model checkpoint (`.keras`),
  - automatic resume from `next_window_index` when window plan matches,
  - cleanup of resume artifacts on successful completion.
- Added sequential snapshot evaluation in `main.py` to avoid one monolithic end-of-run evaluation snapshot pass.

### 2) Snapshot-Compatible HPO

- Removed prior snapshot+HPO hard block in `main.py`.
- Enabled snapshot HPO in both trial/production paths with guardrails.
- Added HPO metric extraction from training history and propagated metric through snapshot paths.
- Implemented weighted aggregation of HPO metrics across sequential windows:
  - `val_*` metrics weighted by validation sample count,
  - non-`val_*` metrics weighted by effective train sample count.

### 3) Runtime GPU Hardening

- Added startup fail-fast check for `training.runtime.device='gpu'`:
  - verifies TensorFlow importability,
  - verifies visible GPU devices.
- If unavailable, raises clear `ConfigError` with remediation hints.

### 4) Reproducible GPU Environment Setup

- Updated Linux Python 3.12 dependency in `requirements.txt` to:
  - `tensorflow[and-cuda]==2.16.2`
- This ensures CUDA runtime/cuDNN wheels are installed in `.venv` consistently.

### 5) Parallel HPO Trials (Process-Based)

- Implemented parallel Optuna trial execution in `models/hyperparameter_tuning.py` using worker processes.
- Added resource pinning per worker:
  - `cpu`
  - `gpu:<id>` (e.g., `gpu:0`, `gpu:1`)
- Added trial distribution logic across workers.
- Added shared-study storage handling with auto SQLite fallback under `mlflow.local_tmp_dir/optuna` when `storage_uri` is null.
- Added optional parallel HPO config block:
  - `hyperparameter_optimization.parallel.enabled`
  - `hyperparameter_optimization.parallel.resources`
  - `hyperparameter_optimization.parallel.storage_uri`
  - `hyperparameter_optimization.parallel.study_name`

## E2E Config Profiles Added/Updated

- `config/e2e_trial_13_snapshot_hpo_trial.yaml`
  - tuned for snapshot trial HPO.
- `config/e2e_trial_14_snapshot_hpo_production.yaml`
  - tuned for snapshot production HPO.
- `config/e2e_trial_15_parallel_hpo_trial.yaml`
  - prepared for multi-resource HPO workers: `gpu:0`, `gpu:1`, `cpu`.
  - executed multiple times on dual RTX 3090 + CPU after hardware update.

## New / Updated Tests

- `tests/test_sequential_training_windows.py`
- `tests/test_runtime_device_validation.py`
- `tests/test_sequential_hpo_metric_aggregation.py`
- `tests/test_main_hpo_snapshot_flows.py`
- `tests/test_sequential_resume.py`
- `tests/test_hpo_parallel.py`
- `tests/test_parallel_hpo_properties.py` (property-based)
- `tests/test_e2e_snapshot_hpo_configs.py` (extended for test 15 config)
- `tests/test_config_loader.py` (optional key type validation coverage)

## Property Tests Added (Hypothesis)

- Allocation invariants for trial-worker distribution.
- Resource parser fuzz validity checks.
- Resume path stability and change sensitivity.
- Sequential window generation coverage/contiguity.
- Weighted metric aggregation boundedness checks.
- Cleanup retention tail-set correctness.

## Runtime Verifications Performed

- Verified TensorFlow GPU visibility in `.venv`.
- Verified GPU matmul smoke test executes on `GPU:0`.
- Executed end-to-end `config/e2e_trial_14_snapshot_hpo_production.yaml` successfully with GPU active.
- Confirmed sequential training + sequential evaluation + HPO complete path and MLflow run completion.

## Files with Key Behavior Changes

- `main.py`
- `training/pipeline.py`
- `models/hyperparameter_tuning.py`
- `utils/config_loader.py`
- `config/validation_schema.yaml`
- `config/training_config.yaml`
- `config/training_config_default.yaml`
- `requirements.txt`

## Addendum — 2026-02-07 (Experiment 15 live runs, failures, and current status)

### Trial Results Observed

- First live run result: `4/6` trials completed, `1` failed, `1` skipped.
- Failure pattern was GPU-side memory pressure/OOM on one worker branch.
- Subsequent reruns with reduced batch ranges still failed in early parallel execution with:
  - `CUDA_ERROR_OUT_OF_MEMORY`,
  - `No DNN in stream executor`,
  - occasional process-pool abrupt termination.

### Config Iterations Performed (Experiment 15)

- Batch-size narrowing sequence tested in `config/e2e_trial_15_parallel_hpo_trial.yaml`:
  - baseline around `[8, 16]`,
  - then `[7, 14]`,
  - then `[6, 12]`,
  - then `[4, 6]` (current).
- Runtime controls were added and enabled for this experiment:
  - `training.runtime.gpu_memory_growth: true`,
  - `training.runtime.gpu_allocator: "cuda_malloc_async"`,
  - `training.runtime.gpu_init_lock_enabled: true`,
  - `training.runtime.gpu_init_stagger_seconds: 1`.

### Code Changes Introduced for Memory Management

- Added worker-side runtime environment enforcement in `models/hyperparameter_tuning.py` so each HPO worker applies GPU/CPU pinning before trial execution.
- Added optional runtime tuning options (schema + base configs):
  - `gpu_memory_growth`,
  - `gpu_allocator`,
  - `gpu_init_lock_enabled`,
  - `gpu_init_stagger_seconds`.
- Added staged GPU init behavior (lock + optional stagger) to reduce concurrent TensorFlow context initialization spikes.
- Added logging for worker resource assignment and trial lifecycle.

### Tests Updated and Passing

- Updated tests for new runtime controls and worker-runtime behavior:
  - `tests/test_hpo_parallel.py`,
  - `tests/test_runtime_device_validation.py`.
- Verified passing subset:

```bash
.venv/bin/python -m unittest tests.test_hpo_parallel tests.test_runtime_device_validation tests.test_config_loader -v
```

### Mistakes / Lessons Learned

- Reducing batch size alone was not sufficient to stabilize parallel GPU workers.
- Root issue appears to include runtime initialization/allocator behavior across concurrent worker processes, not only model-size scaling.
- Parallel branch independence at Optuna level does not remove GPU allocator contention risk at process/runtime level.

### Current Status (handover)

- Experiment 15 currently configured at `batch_size: 4` and search range `[4, 6]` with runtime memory controls enabled.
- A rerun after these runtime-control changes was manually interrupted by operator during live monitoring.
- Operator observation at interruption time: both GPUs were around ~500 MB usage with weak power draw, suggesting under-utilization in that specific in-progress run stage.
- Next action on restart: rerun experiment 15 and inspect early worker logs to confirm per-worker pinning/init flow before full trial progression.

## Addendum — 2026-02-07 (Regime learning, persistence, and architecture stress validation)

### What Was Implemented

- Added a full HPO regime controller in `models/hyperparameter_tuning.py` with:
  - resource-failure classification (`OOM`/CUDA/resource exhaustion pattern detection),
  - adaptive retry with batch backoff,
  - failure handling policy (`prune` or `penalize`),
  - soft objective penalties (VRAM fraction and optional low-utilization penalties),
  - per-trial telemetry persisted in Optuna `user_attrs`.
- Added persistent safe-envelope memory across runs:
  - stored at `tmp/mlflow/optuna/regime_memory_<study_name>.json`,
  - tracks global and architecture-signature outcomes,
  - supports proactive pre-trial batch clamping from historical evidence.
- Added/extended config keys under `hyperparameter_optimization.regime` in:
  - `config/validation_schema.yaml`,
  - `config/training_config.yaml`,
  - `config/training_config_default.yaml`.

### Experiment 15 (Regime-enabled throughput profile)

- `config/e2e_trial_15_parallel_hpo_trial.yaml` moved to a throughput-oriented dual-GPU profile:
  - `resources: ["gpu:0", "gpu:1"]`,
  - `epochs: 2`,
  - runtime allocator set to `default` with `gpu_memory_growth: false`,
  - regime controller enabled with persistent safe-envelope.
- Verified successful full runs with no OOM failures.
- Confirmed telemetry fields in Optuna DB (`metric_effective`, retry counts, effective batch, GPU memory stats).
- Confirmed persistence file creation and updates:
  - `tmp/mlflow/optuna/regime_memory_e2e_trial_15_parallel_hpo.json`.

### Clarification: High VRAM Without OOM

- Observed near-capacity VRAM (`~22/24 GB`) was expected in this mode and mostly allocator reservation behavior from TensorFlow when:
  - `gpu_memory_growth: false`,
  - `gpu_allocator: default`.
- This is separate from regime-learning logic and does not necessarily imply imminent OOM.

### Why Batch Adapted Without Crashes

- Adaptive changes with `regime_retry_count=0` came from proactive safe-envelope clamp (pre-trial), not from retry path.
- With `safe_envelope.headroom_fraction: 0.95`, successful historical batch 24 can be clamped to effective 22 (`floor(24 * 0.95)`).

### New Architecture-Focused Test 16

- Added `config/e2e_trial_16_architecture_regime_trial.yaml` to test regime learning with architecture variability emphasized:
  - fixed batch search (`batch_size: [24, 24]`),
  - widened architecture ranges:
    - `cnn_filters_1: [32, 128]`,
    - `cnn_filters_2: [64, 256]`,
    - `lstm_units: [64, 192]`,
  - dual-GPU parallel HPO and regime persistence enabled.
- Added config-load coverage for trial 16 in `tests/test_e2e_snapshot_hpo_configs.py`.

### Trial 16 Validation Outcome

- Run completed successfully: `10/10` trials complete, `0` pruned, `0` failed.
- No OOM crash encountered.
- Best result observed:
  - `best_value=0.00021599069711647498`,
  - `best_params={'cnn_filters_1': 98, 'cnn_filters_2': 202, 'lstm_units': 73, 'learning_rate': 0.000960456582503971, 'batch_size': 24}`.
- Regime learning confirmed in DB evidence:
  - multiple trials recorded `params.batch_size=24` with `batch_size_effective=22` and `regime_retry_count=0`,
  - demonstrating proactive clamp based on learned envelope without needing crash-triggered retries.
