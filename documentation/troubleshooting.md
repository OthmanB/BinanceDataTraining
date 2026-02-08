# Troubleshooting Guide

**Date:** 2026-02-08

---

## 1. Configuration Errors

### `ConfigError: Missing required environment variables: ...`

**Cause:** `security.validation.check_env_vars_at_startup` is `true` and one or more environment variables listed in `security.environment_variables` are not set.

**Fix:**
```bash
export DATABASE_URI="https://your-greptimedb-host:4000"
export MLFLOW_TRACKING_TOKEN="your-token"
# etc.
```

Or disable startup validation (not recommended for production):
```yaml
security:
  validation:
    check_env_vars_at_startup: false
```

---

### `ConfigError: Legacy in-memory training pipeline is disabled`

**Cause:** `snapshot.enabled` is `false`. The legacy in-memory pipeline has been removed.

**Fix:** Set `snapshot.enabled: true` in your config YAML.

---

### `ConfigError: Snapshot configuration hash mismatch`

**Cause:** The config hash for your current run differs from the existing snapshot manifest. This happens when you change data, preprocessing, or model parameters but point to the same snapshot directory.

**Fix:**
- Set `snapshot.on_config_mismatch: "create_new"` to auto-create a new snapshot
- Or delete the old snapshot directory manually

---

### `ValueError: training.validation_split must match preprocessing.train_test_split.validation_ratio`

**Cause:** The validation ratio is set differently in two config sections.

**Fix:** Ensure both values are identical:
```yaml
preprocessing:
  train_test_split:
    validation_ratio: 0.15

training:
  validation_split: 0.15
```

---

## 2. Data Loading Errors

### `ConnectionError` or timeout when connecting to GreptimeDB

**Cause:** The database URI is unreachable, or timeouts are too short.

**Fix:**
1. Verify `data.connection.database_uri` resolves correctly
2. Increase timeouts:
   ```yaml
   data:
     connection:
       request_timeout_seconds: 60
       connect_timeout_seconds: 20
       max_retries: 5
   ```
3. Test connectivity:
   ```bash
   curl -s "${DATABASE_URI}/health"
   ```

---

### `ValueError: Snapshot input tensors must have rank 5`

**Cause:** The stored `.npz` chunk files have unexpected tensor shapes. This can happen if the snapshot was created with a different config.

**Fix:**
1. Delete the snapshot directory
2. Re-run with correct configuration
3. The pipeline will rebuild the snapshot from GreptimeDB

---

## 3. Training Errors

### `ResourceExhaustedError` / `OOM` during training

**Cause:** GPU memory exhausted. Common with large batch sizes or model configurations.

**Fix:**
- Reduce `training.batch_size`
- Enable HPO regime safeguards:
  ```yaml
  hyperparameter_optimization:
    regime:
      batch_backoff_enabled: true
      batch_backoff_factor: 0.7
      min_batch_size: 4
  ```
- Use `training.runtime.gpu_memory_growth: true` in config

---

### `ConfigError: Long-term features enabled but computation returned None`

**Cause:** Long-term feature computation failed. Usually because the snapshot dataset doesn't have enough temporal data for the required lookback windows (7, 30, 90 days).

**Fix:**
- Ensure your date range covers enough history
- Or disable long-term features:
  ```yaml
  model:
    long_term:
      enabled: false
  ```

---

### `ConfigError: Failed to load base model for fine-tuning`

**Cause:** The MLflow run ID or registry name for fine-tuning doesn't exist or can't be loaded.

**Fix:**
1. Verify the `training.fine_tuning.base_model_run_id` exists in MLflow
2. Check MLflow tracking URI is correct and accessible
3. Verify the model was logged with `mlflow.tensorflow.log_model`

---

## 4. Evaluation Errors

### `ValueError: No snapshot_features available for evaluation inputs`

**Cause:** Evaluation requires snapshot features but none were found.

**Fix:** Set the strategy in config:
```yaml
evaluation:
  missing_snapshot_strategy: "skip"   # skip evaluation gracefully
```

---

### Model prediction returns unexpected shape

**Cause:** Model was trained with a different architecture or num_classes than the current config.

**Fix:** Ensure `model.output.num_classes` matches the model that was trained. If using fine-tuning, the base model architecture must match.

---

## 5. HPO Errors

### `ValueError: hyperparameter_optimization.parallel.resources must contain valid entries`

**Cause:** Parallel HPO resources are misconfigured.

**Fix:**
```yaml
hyperparameter_optimization:
  parallel:
    enabled: true
    resources:
      - "gpu:0"        # format: "gpu:<id>" or "cpu"
      - "gpu:1"
```

---

### HPO trials all failing with OOM

**Cause:** Search space allows batch sizes or model sizes that exceed GPU memory.

**Fix:**
1. Constrain the search space:
   ```yaml
   hyperparameter_optimization:
     search_space:
       batch_size: {type: int, low: 4, high: 32, step: 4}
   ```
2. Enable regime memory (learns safe batch sizes across trials):
   ```yaml
   hyperparameter_optimization:
     regime:
       memory_path: "./hpo_regime_memory.json"
   ```

---

## 6. Observability Server

### Dashboard shows "No active run"

**Cause:** The run state SQLite file is not being written by the training process.

**Fix:**
1. Set `RUN_STATE_PATH` environment variable for both training and server:
   ```bash
   export RUN_STATE_PATH="/tmp/run_state.db"
   ```
2. Ensure the training process has write permissions to that path

---

### 401 Unauthorized on dashboard

**Cause:** Basic auth credentials are missing or incorrect.

**Fix:** Set credentials via environment variables:
```bash
export OBSERVABILITY_USER="admin"
export OBSERVABILITY_PASSWORD="your-password"
```

---

### Health check endpoint

The `/healthz` endpoint does not require authentication:
```bash
curl http://localhost:8080/healthz
# Returns: {"status": "ok", "run_status": "idle", "run_state_stale": false}
```

---

## 7. MLflow Issues

### `Failed to import MLFlow for ...`

**Cause:** MLflow is not installed or not importable.

**Fix:**
```bash
pip install mlflow
```

MLflow is lazily imported — the pipeline will continue without it but won't log metrics or artifacts. Warning messages are emitted for each failed import.

---

### `Failed to log configuration snapshot artifact to MLFlow`

**Cause:** MLflow tracking server is unreachable or artifact storage is misconfigured.

**Fix:**
1. Verify `mlflow.tracking_uri` is correct
2. Ensure the MLflow server is running
3. Check `mlflow.local_tmp_dir` is writable

---

## 8. Common Environment Setup Issues

### `ModuleNotFoundError: No module named 'pyarrow'`

**Fix:** Install with binary-only flag:
```bash
pip install --only-binary=pyarrow pyarrow
```

---

### Python version incompatibility

**Requirement:** Python >=3.9 and <3.12

**Check:**
```bash
python3 --version
```

---

### Virtual environment not activated

**Symptoms:** ImportError for project dependencies.

**Fix:**
```bash
bash setup_venv.sh
source .venv/bin/activate
```
