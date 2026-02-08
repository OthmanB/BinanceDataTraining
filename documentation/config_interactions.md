# Configuration Key Interactions

**Date:** 2026-02-08

This document describes configuration keys that have dependencies on or interactions with other keys. Misconfiguring these relationships is a common source of errors.

---

## 1. Train/Validation Split Consistency

The validation ratio must match between preprocessing and training:

```yaml
preprocessing:
  train_test_split:
    validation_ratio: 0.15

training:
  validation_split: 0.15    # MUST equal preprocessing.train_test_split.validation_ratio
```

**Error if mismatched:**
> `ValueError: training.validation_split must match preprocessing.train_test_split.validation_ratio`

---

## 2. Snapshot Mode (Mandatory)

The snapshot pipeline is the only supported training mode:

```yaml
snapshot:
  enabled: true              # MUST be true — legacy in-memory pipeline is disabled
  directory: "./snapshots"
  name: "auto"               # or a specific name
  on_config_mismatch: "create_new"  # or "error"
```

**Error if disabled:**
> `ConfigError: Legacy in-memory training pipeline is disabled. Set snapshot.enabled=true`

---

## 3. Output Type Constraint

Only `two_head_intensity` is supported in snapshot training:

```yaml
model:
  output:
    type: "two_head_intensity"   # MUST be this value
    num_classes: 4               # number of intensity bins
```

**Error if different:**
> `ValueError: Only model.output.type='two_head_intensity' is supported in snapshot training`

---

## 4. Long-Term Features ↔ Model Input Count

When long-term features are enabled, the model must have dual inputs:

```yaml
model:
  long_term:
    enabled: true              # ← enables long-term branch
    # Requires model to be built with dual inputs
```

- If `long_term.enabled=true` but model has only 1 input → `ConfigError`
- If `long_term.enabled=false` but model has 2 inputs → `ConfigError`

---

## 5. Sequential Training Windows

Sequential training requires a valid date range and window configuration:

```yaml
training:
  sequential_training:
    enabled: true
    window_days: 30            # each training window spans this many days
    resume:
      enabled: true
      state_dir: "./resume_state"
    cleanup:
      completed_windows: true
      keep_last_windows: 2
```

The date range in `data.time_range.start_date` / `end_date` is split into windows of `window_days` length. If only 1 window results, sequential mode is bypassed.

---

## 6. Fine-Tuning Dependencies

When fine-tuning is enabled, either a run ID or registry name is required:

```yaml
training:
  fine_tuning:
    enabled: true
    use_model_registry: false
    base_model_run_id: "abc123"       # required if use_model_registry=false
    registry_name: ""                  # required if use_model_registry=true
    base_model_stage: "Production"
    freeze_layers: "cnn"              # none | cnn | cnn_lstm | all_but_output
    learning_rate_factor: 0.1
```

---

## 7. Normalization ↔ Training Split

```yaml
preprocessing:
  normalization:
    method: "min_max"          # min_max | standard | robust
    fit_on_train_only: true    # RECOMMENDED: prevents data leakage
```

When `fit_on_train_only=true`, normalization statistics are computed from training data and applied to validation/test sets. When `false`, each split gets its own statistics (not recommended).

---

## 8. Class Weights ↔ Output Config

```yaml
training:
  class_weights:
    compute_from_train: true   # uses model.output.num_classes

model:
  output:
    num_classes: 4             # must match label range [0, num_classes-1]
```

Class weights are computed per-head (up and down intensity) using inverse-frequency from training labels.

---

## 9. HPO ↔ Runtime Device

```yaml
hyperparameter_optimization:
  enabled: true
  parallel:
    enabled: true
    resources:                 # each resource gets a worker process
      - "gpu:0"
      - "gpu:1"
    storage_uri: "sqlite:///optuna.db"
    max_trials_per_worker_process: 5
    rss_watchdog_enabled: true
    rss_watchdog_max_worker_rss_gb: 28.0

training:
  runtime:
    device: "gpu"              # overridden per-worker in parallel HPO
    gpu_visible_devices: "0"
```

In parallel HPO, each worker gets its own `device` and `gpu_visible_devices` override derived from the `resources` list.

---

## 10. Temporal Features ↔ Input Representation

```yaml
model:
  input_representation:
    temporal_features:
      integration_mode: "concat_channels"  # or "none"
      use_local_features: true
      use_global_features: true

data:
  temporal_features:
    local:
      enabled: true
    global:
      enabled: true
```

When `integration_mode=concat_channels`, the corresponding feature arrays must be populated by preprocessing. Missing arrays cause `ValueError`.

---

## 11. Evaluation Missing Snapshot Strategy

```yaml
evaluation:
  missing_snapshot_strategy: "skip"  # fail | skip | synthetic
```

- `fail`: Raise error if no snapshot features for evaluation
- `skip`: Silently skip evaluation stage
- `synthetic`: Use random inputs (for testing only)

---

## 12. Logging Format

```yaml
logging:
  format: "colored"           # "colored" or "json"
  level: "INFO"
  colors:                     # only used when format="colored"
    info: "green"
    warning: "yellow"
    error: "red"
    debug: "blue"
```

When `format="json"`, log output is structured JSON lines. The `colors` section is still required by the schema but only used for `format="colored"`.

---

## 13. MLflow Artifact Logging

```yaml
mlflow:
  artifact_logging:
    trained_model: true        # log model to MLflow
  model_registry:
    register_model: true       # requires artifact_logging.trained_model=true
    model_name_pattern: "{asset}_{model}"
```

Model registration requires the model to be logged first. If `trained_model=false` but `register_model=true`, the registration step is silently skipped.

---

## 14. Backtesting ↔ Evaluation

```yaml
evaluation:
  backtesting:
    enabled: true
    signal_strategy: "net_intensity"  # or "threshold"
    signal_threshold: 0.6             # only used with "threshold" strategy
    intensity_threshold: 1            # minimum class for action
    initial_capital: 10000.0
    transaction_cost_pct: 0.001
    position_sizing: "equal"          # or "confidence"
```

Backtesting runs after the main evaluation and requires the model to produce two-head intensity predictions.
