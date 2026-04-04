# BinanceDataTraining Codebase Audit

**Date:** 2026-02-08  
**Time:** 13:22 UTC+09:00  
**Auditor:** Cascade AI  

---

## Executive Summary

This document provides a comprehensive audit of the BinanceDataTraining machine learning pipeline. The codebase implements a sophisticated system for:
- Fetching order book data from GreptimeDB
- Building snapshot-based training datasets
- Training CNN+LSTM models for price movement prediction
- Evaluating models with calibration, temporal degradation, and backtesting
- Hyperparameter optimization with Optuna
- Observability via HTMX dashboard and Prometheus metrics

**Key Findings:**
- Well-structured, modular architecture with clear separation of concerns
- Strict YAML-first configuration with schema validation
- Robust error handling with fail-fast behavior
- Comprehensive test coverage (38 test files)
- Legacy code paths exist but are disabled (snapshot pipeline is mandatory)

---

## 1. Architecture Overview

### 1.1 Directory Structure

```
BinanceDataTraining/
├── main.py                    # Entry point (411 lines)
├── config/                    # YAML configuration files (23 files)
├── data/                      # Data loading and SQL utilities
├── preprocessing/             # Feature engineering, normalization, transformers
├── training/                  # Training pipeline, snapshot management
├── models/                    # CNN+LSTM architecture, hyperparameter tuning
├── evaluation/                # Model evaluation, calibration, backtesting
├── observability/             # HTMX dashboard, Prometheus metrics, run state
├── diagnostics/               # Data quality checks
├── mlflow_integration/        # MLflow experiment tracking
├── utils/                     # Config loader, logging, env validation
└── tests/                     # Unit and integration tests (38 files)
```

### 1.2 Entry Points

| Entry Point | Description |
|-------------|-------------|
| `main.py` | Primary pipeline entry (training/evaluation) |
| `observability/server.py` | Standalone observability dashboard server |

---

## 2. Configuration System

### 2.1 Configuration Files

| File | Purpose |
|------|---------|
| `config/training_config.yaml` | Full production config (~32KB) |
| `config/training_config_default.yaml` | Simple editor baseline |
| `config/validation_schema.yaml` | Schema with type validation |
| `config/observability.yaml` | Observability server settings |
| `config/e2e_trial_*.yaml` | End-to-end test configs (17 variants) |

### 2.2 Required Configuration Sections

The schema (`config/validation_schema.yaml`) enforces 13 required top-level sections:
1. `data` - Data source and connection settings
2. `targets` - Prediction horizon and labeling
3. `preprocessing` - Normalization, feature engineering, splits
4. `model` - CNN+LSTM architecture configuration
5. `training` - Epochs, batch size, callbacks, fine-tuning
6. `snapshot` - Snapshot dataset management
7. `hyperparameter_optimization` - Optuna HPO settings
8. `mlflow` - Experiment tracking
9. `evaluation` - Metrics, calibration, backtesting
10. `logging` - Colored output configuration
11. `security` - Environment variable validation
12. `diagnostics` - Data quality checks
13. `run_mode` - Production vs trial mode

### 2.3 Configuration Validation

Configuration loading (`utils/config_loader.py`) follows strict patterns:
- **Environment placeholders**: `${VAR}` must resolve or raise `ConfigError`
- **Type validation**: All keys validated against schema types
- **No implicit defaults**: Values must be explicitly set in YAML
- **Base config inheritance**: `base_config` key enables overlay patterns

---

## 3. Execution Flow Analysis

### 3.1 Main Pipeline Flow

```
main.py
  │
  ├─> load_config() + validate schema
  ├─> setup_colored_logging()
  ├─> _apply_runtime_device() (CPU/GPU selection)
  ├─> _validate_runtime_device_availability() (TF GPU check)
  ├─> validate_environment() (env var check)
  ├─> start_run() (MLflow)
  │
  ├─> run_snapshot_diagnostics() (data quality)
  │
  ├─> [HPO enabled?]
  │     ├─> run_hyperparameter_search()
  │     └─> [trial mode?] → return 0
  │
  ├─> run_training_pipeline()
  │     ├─> prepare_snapshot_dataset()
  │     ├─> _resolve_sequential_windows() (if enabled)
  │     ├─> build_training_generator()
  │     ├─> build_cnn_lstm_model() or load fine-tuning base
  │     └─> model.fit()
  │
  ├─> evaluate_snapshot_model()
  │     ├─> Compute metrics (accuracy, precision, recall, F1)
  │     ├─> Calibration analysis
  │     ├─> Temporal degradation analysis
  │     └─> Backtesting
  │
  └─> end_run() (MLflow)
```

### 3.2 Snapshot Pipeline (Primary Path)

The legacy in-memory pipeline is **disabled**. `snapshot.enabled=true` is mandatory.

**Snapshot Dataset Creation (`training/snapshot_dataset.py`):**
1. Stream order book data from GreptimeDB in time chunks
2. Build per-sample tensors (T, H, W, C format)
3. Save to `.npz` chunk files with manifest
4. Support gap handling (forward_fill, interpolate, skip)

**Key Data Structures:**
- `SnapshotDataset`: Manifest + chunk metadata
- `SnapshotRecord`: Per-snapshot depth, mid-price, features
- `SampleRecord`: Final (x, y_up, y_down) training sample
- `NormalizationStats`: min/max, mean/std, or median/IQR

### 3.3 Sequential Training Windows

When `training.sequential_training.enabled=true`:
- Date range is split into `window_days` chunks
- Model trained incrementally across windows
- Resume support via JSON state + .keras checkpoint
- Window cleanup to limit disk usage

### 3.4 Hyperparameter Optimization

**Framework:** Optuna

**Execution Modes:**
1. **Sequential**: Single-process trial execution
2. **Parallel**: Multi-GPU/CPU worker pool with RSS watchdog

**Architecture Regime Features:**
- OOM retry with batch size backoff
- Safe envelope learning from successful trials
- VRAM penalty for exceeding limits
- Low utilization penalty

**Search Space Configuration:**
```yaml
hyperparameter_optimization:
  search_space:
    learning_rate: {type: float, low: 0.0001, high: 0.01, log: true}
    cnn_filters_1: {type: int, low: 16, high: 128, step: 16}
    lstm_units: {type: int, low: 32, high: 256, step: 32}
    # ... more parameters
```

---

## 4. Module Analysis

### 4.1 Data Layer (`data/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `greptime_client.py` | ~1000 | HTTP client for GreptimeDB SQL API |
| `sql_utils.py` | ~250 | Query builders for order book data |
| `data_loader.py` | ~160 | Legacy data loading utilities |
| `data_object.py` | ~90 | DataObject container class |

**Key Features:**
- Time-chunked streaming to limit memory
- Multi-database support with `data.multi_database.enabled`
- Configurable timeouts and retry backoff
- Asset pair alignment with bucket tolerance

### 4.2 Preprocessing Layer (`preprocessing/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `transformer.py` | ~634 | Pipeline orchestration, target building |
| `feature_engineering.py` | ~532 | Order book and momentum features |
| `long_term_features.py` | ~730 | 7/30/90-day market context |
| `normalizer.py` | ~234 | min_max, standard, robust scaling |
| `depth_aggregator.py` | ~400 | Hybrid depth representation |
| `temporal_features.py` | ~490 | Time-of-day, day-of-week encoding |
| `snapshot_sequence_builder.py` | ~330 | Tensor construction for CNN input |

**Supported Features:**
- `bid_ask_spread`, `volume_imbalance`, `depth_imbalance`, `weighted_mid_price`
- `price_momentum`, `volume_momentum` with edge decay
- Long-term context: volatility, returns, volume trends

### 4.3 Training Layer (`training/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `pipeline.py` | ~2050 | Main training orchestration |
| `snapshot_dataset.py` | ~2500 | Streaming dataset/generator builders |
| `snapshot_store.py` | ~280 | Manifest management, eviction |
| `fine_tuning.py` | ~750 | Base model loading, layer freezing |
| `long_term_context.py` | ~360 | Dual-input generator wrapping |
| `class_weights.py` | ~210 | Inverse-frequency weighting |
| `callbacks.py` | ~100 | Early stopping, LR reduction |

**Model Architecture (`models/cnn_lstm_multiclass.py`):**
- TimeDistributed Conv2D layers per temporal slice
- LSTM for sequential aggregation
- Optional long-term branch (dual-input)
- Two-head output: `up_intensity`, `down_intensity`

### 4.4 Evaluation Layer (`evaluation/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `evaluator.py` | ~1900 | Main evaluation pipeline |
| `calibration.py` | ~610 | Temperature scaling, ECE/MCE metrics |
| `temporal_degradation.py` | ~520 | Time-window performance analysis |
| `backtesting.py` | ~700 | Simulated trading with Sharpe, drawdown |

**Evaluation Metrics:**
- Per-class precision, recall, F1
- Macro-averaged metrics
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Sharpe ratio, max drawdown, win rate

### 4.5 Observability Layer (`observability/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `server.py` | ~2200 | HTMX dashboard + Prometheus `/metrics` |
| `run_state.py` | ~350 | SQLite-backed run state tracking |
| `training_progress.py` | ~85 | Keras callback for progress updates |

**Dashboard Features:**
- Real-time training progress
- Log tailing
- System metrics (CPU, memory, GPU)
- Run control (start/stop)
- Basic auth required

### 4.6 MLflow Integration (`mlflow_integration/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `experiment_tracker.py` | ~190 | Run management, config logging |
| `model_registry.py` | ~70 | Model version registration |

**Logged Artifacts:**
- Trained model (`mlflow.tensorflow.log_model`)
- Configuration snapshot
- Training plots, confusion matrix
- Class distribution charts

---

## 5. Test Coverage Analysis

### 5.1 Test Files (38 total)

| Category | Files | Focus |
|----------|-------|-------|
| Config/Validation | 3 | Schema, fail-fast, config loader |
| Data Pipeline | 5 | SQL utils, Greptime client, time chunks |
| Preprocessing | 7 | Features, normalizer, depth aggregator |
| Training | 9 | Pipeline, fine-tuning, sequential, snapshot |
| Evaluation | 5 | Calibration, backtesting, temporal degradation |
| Models | 4 | Dual input, HPO parallel, architecture |
| Integration | 5 | E2E flows, observability |

### 5.2 Testing Frameworks

- **Primary:** `unittest.TestCase`
- **Secondary:** `pytest` for some modules
- **Property-based:** `hypothesis` (optional, skipped in CI)

---

## 6. Code Quality Observations

### 6.1 Strengths

1. **Strict Configuration**
   - No hardcoded defaults in runtime logic
   - Schema validation at load time
   - Environment variable placeholders with mandatory resolution

2. **Fail-Fast Philosophy**
   - `ConfigError` raised on missing/invalid config
   - Early validation of data shapes and types
   - Clear error messages with context

3. **Modular Design**
   - Clear separation: data → preprocessing → training → evaluation
   - Each module has well-defined `__all__` exports
   - Lazy imports for heavy dependencies (TensorFlow)

4. **Logging Quality**
   - Structured logging with colored output
   - Function names and line numbers included
   - Appropriate log levels throughout

5. **Robustness Features**
   - Retry logic for network operations
   - Snapshot resume for long training runs
   - OOM recovery in HPO with batch backoff

### 6.2 Areas for Consideration

1. **Legacy Code Presence**
   - `training/pipeline.py` lines 1286-1399 contain unreachable legacy code
   - The code after `raise ConfigError(...)` on line 1286 will never execute
   - Consider removing to reduce maintenance burden

2. **Large File Sizes**
   - `training/snapshot_dataset.py`: 2500 lines
   - `training/pipeline.py`: 2050 lines
   - `observability/server.py`: 2200 lines
   - Consider breaking into smaller focused modules

3. **Exception Handling Patterns**
   - Many `except Exception as exc: # noqa: BLE001` blocks
   - Consistent but could benefit from more specific exception types in some cases

4. **Documentation**
   - Module docstrings present and informative
   - Some complex functions could benefit from more detailed inline comments

---

## 7. Execution Path Matrix

### 7.1 Mode Combinations

| run_mode | HPO enabled | Sequential | Result |
|----------|-------------|------------|--------|
| production | false | false | Single training + eval |
| production | false | true | Multi-window training + eval |
| production | true | false | HPO → final training + eval |
| production | true | true | HPO with sequential windows → final |
| trial | true | true | HPO only, no final training |
| trial | true | false | Error (requires sequential windows) |

### 7.2 Data Flow States

```
GreptimeDB ─────────────────────────────────┐
     │                                       │
     ▼                                       │
stream_order_book_chunks_by_time()          │
     │                                       │
     ▼                                       │
GapHandler.iter_gap_handled()               │
     │                                       │
     ▼                                       │
Snapshot Chunks (.npz files) ◄──────────────┘
     │                                     (cached)
     ▼
prepare_snapshot_dataset()
     │
     ▼
build_training_generator() ─────► model.fit()
     │                                  │
     ▼                                  ▼
NormalizationStats              MLflow logging
     │
     ▼
evaluate_snapshot_model()
     │
     ├─► Calibration analysis
     ├─► Temporal degradation
     └─► Backtesting
```

---

## 8. Dependencies

### 8.1 Core Dependencies (from requirements.txt)

- **TensorFlow**: Deep learning framework
- **NumPy**: Numerical operations
- **Requests**: HTTP client for GreptimeDB
- **PyYAML**: Configuration parsing
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization
- **Prometheus-client**: Metrics endpoint
- **termcolor**: Colored logging output

### 8.2 Optional Dependencies

- **matplotlib**: Visualization/plots
- **hypothesis**: Property-based testing

---

## 9. Security Considerations

### 9.1 Secrets Management

- API keys and credentials via environment variables only
- `${VAR}` placeholders in YAML resolve from env
- `security.validation.check_env_vars_at_startup` enforces presence
- No secrets logged (validation in place)

### 9.2 Observability Server

- Basic Auth required (`OBSERVABILITY_USER`, `OBSERVABILITY_PASSWORD`)
- `allow_run_control` must be explicitly enabled
- Config file path restrictions via glob pattern

---

## 10. Recommendations

### 10.1 Code Cleanup

1. ~~**Remove dead code** in `training/pipeline.py`~~ — **DONE** (2026-02-08): Removed ~760 lines of unreachable legacy in-memory training pipeline code. File reduced from 2051 to 1292 lines.
2. **Split large modules** for better maintainability — *Deferred: high-risk refactoring, needs careful test coverage first*
3. **Add type stubs** for TensorFlow imports where missing — *Deferred*

### 10.2 Documentation

1. ~~Add architecture diagrams~~ — **DONE** (2026-02-08): Created `documentation/2026-02-08/architecture.md` with pipeline flow, data flow, module dependency graph, model architecture, and observability server endpoint map.
2. ~~Document config key interactions~~ — **DONE** (2026-02-08): Created `documentation/2026-02-08/config_interactions.md` documenting 14 key interaction patterns.
3. ~~Create troubleshooting guide~~ — **DONE** (2026-02-08): Created `documentation/2026-02-08/troubleshooting.md` covering configuration, data loading, training, evaluation, HPO, observability, MLflow, and environment setup issues.

### 10.3 Testing

1. Add integration tests for multi-database mode — *Pending*
2. Add load tests for observability server — *Pending*
3. Consider adding mutation testing — *Pending*

### 10.4 Operational

1. ~~Add health check endpoint to observability server~~ — **DONE** (2026-02-08): Added `GET /healthz` endpoint (no auth required) returning JSON `{status, run_status, run_state_stale}`.
2. ~~Add structured JSON logging option for production~~ — **DONE** (2026-02-08): Added `StructuredJsonFormatter` to `utils/colored_logging.py`, new `logging.format` config key (`"colored"` or `"json"`), updated schema and both YAML configs.
3. Add Grafana dashboard templates — *Pending*

---

## Appendix A: Key Configuration Paths

```yaml
# Core training settings
training.epochs
training.batch_size
training.debug_max_samples
training.runtime.device (cpu|gpu)

# Model architecture
model.architecture
model.cnn.filters
model.lstm.units
model.output.num_classes

# Data settings
data.connection.database_uri
data.asset_pairs.target_asset
data.time_range.start_date
data.time_range.end_date

# Snapshot settings
snapshot.enabled (must be true)
snapshot.directory
snapshot.name (or "auto")

# HPO settings
hyperparameter_optimization.enabled
hyperparameter_optimization.n_trials
hyperparameter_optimization.parallel.enabled
```

---

## Appendix B: Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OBSERVABILITY_USER` | Dashboard auth | For observability |
| `OBSERVABILITY_PASSWORD` | Dashboard auth | For observability |
| `RUN_STATE_PATH` | SQLite state file | For observability |
| `RUN_LOG_PATH` | Log file path | Optional |
| `CUDA_VISIBLE_DEVICES` | GPU selection | Set by runtime config |
| `TF_GPU_ALLOCATOR` | Memory allocator | Optional |

---

*End of Audit Document*
