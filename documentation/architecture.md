# Architecture Overview

**Date:** 2026-02-08

---

## Pipeline Flow

```
                          ┌─────────────────────┐
                          │      main.py         │
                          │   (entry point)      │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            load_config()    setup_logging()    validate_env()
            (YAML + schema)  (colored/json)    (env vars)
                    │
                    ▼
           ┌────────────────┐
           │  MLflow Run    │
           │  start_run()   │
           └───────┬────────┘
                   │
         ┌─────────┼──────────────────┐
         ▼                            ▼
  ┌──────────────┐          ┌─────────────────┐
  │ Diagnostics  │          │  HPO enabled?   │
  │ (optional)   │          └────────┬────────┘
  └──────────────┘               yes │ no
                          ┌──────────┴──────────┐
                          ▼                     ▼
                 ┌─────────────────┐   ┌────────────────┐
                 │ run_hpo_search  │   │  Training       │
                 │ (Optuna)        │   │  Pipeline       │
                 └────────┬────────┘   └───────┬────────┘
                          │                    │
                          ▼                    ▼
                 ┌─────────────────┐   ┌────────────────┐
                 │ trial mode?     │   │ Evaluation      │
                 │ yes → exit      │   │ Pipeline        │
                 │ no → training   │   └───────┬────────┘
                 └─────────────────┘           │
                                               ▼
                                      ┌────────────────┐
                                      │  MLflow Run    │
                                      │  end_run()     │
                                      └────────────────┘
```

## Data Flow

```
GreptimeDB (HTTP SQL API)
       │
       ▼
stream_order_book_chunks_by_time()     ← data/greptime_client.py
       │
       ▼
GapHandler.iter_gap_handled()          ← training/snapshot_dataset.py
       │
       ▼
Snapshot Chunks (.npz files)           ← training/snapshot_store.py
       │                                  (manifest.json tracks state)
       ▼
prepare_snapshot_dataset()             ← training/snapshot_dataset.py
       │
       ├──► NormalizationStats         ← preprocessing/normalizer.py
       │       (fit on train only)
       │
       ├──► FeatureEngineer            ← preprocessing/feature_engineering.py
       │       (order book + momentum)
       │
       ├──► Long-Term Features         ← training/long_term_context.py
       │       (7/30/90-day context)
       │
       ▼
build_training_generator()             ← training/snapshot_dataset.py
       │
       ▼
model.fit()                            ← models/cnn_lstm_multiclass.py
       │                                  (CNN+LSTM dual-input)
       ▼
evaluate_snapshot_model()              ← evaluation/evaluator.py
       │
       ├──► Calibration Analysis       ← evaluation/calibration.py
       ├──► Temporal Degradation       ← evaluation/temporal_degradation.py
       └──► Backtesting                ← evaluation/backtesting.py
```

## Module Dependency Graph

```
main.py
  ├── utils/config_loader.py        (YAML loading, schema validation)
  ├── utils/colored_logging.py      (colored or JSON logging)
  ├── utils/env_validator.py        (env var checks)
  ├── mlflow_integration/
  │     ├── experiment_tracker.py   (run lifecycle)
  │     └── model_registry.py      (model registration)
  ├── diagnostics/
  │     └── snapshot_diagnostics.py (data quality checks)
  ├── models/
  │     ├── cnn_lstm_multiclass.py  (model architecture)
  │     └── hyperparameter_tuning.py (Optuna HPO)
  ├── training/
  │     ├── pipeline.py             (orchestration)
  │     ├── snapshot_dataset.py     (streaming data)
  │     ├── snapshot_store.py       (manifest, eviction)
  │     ├── fine_tuning.py          (layer freezing, LR adjustment)
  │     ├── long_term_context.py    (dual-input wrapping)
  │     ├── class_weights.py        (inverse-frequency weights)
  │     └── callbacks.py            (early stopping, LR reduction)
  ├── preprocessing/
  │     ├── transformer.py          (pipeline entry)
  │     ├── feature_engineering.py  (derived features)
  │     ├── normalizer.py           (min_max, standard, robust)
  │     ├── depth_aggregator.py     (hybrid depth bins)
  │     ├── temporal_features.py    (time encodings)
  │     ├── long_term_features.py   (multi-horizon context)
  │     └── snapshot_sequence_builder.py (tensor construction)
  └── evaluation/
        ├── evaluator.py            (metric computation)
        ├── calibration.py          (temperature scaling, ECE)
        ├── temporal_degradation.py (window analysis)
        └── backtesting.py          (simulated trading)

observability/server.py  (standalone)
  ├── observability/run_state.py    (SQLite state tracking)
  └── observability/training_progress.py (Keras callback)
```

## Model Architecture

```
Short-Term Input                  Long-Term Input (optional)
(T, H, W, C)                     (long_term_dim,)
     │                                   │
     ▼                                   ▼
TimeDistributed(Conv2D)              Dense(64)
     │                                   │
TimeDistributed(MaxPool2D)           Dense(32)
     │                                   │
TimeDistributed(Conv2D)                  │
     │                                   │
TimeDistributed(MaxPool2D)               │
     │                                   │
TimeDistributed(Flatten)                 │
     │                                   │
     ▼                                   │
   LSTM                                  │
     │                                   │
     └──────────┬────────────────────────┘
                │ (concatenate)
                ▼
            Dense(128)
                │
         ┌──────┴──────┐
         ▼             ▼
   up_intensity   down_intensity
   (softmax)      (softmax)
   num_classes    num_classes
```

---

## Observability Server

```
observability/server.py
     │
     ├── GET  /healthz          → JSON health status (no auth)
     ├── GET  /                  → HTMX Dashboard
     ├── GET  /metrics           → Prometheus metrics
     ├── GET  /api/run           → JSON run state
     ├── GET  /api/logs          → JSON log tail
     ├── GET  /ui/*              → HTMX fragments
     ├── POST /ui/start          → Start training run
     ├── POST /ui/stop           → Stop training run
     └── POST /ui/config/*       → Config editor actions
```
