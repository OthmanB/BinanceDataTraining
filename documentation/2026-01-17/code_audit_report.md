# BinanceDataTraining Code Audit Report

**Date:** 2026-01-17 08:00 UTC+09:00  
**Auditor:** Cascade AI  
**Scope:** Full codebase audit against user global coding rules + CNN+LSTM training goal assessment

This audit reflects the state as of 2026-01-17. The codebase has since converged on a snapshot-first pipeline, and an implementation-aligned vision update is available at documentation/2025-11-03/vision_update_2026-02-02.md. The aligned priority and technical debt updates are in documentation/2025-11-03/implementation_priority_matrix_update_2026-02-02.md and documentation/2025-11-03/technical_debt_register_update_2026-02-02.md.

---

## Executive Summary

The BinanceDataTraining project is a well-structured ML training platform for predicting price movements using order book data from Binance. The codebase demonstrates **strong adherence** to many global coding rules, particularly around YAML-first configuration and modular design. However, several **critical gaps** exist that prevent the stated goal (training accurate CNN+LSTM models on order book timeseries) from being fully achieved.

**Overall Assessment:** The project is approximately **70% complete** for its stated goal. Core infrastructure is solid, but key components for production-grade model training are missing or incomplete.

---

## Table of Contents

1. [Codemap Overview](#1-codemap-overview)
2. [Global Rules Compliance Audit](#2-global-rules-compliance-audit)
3. [CNN+LSTM Training Pipeline Assessment](#3-cnnlstm-training-pipeline-assessment)
4. [Critical Gaps and Missing Components](#4-critical-gaps-and-missing-components)
5. [Recommendations](#5-recommendations)
6. [Detailed Findings by Module](#6-detailed-findings-by-module)

---

## 1. Codemap Overview

### Project Structure

```
BinanceDataTraining/
├── main.py                      # Entry point, orchestrates pipeline phases
├── config/
│   ├── training_config.yaml     # Primary configuration (382 lines)
│   └── validation_schema.yaml   # Schema validation definitions
├── data/
│   ├── data_loader.py           # Order book data loading
│   ├── data_object.py           # DataObject structure definitions
│   ├── greptime_client.py       # GreptimeDB HTTP client
│   └── external_sources.py      # Stub for external data
├── preprocessing/
│   ├── transformer.py           # Label construction, preprocessing pipeline
│   ├── temporal_features.py     # Cyclical time encodings
│   ├── snapshot_sequence_builder.py # Tensor construction
│   ├── train_test_split.py      # Chronological splitting
│   ├── validator.py             # DataObject validation
│   └── time_utils.py            # Timestamp normalization
├── models/
│   ├── cnn_lstm_multiclass.py   # CNN+LSTM model builder
│   ├── hyperparameter_tuning.py # Optuna HPO integration
│   └── layers.py                # Custom layers (stub)
├── training/
│   ├── pipeline.py              # Training execution (734 lines)
│   ├── callbacks.py             # Keras callbacks
│   ├── dataset_cache.py         # NPZ caching
│   └── metrics.py               # Metrics utilities
├── evaluation/
│   ├── evaluator.py             # Model evaluation
│   ├── calibration.py           # Calibration metrics
│   └── visualization.py         # Plotting utilities
├── diagnostics/
│   └── data_diagnostics.py      # Pre-training data quality checks
├── mlflow_integration/
│   ├── experiment_tracker.py    # MLFlow run management
│   ├── artifact_manager.py      # Artifact logging
│   └── model_registry.py        # Model registration
├── utils/
│   ├── config_loader.py         # YAML loading + validation
│   ├── colored_logging.py       # Structured colored logging
│   └── env_validator.py         # Environment variable checks
└── tests/                       # 6 test files
```

### Key Data Flow

```
GreptimeDB → data_loader → DataObject → transformer → temporal_features 
    → snapshot_sequence_builder → training/pipeline → model.fit() → evaluator
```

---

## 2. Global Rules Compliance Audit

### 2.1 Configuration (YAML-first, Strict, Validated) ✅ MOSTLY COMPLIANT

| Rule | Status | Evidence |
|------|--------|----------|
| No hardcoded parameters | ✅ PASS | All tunables externalized to `training_config.yaml` |
| No implicit defaults in runtime | ⚠️ PARTIAL | Most params validated; some fallbacks exist |
| Schema validation | ✅ PASS | `validation_schema.yaml` with type checking |
| Early validation | ✅ PASS | `config_loader.py` validates at startup |
| Single source of truth | ✅ PASS | YAML is authoritative |

**Issues Found:**

1. **Implicit fallbacks in code** (violates Rule 2.2):
   - `@config/training_config.yaml:62-65`: Default pattern `"{asset}_{model}_{timestamp}"` embedded in code
   - `@utils/colored_logging.py:35`: Default level `"INFO"` if not in config
   - `@training/pipeline.py:386`: `sw_cfg = None` fallback pattern

2. **Missing validation for some constraints**:
   - No range validation for `train_ratio + validation_ratio + test_ratio = 1.0` in schema
   - No validation that `num_classes == len(boundaries) + 1`

### 2.2 Security ✅ COMPLIANT

| Rule | Status | Evidence |
|------|--------|----------|
| No secrets in code | ✅ PASS | All credentials via `${ENV_VAR}` placeholders |
| Environment variable validation | ✅ PASS | `env_validator.py` checks at startup |
| No logging of secrets | ✅ PASS | No credential logging observed |

**Configuration properly handles:**
- `DATABASE_URI`, `DATABASE_URI_HIST`, `DATABASE_URI_LIVE`
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`

### 2.3 Logging ✅ MOSTLY COMPLIANT

| Rule | Status | Evidence |
|------|--------|----------|
| No `print` for logs | ✅ PASS | All modules use `logging.getLogger(__name__)` |
| Configurable logging | ✅ PASS | Level, colors configurable via YAML |
| Useful context | ✅ PASS | Function names, timestamps, line numbers included |
| Color for readability | ✅ PASS | `ColoredFormatter` in `colored_logging.py` |

**Excellent Implementation:**
```python
# @utils/colored_logging.py:49
fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
```

### 2.4 Architecture & Code Organization ✅ COMPLIANT

| Rule | Status | Evidence |
|------|--------|----------|
| Modular design | ✅ PASS | Clear separation: data/preprocessing/models/training/evaluation |
| File granularity | ✅ PASS | Each major component in separate file |
| Remove dead legacy | N/A | Codebase appears fresh, no obvious legacy |

### 2.5 Terminal / Command Execution Safety ⚠️ NOT APPLICABLE

The codebase does not execute terminal commands; uses HTTP requests to GreptimeDB instead.

**Issue Found:**
- `@data/greptime_client.py:91-99`: HTTP requests have no timeout configured

```python
# Missing timeout parameter
resp = requests.post(
    url,
    data={"sql": sql},
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)  # Should add timeout=30 or similar
```

---

## 3. CNN+LSTM Training Pipeline Assessment

### 3.1 Model Architecture ✅ IMPLEMENTED

The CNN+LSTM model in `@models/cnn_lstm_multiclass.py` is properly structured:

```
Input (T, H, W, C) → TimeDistributed(Conv2D) → TimeDistributed(MaxPool) 
    → TimeDistributed(Flatten) → LSTM → Dense → Two-Head Softmax Output
```

**Strengths:**
- TimeDistributed wrappers for proper temporal processing
- Configurable layer counts, filters, kernel sizes via YAML
- Two-head intensity output for up/down predictions
- Proper dropout and activation handling

**Weaknesses:**
- No residual connections or attention mechanisms
- No batch normalization layers
- Input tensor construction is simplistic (only top-of-book 2x2 patch used)

### 3.2 Data Ingestion ⚠️ PARTIALLY IMPLEMENTED

**What Works:**
- GreptimeDB HTTP SQL API integration
- Multi-database time-split strategy
- Basic bid/ask > 0 filtering on DB side (matches hybrid filtering rule)

**What's Missing:**
- **No depth levels beyond top-of-book**: Configuration specifies `depth_levels: 1000`, but `snapshot_sequence_builder.py` only uses top 4 values (best bid/ask price/qty)
- **No correlated assets used**: Configuration lists 6 correlated assets, but only target asset is processed
- **No chunked ingestion**: Despite `chunk_hours: 12` config, queries fetch entire time range at once

### 3.3 Feature Engineering ⚠️ INCOMPLETE

**Implemented:**
- Temporal features: hour_of_day, day_of_week, minute_of_hour (cyclical encoding)
- Global features: days_since_start, market_session (one-hot)
- Basic spread/mid-price computation

**Not Implemented (but configured):**
- `bid_ask_spread` feature
- `volume_imbalance` feature  
- `depth_imbalance` feature
- `weighted_mid_price` feature
- `price_momentum` feature
- `volume_momentum` feature

These are listed in `preprocessing.feature_engineering` but have no implementation.

### 3.4 Label Construction ✅ IMPLEMENTED

Two-head intensity labeling works correctly:
- Computes percentage price changes over prediction horizon
- Maps to discrete classes based on boundaries [1.0, 2.0, 5.0]%
- Separate up/down intensity heads

### 3.5 Training Loop ✅ FUNCTIONAL

- Chronological train/val/test split
- Class weight support (but not auto-computed from imbalance)
- Sample weighting with exponential decay
- EarlyStopping and ReduceLROnPlateau callbacks
- MLFlow metric logging per epoch
- Dataset caching to NPZ

### 3.6 Evaluation ✅ IMPLEMENTED

- Accuracy, precision, recall, F1 per class
- Confusion matrices
- Calibration analysis (Brier score, ECE)
- Artifact logging to MLFlow

---

## 4. Critical Gaps and Missing Components

### 4.1 CRITICAL: Order Book Depth Not Utilized

**Impact:** The model cannot learn order book microstructure patterns.

**Evidence:**
```python
# @preprocessing/snapshot_sequence_builder.py:117-123
# Only 4 values used despite depth_levels: 1000 config
x_seq[s_idx, tau, 0, 0, 0] = bid_price_f
x_seq[s_idx, tau, 0, 1, 0] = bid_qty_f
x_seq[s_idx, tau, 1, 0, 0] = ask_price_f
x_seq[s_idx, tau, 1, 1, 0] = ask_qty_f
```

**Required:** Implement full order book depth tensor construction with all price levels.

### 4.2 CRITICAL: Correlated Assets Not Used

**Impact:** Multi-asset information theoretic advantages lost.

**Evidence:** Configuration defines 6 correlated assets, but `training/pipeline.py` only processes `target_asset`.

**Required:** Implement multi-asset input tensor construction or remove from config.

### 4.3 CRITICAL: Feature Engineering Not Implemented

**Impact:** Model receives raw prices only, missing derived features.

**Evidence:** `preprocessing.feature_engineering.order_book_features` and `derived_features` configs have no corresponding implementation.

**Required:** Implement feature computation for:
- bid_ask_spread
- volume_imbalance
- depth_imbalance
- weighted_mid_price
- price_momentum
- volume_momentum

### 4.4 HIGH: No Normalization Applied

**Impact:** Training instability due to unnormalized inputs.

**Evidence:** `preprocessing.normalization` config exists but no normalization code in `transformer.py`.

**Required:** Implement min-max/standard/robust scaling with train-only fitting.

### 4.5 HIGH: Class Imbalance Not Addressed

**Impact:** Model likely biased toward majority class.

**Evidence:** `class_weights.compute_from_train: true` but no automatic weight computation.

**Required:** Implement sklearn-style class weight computation.

### 4.6 HIGH: Missing Request Timeouts

**Impact:** Pipeline can hang indefinitely on network issues.

**Evidence:** All `requests.post()` calls in `greptime_client.py` lack timeout parameter.

**Required:** Add `timeout=` parameter to all HTTP calls.

### 4.7 MEDIUM: Backtesting Not Implemented

**Evidence:** `evaluation.backtesting.enabled: false` with placeholder config but no implementation.

### 4.8 MEDIUM: Fine-tuning Not Implemented

**Evidence:** `training.fine_tuning` config exists but `base_model_run_id` loading not implemented.

### 4.9 MEDIUM: External Data Sources Stub Only

**Evidence:** `external_sources.py` returns empty dict unconditionally.

### 4.10 LOW: Custom Layers Stub Only

**Evidence:** `models/layers.py` returns empty dict.

---

## 5. Recommendations

### Priority 1: Critical (Required for Functional Training)

1. **Implement Full Order Book Tensor Construction**
   - Modify `snapshot_sequence_builder.py` to use configured `depth_levels`
   - Create proper (T, H, W, C) tensor where H represents price levels

2. **Implement Input Normalization**
   - Add `Normalizer` class to preprocessing
   - Fit on training data only, apply to val/test
   - Support min-max, standard, robust methods

3. **Add HTTP Request Timeouts**
   - Add `timeout=30` to all `requests.post()` calls
   - Add retry logic with exponential backoff

### Priority 2: High (Required for Accurate Models)

4. **Implement Feature Engineering Pipeline**
   - Create `feature_engineering.py` module
   - Implement all configured features
   - Integrate into preprocessing pipeline

5. **Implement Class Weight Computation**
   - Auto-compute inverse frequency weights from training labels
   - Pass to model.fit() via class_weight parameter

6. **Implement Chunked Data Ingestion**
   - Use `chunk_hours` config to fetch data in batches
   - Aggregate incrementally to manage memory

### Priority 3: Medium (Required for Production)

7. **Implement Multi-Asset Input Processing**
   - Either implement correlated asset integration or remove from config
   - If keeping, define multi-channel input strategy

8. **Implement Fine-tuning Support**
   - Load model from MLFlow by run_id
   - Freeze/unfreeze layers based on config

9. **Implement Backtesting Framework**
   - Add position simulation logic
   - Compute Sharpe, max drawdown, win rate metrics

### Priority 4: Low (Nice to Have)

10. **Add Model Architecture Improvements**
    - Consider attention mechanisms
    - Add batch normalization
    - Consider transformer architectures

---

## 6. Detailed Findings by Module

### 6.1 `config/training_config.yaml`

**Strengths:**
- Comprehensive 382-line configuration
- Well-documented with inline comments
- Covers all pipeline stages

**Issues:**
- Lines 69-70: `start_date` and `end_date` are identical (`"2024-01-01"`), which would load no data
- Line 78: `representation: "hybrid"` has no corresponding hybrid implementation
- Lines 134-143: Feature engineering config has no implementation

### 6.2 `data/greptime_client.py`

**Strengths:**
- Clean HTTP client implementation
- Multi-database support
- Time range filtering on DB side

**Issues:**
- Line 92-96, 272-276: No timeout on requests
- Line 254-259: SQL injection risk (unescaped date literals)

```python
# Current (vulnerable):
sql = f"... WHERE {ts_col} >= '{start_ts_literal}' ..."

# Should use parameterized queries or proper escaping
```

### 6.3 `preprocessing/transformer.py`

**Strengths:**
- Complete label construction logic
- Gap handling (forward_fill, interpolate)
- Validation integration

**Issues:**
- No normalization implementation
- No feature engineering calls
- 537 lines in single function could be split

### 6.4 `training/pipeline.py`

**Strengths:**
- Comprehensive 734-line training orchestration
- Sample weighting implementation
- MLFlow integration throughout

**Issues:**
- Lines 148-228: Synthetic fallback masks real data issues
- Temporal feature integration duplicates evaluation code
- No GPU memory management

### 6.5 `models/cnn_lstm_multiclass.py`

**Strengths:**
- Clean Keras model construction
- Proper TimeDistributed usage
- Configurable architecture

**Issues:**
- No model summary logging
- No input shape inference from config
- Hardcoded "two_head_intensity" assumption

### 6.6 `tests/`

**Coverage Assessment:**
- `test_config_loader.py`: Basic config loading ✅
- `test_evaluator.py`: Evaluation with synthetic data ✅
- `test_greptime_client.py`: DB connectivity ✅
- `test_temporal_features.py`: Feature construction ✅
- `test_train_test_split.py`: Split logic ✅
- `test_training_pipeline_sample_weighting.py`: Sample weights ✅

**Missing Tests:**
- No tests for `snapshot_sequence_builder.py`
- No tests for `transformer.py` label construction
- No integration tests for full pipeline
- No tests for `cnn_lstm_multiclass.py` model building

---

## Appendix A: Hybrid Filtering Architecture Compliance

Per the retrieved memory about hybrid filtering:

> "Always perform base/coarse filtering (e.g., time range, symbol, simple bid/ask > 0 predicates) on the GreptimeDB/NAS side, and always perform complex, iterative, or cross-snapshot quality filtering and feature construction on the powerful training machine."

**Current Implementation Status:**

| Filtering Type | Location | Status |
|---------------|----------|--------|
| Time range | GreptimeDB SQL | ✅ Correct |
| Symbol filtering | GreptimeDB table name | ✅ Correct |
| bid/ask > 0 | GreptimeDB SQL WHERE | ✅ Correct |
| Gap validation | Training machine | ✅ Correct |
| Spread anomalies | Training machine | ✅ Correct |
| Feature construction | Training machine | ✅ Correct |

The architecture correctly follows the hybrid filtering rule.

---

## Appendix B: Files Analyzed

| File | Lines | Status |
|------|-------|--------|
| main.py | 180 | Reviewed |
| config/training_config.yaml | 382 | Reviewed |
| config/validation_schema.yaml | 354 | Reviewed |
| data/data_loader.py | 143 | Reviewed |
| data/data_object.py | 71 | Reviewed |
| data/greptime_client.py | 346 | Reviewed |
| data/external_sources.py | 29 | Reviewed |
| preprocessing/transformer.py | 537 | Reviewed |
| preprocessing/temporal_features.py | 323 | Reviewed |
| preprocessing/snapshot_sequence_builder.py | 126 | Reviewed |
| preprocessing/train_test_split.py | 54 | Reviewed |
| preprocessing/validator.py | 38 | Reviewed |
| preprocessing/time_utils.py | 37 | Reviewed |
| models/cnn_lstm_multiclass.py | 172 | Reviewed |
| models/hyperparameter_tuning.py | 218 | Reviewed |
| models/layers.py | 21 | Reviewed |
| training/pipeline.py | 734 | Reviewed |
| training/callbacks.py | 95 | Reviewed |
| training/metrics.py | 26 | Reviewed |
| training/dataset_cache.py | 133 | Reviewed |
| evaluation/evaluator.py | 565 | Reviewed |
| evaluation/calibration.py | 109 | Reviewed |
| diagnostics/data_diagnostics.py | 1376 | Partially reviewed |
| mlflow_integration/experiment_tracker.py | 143 | Reviewed |
| mlflow_integration/model_registry.py | 65 | Reviewed |
| utils/config_loader.py | 140 | Reviewed |
| utils/colored_logging.py | 59 | Reviewed |
| utils/env_validator.py | 49 | Reviewed |
| requirements.txt | 27 | Reviewed |
| All test files | ~500 | Reviewed |

**Total Lines Reviewed:** ~6,000+

---

## Conclusion

The BinanceDataTraining project has a **solid architectural foundation** that adheres well to the user's global coding rules, particularly around YAML-first configuration, proper logging, and modular design. However, **critical implementation gaps** prevent the system from achieving its stated goal of training accurate CNN+LSTM models on order book timeseries.

The most impactful missing components are:
1. Full order book depth utilization
2. Input normalization
3. Feature engineering implementation
4. HTTP timeout handling

Addressing the Priority 1 and 2 recommendations would bring the project to a production-ready state for ML model training.

---

*End of Audit Report*
