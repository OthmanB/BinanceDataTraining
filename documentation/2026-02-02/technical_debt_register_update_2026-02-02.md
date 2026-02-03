# Technical Debt Register - Snapshot Pipeline Update

**Date:** 2026-02-02  
**Time:** 11:45 UTC+09:00  
**Reference:** documentation/2026-01-17/technical_debt_register.md  
**Aligned Vision:** documentation/2026-02-02/vision_update_2026-02-02.md

---

## Debt Classification

| Severity | Definition |
|----------|------------|
| **CRITICAL** | Prevents core functionality or causes failures |
| **HIGH** | Significantly impacts quality or performance |
| **MEDIUM** | Causes maintenance burden or limits features |
| **LOW** | Code quality issues, minor improvements |

---

## Resolved or Mitigated in Snapshot Pipeline

The following items from the original register have been resolved or substantially mitigated in the snapshot pipeline.

- **TD-003 (HTTP timeouts)** resolved via configured connect/request timeouts and retries in data/greptime_client.py.
- **TD-002 (Missing input normalization)** resolved in training/snapshot_dataset.py with mask-aware stats and application.
- **TD-004 (Feature engineering missing)** resolved by preprocessing/feature_engineering.py and snapshot integration.
- **TD-007 (No chunked ingestion)** resolved via time-chunk streaming ingestion in data/greptime_client.py.
- **TD-006 (Correlated assets unused)** resolved by multi-asset alignment and mask channels in training/snapshot_dataset.py.

TD-001 (Full depth underutilization) is partially mitigated by the hybrid depth representation, but full 1000-level tensors remain unimplemented and are still treated as an optional enhancement.

---

## Active Technical Debt Items

### TD-001: Full Order Book Depth Not Fully Utilized

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | training/snapshot_dataset.py (hybrid aggregation) |
| **Description** | Hybrid depth aggregation is implemented; full 1000-level tensors are still not supported. |
| **Impact** | Potential loss of deep order book signal beyond hybrid aggregation. |
| **Remediation** | Add an optional full-depth representation if needed, keeping hybrid as the default. |

### TD-005: Class Weights Not Computed for Two-Head Outputs

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | training/pipeline.py |
| **Description** | compute_from_train is not implemented for two-head intensity outputs. |
| **Impact** | Minority classes may be underweighted; training may be biased. |
| **Remediation** | Implement per-head class weights or a unified weighting strategy compatible with snapshot generators. |

### TD-008: SQL Injection Risk in Greptime Queries

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | data/greptime_client.py |
| **Description** | Date literals are interpolated into SQL strings without escaping. |
| **Impact** | Potential SQL injection if date inputs are untrusted. |
| **Remediation** | Use parameterized queries or escape literals before interpolation. |

### TD-009: Fine-tuning Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | training/pipeline.py |
| **Description** | No support for loading and fine-tuning existing MLflow models. |
| **Impact** | Cannot resume or adapt models without full retraining. |
| **Remediation** | Add MLflow model loading, optional layer freezing, and reduced learning rates. |

### TD-010: Backtesting Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | evaluation/ (missing module) |
| **Description** | No PnL simulation or trading metrics for model outputs. |
| **Impact** | Cannot validate strategy-level performance. |
| **Remediation** | Implement evaluation/backtesting.py with PnL, drawdown, and risk metrics. |

### TD-012: Implicit Defaults in Code

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | main.py, utils/colored_logging.py |
| **Description** | A few fallback defaults remain instead of fail-fast validation. |
| **Impact** | Potential silent misconfiguration. |
| **Remediation** | Remove fallbacks and require explicit YAML values. |

### TD-013: Missing Integration Tests

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | tests/ |
| **Description** | No full pipeline integration tests; model build tests are limited. |
| **Impact** | Regression risk across pipeline boundaries. |
| **Remediation** | Add end-to-end snapshot pipeline tests using small fixtures. |

### TD-016: Post-hoc Calibration Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | evaluation/ |
| **Description** | Calibration metrics exist but temperature scaling or isotonic calibration is absent. |
| **Impact** | Predicted probabilities may be miscalibrated for trading decisions. |
| **Remediation** | Add temperature scaling or isotonic calibration in evaluation. |

### TD-017: Temporal Degradation Testing Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | evaluation/ |
| **Description** | No multi-window evaluation to validate non-stationarity assumptions. |
| **Impact** | Unclear how model performance degrades over older data. |
| **Remediation** | Implement rolling window evaluation with MLflow logging of degradation curves. |

### TD-018: MLflow CWD Side Effect

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | mlflow_integration/experiment_tracker.py |
| **Description** | start_run changes working directory to mlflow.local_tmp_dir, affecting relative paths. |
| **Impact** | Relative snapshot.directory values may resolve to unexpected locations. |
| **Remediation** | Either avoid changing CWD or convert critical paths to absolute paths during config load. |

### TD-019: Dual-Channel Long-Term Context Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | models/ and training/ |
| **Description** | No long-term channel or dual-branch model path for regime context. |
| **Impact** | Model lacks explicit regime awareness described in the original vision. |
| **Remediation** | Add long-term aggregation pipeline and dual-input model architecture. |

---

## Debt Reduction Roadmap (Snapshot-Aligned)

**Phase 1: Critical Stability**
Complete class weight computation (TD-005) and address SQL literal safety (TD-008).

**Phase 2: Model Reliability**
Implement post-hoc calibration (TD-016) and temporal degradation testing (TD-017).

**Phase 3: Production Readiness**
Add fine-tuning support (TD-009), backtesting (TD-010), and dual-channel context (TD-019).

**Phase 4: Cleanup and Testing**
Remove implicit defaults (TD-012), add integration tests (TD-013), and address MLflow CWD behavior (TD-018).

---

## References

This update is linked to the original register at documentation/2026-01-17/technical_debt_register.md and the snapshot-aligned vision in documentation/2026-02-02/vision_update_2026-02-02.md. The related priority matrix updates are in documentation/2026-02-02/implementation_priority_matrix_update_2026-02-02.md and the original matrix in documentation/2026-01-17/implementation_priority_matrix.md.
