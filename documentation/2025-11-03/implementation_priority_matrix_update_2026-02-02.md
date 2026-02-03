# Implementation Priority Matrix - Snapshot Pipeline Update

**Date:** 2026-02-02  
**Time:** 11:45 UTC+09:00  
**Reference:** documentation/2026-01-17/implementation_priority_matrix.md  
**Aligned Vision:** documentation/2025-11-03/vision_update_2026-02-02.md

---

## Purpose

This document provides an implementation-aligned update to the original priority matrix. It reflects the snapshot-first pipeline that is now the canonical training path and records what has been completed, what remains, and what must be added without regressing the current architecture.

## Priority Classification

| Priority | Timeline | Impact on Goal |
|----------|----------|----------------|
| **P1 - Critical** | Immediate | Blocks functional training |
| **P2 - High** | 1-2 weeks | Blocks accurate models |
| **P3 - Medium** | 1 month | Blocks production deployment |
| **P4 - Low** | Backlog | Nice to have |

---

## Status Summary (Snapshot Pipeline)

| Item | Status | Notes |
|------|--------|-------|
| P1.1 Full depth tensor | **Partially Implemented** | Hybrid depth representation implemented; full 1000-level tensor not implemented. |
| P1.2 Input normalization | **Implemented** | Snapshot pipeline supports min_max and standard normalization; mask channels excluded. |
| P1.3 HTTP timeouts | **Implemented** | Greptime client uses configured connect/request timeouts and retries. |
| P2.1 Feature engineering | **Implemented** | FeatureEngineer integrated into snapshot pipeline. |
| P2.2 Class weights | **Not Implemented** | Two-head class weights computation still missing. |
| P2.3 Chunked ingestion | **Implemented** | Time-chunk streaming for GreptimeDB ingestion. |
| P3.1 Multi-asset inputs | **Implemented** | Alignment (interpolate/bucket) + confidence mask channels. |
| P3.2 Fine-tuning | **Not Implemented** | No MLflow load + layer freezing support yet. |
| P3.3 Backtesting | **Not Implemented** | No PnL or simulation framework yet. |
| P4.1 Architecture improvements | **Not Implemented** | No attention, residuals, or BN additions yet. |
| P4.2 External data sources | **Not Implemented** | external_sources remains stub. |
| P4.3 Custom layers | **Not Implemented** | models/layers remains stub. |
| P4.4 Additional tests | **Partially Implemented** | Property tests added for alignment and normalization; integration tests remain missing. |

---

## P1 - Critical Items (Blocks Functional Training)

### 1.1 Full Order Book Depth Tensor Construction

**Status:** Partially implemented through hybrid depth aggregation in the snapshot pipeline.

**Snapshot-aligned interpretation:** The implemented hybrid representation delivers near-market depth resolution plus aggregated deep levels. Full 1000-level tensors remain unimplemented and should be considered an optional extension rather than a prerequisite for functional training.

### 1.2 Input Normalization Implementation

**Status:** Implemented in snapshot pipeline for min_max and standard methods, with mask channels excluded from stats.

**Remaining work:** Extend to robust normalization only if needed for stability; ensure new methods remain mask-aware.

### 1.3 HTTP Request Timeouts

**Status:** Implemented with configurable connect/request timeouts and retries in Greptime client.

---

## P2 - High Priority Items (Blocks Accurate Models)

### 2.1 Feature Engineering Pipeline

**Status:** Implemented and integrated in snapshot pipeline.

### 2.2 Class Weight Computation

**Status:** Not implemented. Two-head outputs require per-head class weighting or a unified weighting strategy. This remains a priority to improve minority class performance.

### 2.3 Chunked Data Ingestion

**Status:** Implemented via time-chunk streaming ingestion with throttling.

---

## P3 - Medium Priority Items (Blocks Production)

### 3.1 Multi-Asset Input Processing

**Status:** Implemented with configurable alignment, large-gap policy, and confidence mask channels.

### 3.2 Fine-tuning Support

**Status:** Not implemented. Requires loading MLflow-registered models and configurable layer freezing in snapshot mode.

### 3.3 Backtesting Framework

**Status:** Not implemented. Needs a dedicated evaluation module and artifact logging.

---

## P4 - Low Priority Items (Nice to Have)

### 4.1 Model Architecture Improvements

**Status:** Not implemented. Should be added only after the snapshot pipeline is stable for production use.

### 4.2 External Data Sources

**Status:** Not implemented. external_sources remains a stub.

### 4.3 Custom Keras Layers

**Status:** Not implemented. models/layers remains a stub.

### 4.4 Additional Tests

**Status:** Partially implemented. Property tests were added for multi-asset alignment and mask-aware normalization. Integration tests for full pipeline and model building remain missing.

---

## Snapshot-Specific Addendum (New Items)

The snapshot pipeline introduces additional priorities that were not explicit in the 2026-01-17 matrix but are required to complete the original vision.

1. Post-hoc calibration (temperature scaling or isotonic) for probability reliability.
2. Temporal degradation testing and rolling window validation to validate non-stationarity assumptions.
3. MLflow multi-stage lineage support, anchored on snapshot manifests.
4. Hyperparameter optimization re-enabled in snapshot mode with reproducible datasets.

These items should be treated as P3 priorities because they affect production readiness and model trustworthiness but do not block functional training.

---

## References

This update is linked to the original matrix at documentation/2026-01-17/implementation_priority_matrix.md and the snapshot-aligned vision in documentation/2025-11-03/vision_update_2026-02-02.md. The related technical debt updates are in documentation/2025-11-03/technical_debt_register_update_2026-02-02.md and the original register in documentation/2026-01-17/technical_debt_register.md.
