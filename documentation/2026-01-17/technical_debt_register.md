# Technical Debt Register

**Date:** 2026-01-17 08:00 UTC+09:00  
**Reference:** Code Audit Report 2026-01-17

This register reflects the state as of 2026-01-17. For the snapshot-pipeline-aligned update and current status, see documentation/2026-02-02/technical_debt_register_update_2026-02-02.md and the aligned vision at documentation/2026-02-02/vision_update_2026-02-02.md.

---

## Debt Classification

| Severity | Definition |
|----------|------------|
| **CRITICAL** | Prevents core functionality or causes failures |
| **HIGH** | Significantly impacts quality or performance |
| **MEDIUM** | Causes maintenance burden or limits features |
| **LOW** | Code quality issues, minor improvements |

---

## Active Technical Debt Items

### TD-001: Order Book Depth Underutilization

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Location** | `preprocessing/snapshot_sequence_builder.py:117-123` |
| **Description** | Configuration specifies `depth_levels: 1000` but implementation only uses top-of-book (4 values: best bid/ask price/quantity) |
| **Impact** | Model cannot learn order book microstructure patterns; significant information loss |
| **Root Cause** | Incomplete implementation during initial development |
| **Remediation** | Implement full depth tensor construction |
| **Estimated Effort** | 3-5 days |

---

### TD-002: Missing Input Normalization

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Location** | `preprocessing/transformer.py` |
| **Description** | `preprocessing.normalization` config exists but no implementation |
| **Impact** | Training instability, poor convergence, scale-sensitive model behavior |
| **Root Cause** | Implementation deferred/forgotten |
| **Remediation** | Create `Normalizer` class supporting min-max/standard/robust |
| **Estimated Effort** | 2-3 days |

---

### TD-003: HTTP Requests Without Timeout

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Location** | `data/greptime_client.py:92, 272` |
| **Description** | All `requests.post()` calls lack timeout parameter |
| **Impact** | Pipeline can hang indefinitely on network issues |
| **Root Cause** | Oversight during implementation |
| **Remediation** | Add `timeout=30` to all HTTP calls; add retry logic |
| **Estimated Effort** | 1-2 hours |

---

### TD-004: Feature Engineering Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | `preprocessing/` (missing file) |
| **Description** | Config defines 8 features (bid_ask_spread, volume_imbalance, depth_imbalance, weighted_mid_price, price_momentum, volume_momentum) with no implementation |
| **Impact** | Model receives raw data only; missing derived signals |
| **Root Cause** | Deferred to later phase |
| **Remediation** | Create `feature_engineering.py` with all configured features |
| **Estimated Effort** | 5-7 days |

---

### TD-005: Class Weights Not Computed

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | `training/pipeline.py` |
| **Description** | `class_weights.compute_from_train: true` but no automatic computation |
| **Impact** | Model biased toward majority class; poor minority class performance |
| **Root Cause** | Partial implementation |
| **Remediation** | Implement inverse frequency class weight computation |
| **Estimated Effort** | 2-4 hours |

---

### TD-006: Correlated Assets Unused

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | `training/pipeline.py`, `preprocessing/` |
| **Description** | 6 correlated assets configured but only target asset processed |
| **Impact** | Multi-asset information advantages not utilized |
| **Root Cause** | Feature scope reduction without config cleanup |
| **Remediation** | Either implement multi-asset processing or remove from config |
| **Estimated Effort** | 5-7 days (if implementing) or 30 min (if removing) |

---

### TD-007: No Chunked Data Ingestion

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Location** | `data/greptime_client.py` |
| **Description** | `chunk_hours: 12` config ignored; entire time range queried at once |
| **Impact** | Memory issues with large date ranges; potential timeout |
| **Root Cause** | Simplification during initial development |
| **Remediation** | Implement iterative chunked fetching |
| **Estimated Effort** | 1-2 days |

---

### TD-008: SQL Injection Risk

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | `data/greptime_client.py:254-259` |
| **Description** | Date literals inserted via f-string without escaping |
| **Impact** | Potential SQL injection if dates come from untrusted source |
| **Root Cause** | Convenience over security |
| **Remediation** | Use parameterized queries or proper escaping |
| **Estimated Effort** | 2-4 hours |

---

### TD-009: Fine-tuning Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | `training/pipeline.py` |
| **Description** | `training.fine_tuning` config exists but model loading not implemented |
| **Impact** | Cannot resume training from previous models |
| **Root Cause** | Deferred feature |
| **Remediation** | Implement MLFlow model loading by run_id |
| **Estimated Effort** | 2-3 days |

---

### TD-010: Backtesting Not Implemented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | `evaluation/` (missing file) |
| **Description** | `evaluation.backtesting` config is placeholder only |
| **Impact** | Cannot validate trading strategy profitability |
| **Root Cause** | Out of initial scope |
| **Remediation** | Create `backtesting.py` with PnL simulation |
| **Estimated Effort** | 5-10 days |

---

### TD-011: External Data Sources Stub

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Location** | `data/external_sources.py` |
| **Description** | Returns empty dict unconditionally |
| **Impact** | Cannot incorporate macro/sentiment data |
| **Root Cause** | Placeholder implementation |
| **Remediation** | Implement external data fetching or remove stub |
| **Estimated Effort** | 3-5 days (if implementing) |

---

### TD-012: Implicit Defaults in Code

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | `main.py:62-65`, `utils/colored_logging.py:35` |
| **Description** | Some fallback defaults embedded in code instead of failing on missing config |
| **Impact** | Violates "no implicit defaults" rule; potential silent misconfiguration |
| **Root Cause** | Convenience during development |
| **Remediation** | Remove fallbacks; require all values in YAML |
| **Estimated Effort** | 1-2 hours |

---

### TD-013: Missing Test Coverage

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | `tests/` |
| **Description** | No tests for: snapshot_sequence_builder, transformer label construction, cnn_lstm model, integration tests |
| **Impact** | Regressions may go undetected |
| **Root Cause** | Test development not prioritized |
| **Remediation** | Add comprehensive test suite |
| **Estimated Effort** | 3-5 days |

---

### TD-014: Custom Layers Stub

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | `models/layers.py` |
| **Description** | Returns empty dict; no custom layers implemented |
| **Impact** | Cannot use advanced layer types |
| **Root Cause** | Not yet needed |
| **Remediation** | Implement as needed or remove file |
| **Estimated Effort** | Variable |

---

### TD-015: Large Function in transformer.py

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Location** | `preprocessing/transformer.py:181-533` |
| **Description** | `_build_targets_from_order_book` is 350+ lines |
| **Impact** | Difficult to maintain and test |
| **Root Cause** | Organic growth |
| **Remediation** | Refactor into smaller focused functions |
| **Estimated Effort** | 1-2 days |

---

## Debt Summary by Severity

| Severity | Count | Total Estimated Effort |
|----------|-------|----------------------|
| CRITICAL | 3 | 6-10 days |
| HIGH | 4 | 12-18 days |
| MEDIUM | 4 | 12-20 days |
| LOW | 4 | 6-10 days |
| **TOTAL** | **15** | **36-58 days** |

---

## Debt Reduction Roadmap

### Phase 1: Critical Fixes (Week 1-2)
- TD-003: HTTP Timeouts
- TD-002: Input Normalization
- TD-001: Order Book Depth

### Phase 2: Quality Improvements (Week 3-4)
- TD-004: Feature Engineering
- TD-005: Class Weights
- TD-007: Chunked Ingestion
- TD-006: Correlated Assets (decision)

### Phase 3: Production Readiness (Week 5-8)
- TD-008: SQL Injection
- TD-009: Fine-tuning
- TD-010: Backtesting
- TD-013: Test Coverage

### Phase 4: Cleanup (Backlog)
- TD-011: External Sources
- TD-012: Implicit Defaults
- TD-014: Custom Layers
- TD-015: Function Refactoring

---

*End of Technical Debt Register*
