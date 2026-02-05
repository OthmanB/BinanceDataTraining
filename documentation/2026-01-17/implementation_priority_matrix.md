# Implementation Priority Matrix

**Date:** 2026-01-17 08:00 UTC+09:00  
**Reference:** Code Audit Report 2026-01-17

This matrix reflects the state as of 2026-01-17. For the snapshot-pipeline-aligned update and current status, see documentation/2026-02-02/implementation_priority_matrix_update_2026-02-02.md and the aligned vision at documentation/2026-02-02/vision_update_2026-02-02.md.

---

## Priority Classification

| Priority | Timeline | Impact on Goal |
|----------|----------|----------------|
| **P1 - Critical** | Immediate | Blocks functional training |
| **P2 - High** | 1-2 weeks | Blocks accurate models |
| **P3 - Medium** | 1 month | Blocks production deployment |
| **P4 - Low** | Backlog | Nice to have |

---

## P1 - Critical Items (Blocks Functional Training)

### 1.1 Full Order Book Depth Tensor Construction

**Current State:** Only top-of-book (4 values) used despite `depth_levels: 1000` config.

**Files to Modify:**
- `preprocessing/snapshot_sequence_builder.py`
- `data/greptime_client.py` (ensure depth data returned)
- `preprocessing/transformer.py` (store full depth in snapshot_features)

**Estimated Effort:** 3-5 days

**Implementation Notes:**
- Input tensor should be `(N, T, depth_levels, 2, C)` where 2 = bid/ask sides
- Or flatten to `(N, T, H, W, C)` with H = depth, W = features per level
- Consider aggregated representation as per `order_book.representation: "hybrid"`

---

### 1.2 Input Normalization Implementation

**Current State:** Config defines `preprocessing.normalization` but no code exists.

**Files to Create/Modify:**
- Create `preprocessing/normalizer.py`
- Modify `preprocessing/transformer.py` to call normalizer
- Modify `training/pipeline.py` to apply fitted normalizer

**Estimated Effort:** 2-3 days

**Implementation Notes:**
```python
# Example interface
class Normalizer:
    def fit(self, X_train: np.ndarray) -> None: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray: ...
```

Support methods: `min_max`, `standard`, `robust` as per config.

---

### 1.3 HTTP Request Timeouts

**Current State:** All HTTP requests can hang indefinitely.

**Files to Modify:**
- `data/greptime_client.py` (lines 92, 272)

**Estimated Effort:** 1 hour

**Implementation:**
```python
resp = requests.post(
    url,
    data={"sql": sql},
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    timeout=30,  # Add this
)
```

Consider adding retry logic with exponential backoff.

---

## P2 - High Priority Items (Blocks Accurate Models)

### 2.1 Feature Engineering Pipeline

**Current State:** Config lists 6 order book features and 2 derived features with no implementation.

**Files to Create:**
- `preprocessing/feature_engineering.py`

**Features to Implement:**

| Feature | Formula | Complexity |
|---------|---------|------------|
| `bid_ask_spread` | `(ask_price - bid_price) / mid_price` | Low |
| `volume_imbalance` | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` | Low |
| `depth_imbalance` | Sum imbalance across depth levels | Medium |
| `weighted_mid_price` | `(bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)` | Low |
| `price_momentum` | Rolling window price change | Medium |
| `volume_momentum` | Rolling window volume change | Medium |

**Estimated Effort:** 5-7 days

---

### 2.2 Class Weight Computation

**Current State:** `class_weights.compute_from_train: true` but weights not computed.

**Files to Modify:**
- `training/pipeline.py`

**Estimated Effort:** 2-4 hours

**Implementation:**
```python
from collections import Counter
import numpy as np

def compute_class_weights(labels: np.ndarray, num_classes: int) -> dict:
    counts = Counter(labels)
    total = len(labels)
    weights = {c: total / (num_classes * counts[c]) for c in range(num_classes)}
    return weights
```

---

### 2.3 Chunked Data Ingestion

**Current State:** Entire time range fetched in single query.

**Files to Modify:**
- `data/greptime_client.py`

**Estimated Effort:** 1-2 days

**Implementation Notes:**
- Use `ingestion.chunk_hours: 12` config
- Iterate over time windows
- Aggregate rows incrementally
- Consider memory-mapped storage for large datasets

---

## P3 - Medium Priority Items (Blocks Production)

### 3.1 Multi-Asset Input Processing

**Current State:** 6 correlated assets configured but unused.

**Decision Required:** Keep or remove multi-asset feature?

**If Keeping:**
- Files to modify: `training/pipeline.py`, `snapshot_sequence_builder.py`
- Input becomes multi-channel with asset dimension
- Estimated effort: 5-7 days

**If Removing:**
- Remove from `training_config.yaml` to avoid confusion
- Estimated effort: 30 minutes

---

### 3.2 Fine-tuning Support

**Current State:** Config exists but loading not implemented.

**Files to Modify:**
- `training/pipeline.py`
- Possibly `mlflow_integration/model_registry.py`

**Estimated Effort:** 2-3 days

**Implementation Notes:**
- Load model from MLFlow using `base_model_run_id`
- Optionally freeze early layers
- Reduce learning rate for fine-tuning

---

### 3.3 Backtesting Framework

**Current State:** Placeholder config only.

**Files to Create:**
- `evaluation/backtesting.py`

**Estimated Effort:** 5-10 days

**Metrics to Implement:**
- Cumulative PnL
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average trade duration

---

## P4 - Low Priority Items (Nice to Have)

### 4.1 Model Architecture Improvements

- Add attention mechanisms
- Add batch normalization
- Consider transformer architecture
- Residual connections

### 4.2 External Data Sources

- Implement `external_sources.py`
- Integrate macro indicators
- Sentiment feeds

### 4.3 Custom Keras Layers

- Implement MaxRescaling or other custom layers
- Add to `models/layers.py`

### 4.4 Additional Tests

- Integration tests for full pipeline
- Tests for snapshot_sequence_builder
- Tests for transformer label construction
- Performance benchmarks

---

## Dependency Graph

```
P1.3 (Timeouts) ──┐
                  │
P1.2 (Normalize) ─┼──► P2.1 (Features) ──► P2.2 (Class Weights) ──► P3.2 (Fine-tune)
                  │
P1.1 (Depth) ─────┤
                  │
P2.3 (Chunked) ───┘
                        ┌──► P3.1 (Multi-Asset)
                        │
                        └──► P3.3 (Backtest)
```

**Recommended Implementation Order:**
1. P1.3 (Timeouts) - Quick win
2. P1.2 (Normalization) - Foundation
3. P1.1 (Full Depth) - Core feature
4. P2.1 (Feature Engineering) - Model input quality
5. P2.2 (Class Weights) - Training improvement
6. P2.3 (Chunked Ingestion) - Scale to larger data
7. P3.x items as needed

---

## Effort Summary

| Priority | Items | Total Days |
|----------|-------|------------|
| P1 | 3 | 6-9 days |
| P2 | 3 | 8-13 days |
| P3 | 3 | 12-20 days |
| P4 | 4 | Backlog |

**Minimum Viable Training:** P1 items (~1-2 weeks)  
**Production Ready:** P1 + P2 items (~3-4 weeks)  
**Full Feature Set:** All items (~2-3 months)

---

*End of Priority Matrix*
