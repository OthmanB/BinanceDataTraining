# Binance ML Training Platform - Documentation Index

**Date:** 2025-11-03  
**Time:** 17:53 UTC+09:00

---

## Document Overview

This directory contains the initial draft specifications for the Binance ML Training Platform project. These documents are intended as a foundation for discussion and refinement before full implementation begins.

---

## Documents

### 1. Vision Document
**File:** `vision.md`

**Purpose:** Outlines the high-level goals, objectives, strategic approach, and key open questions for the platform.

**Key Sections:**
- Executive Summary
- Problem Statement and Assumptions
- Solution Overview
- Architecture Philosophy
- Model Architecture
- **10 Key Open Questions** requiring decisions
- Success Criteria and Non-Goals

**Status:** Draft for Review

---

### 2. Technical Specifications
**File:** `technical-specifications.md`

**Purpose:** Provides detailed technical design including architecture, data schemas, configuration format, and implementation plan.

**Key Sections:**
- System Architecture (component diagram)
- Data Object Structure (replaces podObject from training-core)
- Complete YAML Configuration Schema
- Module Structure
- 6 Key Technical Decisions Pending
- Implementation Phases (10-week plan)
- Dependencies and Security

**Status:** Draft for Review

---

## Critical Decisions Required

The following questions must be answered before implementation can proceed. They are organized by priority and impact.

### High Priority (Architecture-Defining)

#### 1. Order Book Representation
**Location:** Vision Doc §6.2, Tech Spec §4.2

**Question:** How should we structure the order book depth data?

**Options:**
- Full 1000-level depth: `(N, T, 1000, 4, K)` tensor → Large but complete
- Aggregated 50-level depth: Reduced size, potential information loss
- Hybrid: High-res near market + aggregated deep levels

**Impact:** Memory usage, model architecture, training time

**Recommendation Needed:** This is the most critical architectural decision

---

#### 2. Multi-Asset Input Strategy
**Location:** Vision Doc §6.4, Tech Spec §4.1

**Question:** How should multiple asset order books be fed to the model?

**Options:**
- Stacked channels (concatenate as channels)
- Separate input branches (per-asset CNNs, merge before LSTM)
- Attention mechanism (dynamic asset weighting)

**Impact:** Model complexity, parameter count, interpretability

**Recommendation Needed:** Affects entire model architecture

---

#### 3. Prediction Horizon (Δt)
**Location:** Vision Doc §6.3, Tech Spec §4.3

**Question:** What time interval should the model predict?

**Options:**
- Short-term: 1-5 minutes (high-frequency trading)
- Medium-term: 15-60 minutes (swing trading)
- Long-term: 1-24 hours (position trading)
- Multi-horizon: Separate models for each

**Impact:** Use case, data requirements, label distribution, model complexity

**Recommendation Needed:** Defines the fundamental use case

---

### Medium Priority (Performance-Affecting)

#### 4. Price Classification Granularity
**Location:** Vision Doc §6.1, Tech Spec §4.4

**Question:** How many and which price range classes?

**Current Proposal:** 8 classes (±10%, ±5%, ±2%, 0%)

**Alternatives:**
- More granular (16 classes with 1% intervals)
- Fewer classes (4 classes: large down, small down/up, large up)
- Adaptive based on volatility

**Impact:** Model complexity, actionability of predictions

**Recommendation Needed:** Balance between granularity and model performance

---

#### 5. Training Strategy
**Location:** Vision Doc §6.7

**Question:** What train/validation/test strategy?

**Options:**
- Simple chronological split (70/15/15)
- Walk-forward cross-validation
- Stratified sampling for class balance
- Regime-aware splitting (bull/bear markets separately)

**Impact:** Model generalization, overfitting risk

**Recommendation Needed:** Time-series requires special consideration

---

#### 6. Hyperparameter Optimization
**Location:** Vision Doc §6.8, Tech Spec §4.5

**Question:** Include automated hyperparameter tuning?

**Options:**
- Full Optuna integration (like training-core)
- Manual tuning only
- Optional flag (can be enabled/disabled in config)

**Trade-offs:** Performance gains vs. computational cost

**Recommendation Needed:** Balance effort and benefit

---

### Lower Priority (Infrastructure)

#### 7. Temporal Features
**Location:** Vision Doc §6.5

**Question:** How to encode time-of-day, day-of-week, etc.?

**Options:**
- Embedded features in main tensor
- Separate auxiliary inputs
- Learned embeddings
- Combination

**Impact:** Minor, can be refined iteratively

---

#### 8. Data Storage and Versioning
**Location:** Vision Doc §6.6, Tech Spec §4.6

**Question:** Where to store and version large historical datasets?

**Options:**
- MLFlow artifacts (simple, potentially expensive)
- DVC (separate versioning system)
- Custom database with snapshots
- Hybrid approach

**Impact:** Reproducibility, storage costs, tooling complexity

**Recommendation Needed:** Long-term data management strategy

---

#### 9. Model Registry Strategy
**Location:** Vision Doc §6.9

**Question:** How to organize and version trained models?

**Options:**
- MLFlow Model Registry with stage promotion
- Custom semantic versioning
- Asset-specific repositories
- Unified registry with rich tagging

**Impact:** Downstream model selection, deployment workflow

---

#### 10. Evaluation Metrics
**Location:** Vision Doc §6.10

**Question:** Which domain-specific metrics beyond standard classification?

**Candidates:**
- Calibration error (probability calibration)
- Simulated P&L (backtest-style)
- Directional accuracy (up/down only)
- Confidence intervals

**Impact:** Model selection criteria, interpretability

---

## Similarities to training-core

The platform intentionally borrows proven patterns from `training-core`:

### Retained Patterns
- ✅ Modular architecture with separation of concerns
- ✅ YAML-based configuration with validation
- ✅ MLFlow experiment tracking
- ✅ CNN+LSTM architecture foundation
- ✅ Colored logging with function names and parameter highlighting
- ✅ Comprehensive preprocessing pipeline
- ✅ Hyperparameter optimization (Optuna)

### Removed Complexity
- ❌ Kubernetes pod tracking → Simplified to DataObject
- ❌ Artificial anomaly injection → Real data provides examples
- ❌ Daemon/scheduler → One-off execution
- ❌ Smart refetch → Infrequent model updates

### Key Differences
- **Problem Type:** Anomaly detection (binary) → Price prediction (multi-class probability)
- **Data Source:** Prometheus metrics → Binance order book
- **Execution Model:** Continuous monitoring → Batch training
- **Input Complexity:** Single pod metrics → Multi-asset order books

---

## Next Steps

### For Reviewer

1. **Read Vision Document** to understand goals and approach
2. **Review 10 Key Open Questions** and provide decisions
3. **Scan Technical Specifications** for feasibility assessment
4. **Provide Feedback** on ambiguities or missing considerations

### After Decisions

1. **Finalize Technical Spec** based on decisions
2. **Create Configuration Template** (YAML)
3. **Set Up Project Structure** (directories, files)
4. **Implement Phase 1** (foundation and configuration)
5. **Iterate on Architecture** with small-scale prototypes

---

## Contact and Feedback

Please provide feedback and decisions on the open questions. Any ambiguities or additional considerations should be raised before proceeding to implementation.

---

**Document Status:** Initial Draft  
**Review By:** TBD  
**Target Decision Date:** TBD  
**Implementation Start:** After decisions finalized

---

## File Structure

```
documentation/2025-11-03/
├── README.md                          # This file
├── vision.md                          # Vision document
└── technical-specifications.md        # Technical specifications
```

---
