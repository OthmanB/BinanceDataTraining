# Binance ML Training Platform - Vision Document

**Date:** 2025-11-03  
**Time:** 23:00 UTC+09:00  
**Version:** 0.3 - Final Refinements  
**Status:** Ready for Implementation Planning

---

## 1. Executive Summary

This document outlines the vision for a Machine Learning training platform designed to analyze Binance order book data and produce probabilistic price prediction models. The platform will leverage multi-year, high-frequency order book depth data across multiple cryptocurrency pairs to train CNN+LSTM models that estimate the probability distribution of future price movements.

The models produced will be logged as experiments in MLFlow for downstream consumption by separate signal generation applications, which are outside the scope of this platform.

---

## 2. Problem Statement

### 2.1 Core Challenge

Cryptocurrency price movements are influenced by complex, multi-asset correlations and order book dynamics. Traditional single-asset analysis fails to capture the full picture of market behavior. We need a system that can:

- Process high-volume order book data at 10-second cadence across multiple trading pairs
- Account for cross-asset correlations (e.g., BTC/USDT, ETH/USDT, BNB/USDT, and cross-pairs)
- Produce probabilistic forecasts rather than binary predictions
- Support extensibility for future data sources (external markets, sentiment, temporal factors)

### 2.2 Key Assumptions

**Assumption 1:** Past price actions and order book dynamics are representative of future behavior  
**Assumption 2:** Multi-asset correlation data significantly improves prediction accuracy  
**Assumption 3:** Order book depth up to 1000 levels contains meaningful signal  

---

## 3. Solution Overview

### 3.1 Strategic Approach

The platform will train deep learning models on historical order book data to estimate probability distributions across multiple price range classes. The first canonical implementation is based on a CNN+LSTM architecture, but the pipeline itself remains model-agnostic and can host alternative model families configured via the pipeline orchestration layer.

**Example Price Classes:** `C = [p < -10%, -10% ≤ p < -5%, -5% ≤ p < -2%, -2% ≤ p < 0%, 0% < p < 2%, 2% < p ≤ 5%, 5% < p ≤ 10%, p > +10%]`

The canonical model formulation is: **P(price ∈ C_i | current_state, correlated_assets)** where current_state includes multi-asset order books and temporal features.

### 3.2 Data Foundation

- **Source:** Years of Binance order book depth data (up to 1000 levels)
- **Cadence:** 10-second sampling intervals
- **Assets:** Multiple trading pairs including BTC/USDT, ETH/USDT, BNB/USDT, ADA/USDT, and cross-pairs (ADA/BTC, ETH/BTC, etc.)
- **Volume:** High duty cycle coverage enabling robust training without artificial data generation

### 3.3 Extensibility Vision

The architecture will support future integration of:
- External market data (e.g., S&P 500 correlations)
- Temporal/seasonal factors (time of day, day of week)
- Non-rational signals (sentiment analysis, social media trends)
- News feed digestion via LLM integration (local or remote)
- Psychological factors (market psychology indicators)

---

## 4. Architecture Philosophy

### 4.1 Inspired by training-core

The platform architecture is inspired by the proven `training-core` system, adopting its strengths while simplifying for the specific use case:

**Retain:**
- Modular design with clear separation of concerns
- YAML-based configuration with comprehensive validation
- MLFlow integration for experiment tracking
- Extensible model architecture pattern
- Robust preprocessing pipeline
- Comprehensive logging with color-coded output

**Simplify/Remove:**
- No Kubernetes pod tracking logic → simplified to `dataObject` abstraction
- No artificial anomaly injection → real data provides sufficient examples
- No daemon/scheduler → one-off training execution model
- No smart refetch → models updated infrequently (months)

### 4.2 Design Principles

1. **Configuration-Driven:** All parameters in YAML, no hardcoded values
2. **Fail-Fast Validation:** Validate configuration and parameters at startup
3. **Modular Components:** Clear separation between data loading, preprocessing, model training, and evaluation
4. **Extensible Inputs:** Support for multiple input streams with plugin-like architecture
5. **Reproducibility:** Full MLFlow logging of parameters, metrics, and artifacts
6. **Security-First:** Environment variables for sensitive data (API keys, credentials)

### 4.3 Modular Multi-Stage Pipeline and Multi-Model Support

The platform follows a configuration-driven, stage-based pipeline inspired by `training-core`. A central orchestration layer reads a YAML configuration that specifies which master function to call for each stage and which MLFlow experiment to use for logging.

The logical stages are:

1. **Data Collection (model-agnostic):** Fetch raw order book and auxiliary data from the configured sources (e.g., GreptimeDB, external feeds).
2. **DataObject Construction (model-agnostic):** Organize raw data into an intelligible `DataObject` structure that is strictly model-agnostic and reusable across all downstream models.
3. **General Preprocessing (model-agnostic):** Apply a configurable sequence of preprocessing steps (e.g., gap interpolation, NaN removal, time alignment, sample slicing) defined by master function names in YAML.
4. **Model-Specific Feature Pipeline:** Transform the generic `DataObject` into model-ready tensors for a specific ML approach (e.g., CNN+LSTM, Transformer, gradient-boosted trees). The selected pipeline is configured by name in YAML.
5. **Model Training & Evaluation:** Invoke the configured training function for the chosen model implementation, including hyperparameter search when enabled.

Each stage is:

- **Configured in YAML** by function name and parameters (no hardcoded defaults).
- **Executed by the orchestrator**, which enforces ordering and validation.
- **Logged to its own MLFlow experiment**, enabling clear lineage and easy comparison of alternative preprocessing pipelines and model implementations.

---

## 5. Canonical Model Architecture and Multi-Model Strategy

### 5.1 Multi-Input CNN+LSTM Model

The model will accept multiple input channels:

```
Input 1: Order books (multiple assets)          ──┐
Input 2: Local timestamp information            ──┤
Input 3: Global timestamp information           ──┼──> CNN+LSTM Model ──> Probability Distribution (Classes C)
Input 4...K: External information sources       ──┘
```

**Similarity to training-core:** The CNN+LSTM architecture follows the pattern in `strategy_2DCNNLSTM.py` but adapted for:
- Multi-class probability output (softmax) instead of binary classification (sigmoid)
- Multi-asset input tensors instead of single-pod metrics
- Potentially deeper order book representation (up to 1000 levels vs smaller metric matrices)

### 5.2 Output Specification

**Multi-Class Probability Distribution:**
- Output layer: Softmax activation over price range classes
- Loss function: Categorical cross-entropy
- Metrics: Class-wise accuracy, precision, recall, F1, confusion matrix

---

## 6. Key Open Questions

### 6.1 Classification Granularity

**Decision**: Adaptive/configurable class definitions per asset pair

**Implementation**:
- Start with 8 classes as baseline: `[p < -10%, -10% ≤ p < -5%, -5% ≤ p < -2%, -2% ≤ p < 0%, 0% < p < 2%, 2% < p ≤ 5%, 5% < p ≤ 10%, p > +10%]`
- Class boundaries fully configurable in YAML per asset
- Allow experimentation with different granularities (8, 12, 16 classes) via hyperparameter search
- Track class distribution in MLFlow to identify if refinement needed

---

### 6.2 Order Book Representation and Data Pipeline Architecture

**Decision**: Preprocessed adaptive-depth snapshots with intelligent aggregation

**Critical Analysis**: Your GreptimeDB schema stores order book as "tall" format (one row per depth level). Querying millions of rows per training sample creates severe I/O bottleneck. Direct training from database is infeasible.

**Architecture**:

**Phase 1: Preprocessing Pipeline** (batch-by-batch, on-demand)
```
GreptimeDB → Daily Snapshot Extraction → Adaptive Depth Analysis → Temporary .npz Cache
```

**Strategy**: Process day-by-day, create temporary cache, discard after use

1. **Daily batch extraction** (per symbol):
   - Query GreptimeDB for 1 day of data
   - Group by timestamp (`batch_id`) and reshape to (timestamps, depth, 4) tensors
   - **Adaptive depth detection**: Analyze actual non-zero depth levels (typically < 200 instead of 1000)
   - **Smart aggregation**: 
     - Keep high-resolution near market (top 50 levels: bid[0:50], ask[0:50])
     - Aggregate deeper levels into bins (e.g., 10 bins of 10 levels each)
     - Final shape: `(timestamps, ~100 effective levels, 4)`
   - Store as temporary compressed `.npz`: `${CACHE_DIR}/{symbol}_{date}.npz`

2. **Usage and cleanup**:
   - Load into memory-mapped arrays for training samples
   - After processing day's samples: **discard npz file**
   - **Keep random fraction** (e.g., 1%) for diagnostic artifacts in MLFlow
   - **Exception**: Long-term channel aggregations may need persistent storage

3. **Benefits**:
   - 10-100x compression vs raw GreptimeDB
   - Handles sparsity intelligently
   - Minimal disk footprint (only 1-2 days cached at a time)
   - GreptimeDB remains source of truth

**Phase 2: Training Data Loading**
```
Temporary .npz cache → Memory-mapped loading → Sample generation → GPU → Discard cache
```

**Storage Estimate**:
- Raw GreptimeDB: ~1-2 TB (immutable)
- Temporary cache: ~5-10 GB (1-2 days, rolling)
- MLFlow diagnostic samples: ~1-2 GB (random fraction)
- Long-term aggregations: ~20-50 GB (persistent)

**Configuration**:
```yaml
data:
  preprocessing:
    depth_strategy: "adaptive_hybrid"
    near_market_levels: 50  # High resolution
    deep_aggregation_bins: 10
    min_nonzero_threshold: 5  # Discard levels with < 5 non-zero samples
  
  storage:
    format: "npz_compressed"
    cache_dir: "${DATA_CACHE_DIR}/preprocessed"
```

**MLFlow Logging**: Store preprocessing config, sample snapshots (as images), and compression ratios as artifacts.

**Hardware**: 2x RTX 3090 (24GB each) sufficient for this approach.

---

### 6.3 Time Horizon and Prediction Window

**Decision**: Configurable medium-term horizons with multi-confirmation strategy

**Implementation**:
- Primary target: Δt = 15-60 minutes (medium-term swing trading)
- **Fully configurable in YAML**:
  ```yaml
  targets:
    prediction_horizon_seconds: 1800  # 30 minutes (configurable)
    multi_horizon_prediction: false    # Future: train multiple Δt models
  ```
- Training strategy supports multiple confirmations: model can be queried at different time slices
- Inference time (few seconds) acceptable for this horizon
- Balance: Long enough for meaningful trends, short enough for actionable decisions

**Rationale**: Avoid high-frequency noise while maintaining prediction precision. Supports strategy of accumulating multiple confirmations before trading decision (handled by downstream application).

---

### 6.4 Multi-Asset Input Strategy and Critical Preprocessing

**Decision**: Stacked channels with rigorous time-alignment preprocessing

**Architecture**: Concatenate all asset order books as channels in single tensor
- Shape: `(batch, time_steps, depth, features, num_assets)`
- CNN learns cross-asset correlations automatically through convolutional filters
- Enables dimensionality reduction from high-D input → low-D probability classes
- Goal: Discover patterns "humanly impossible to detect"

**CRITICAL: Time Alignment and Gap Handling**

**Problem**: Your concern is valid—multi-asset stacking ONLY works with synchronized timestamps. Misaligned data destroys cross-correlation learning.

**Robust Preprocessing Pipeline**:
```yaml
preprocessing:
  time_alignment:
    reference_asset: "BTCUSDT"        # Master timeline
    target_cadence_seconds: 10
    tolerance_ms: 500                  # ±500ms alignment tolerance
    
  resampling:
    method: "forward_fill"             # Or "linear_interpolate"
    max_gap_seconds: 60               # Fail sample if gap > 1min
    
  missing_data_handling:
    strategy: "skip_sample"            # Discard samples with gaps
    min_assets_required: 6             # Need 6/7 assets minimum
    partial_fill_threshold: 0.02       # Allow 2% missing within tolerance
    
  quality_checks:
    check_monotonic_time: true
    check_price_sanity: true          # Detect outliers/errors
    check_bid_ask_spread: true        # Inversions indicate bad data
```

**Implementation Steps**:
1. **Temporal grid construction**: Create uniform 10s grid based on reference asset
2. **Per-asset resampling**: Align each asset to grid using forward-fill/interpolation
3. **Gap detection**: Mark samples with gaps > threshold
4. **Quality filtering**: Discard low-quality samples
5. **Multi-asset stacking**: Only stack if all required assets present

**MLFlow Logging**: Track preprocessing statistics (% samples discarded, gap distributions, alignment quality) as metrics.

**This preprocessing is NON-NEGOTIABLE**—without it, the model learns from misaligned data and fails.

---

### 6.5 Temporal Information Encoding

**Decision**: Hybrid cyclical + trend encoding (Option D)

**Rationale**: Linear time alone FAILS for markets—it treats Dec 2024 as "farther" from Jan 2024 than Nov 2024, violating periodicity of market patterns.

**Ranking**:

**1st: Cyclical + Trend Combination** (Recommended)
```python
temporal_features = {
    # Cyclical encoding (captures periodic patterns)
    'hour_sin': sin(2π × hour / 24),
    'hour_cos': cos(2π × hour / 24),
    'day_of_week_sin': sin(2π × day / 7),
    'day_of_week_cos': cos(2π × day / 7),
    
    # Trend encoding (captures regime evolution)
    'days_since_epoch': (timestamp - epoch) / max_days,  # Normalized [0, 1]
    'market_regime': regime_indicator,  # From long-term channel
    
    # Volatility context (from medium-term preprocessing)
    'rolling_volatility_24h': vol_24h / vol_max,
    'volume_percentile': current_vol_rank
}
```

**Why cyclical encoding**:
- Hour 23 and Hour 0 are correctly "close" in feature space: `[sin(23×2π/24), cos(23×2π/24)]` ≈ `[sin(0), cos(0)]`
- Captures daily activity patterns (Asian/European/US sessions)
- Weekend vs weekday patterns

**2nd: Learned Embeddings** (Alternative)
- More data required, less interpretable, but flexible

**3rd: Simple embedded features** (Baseline)
- Hour as integer [0-23]—model must learn relationships

**Worst: Linear time only**—mathematically incorrect for periodic phenomena

**Configuration**:
```yaml
data:
  temporal_features:
    cyclical:
      - "hour_of_day"      # → sin/cos pair
      - "day_of_week"      # → sin/cos pair
    trend:
      - "days_since_start"  # Normalized linear
      - "market_regime"     # From long-term channel
    volatility_context:
      - "rolling_vol_24h"
      - "volume_percentile"
```

---

### 6.6 Training Data Management and Multi-Scale Architecture

**Decision**: Dual-channel architecture with time-weighted sampling

**Data Storage Strategy**:
- **Raw data**: Remains in GreptimeDB (already managed, historical, immutable)
- **Preprocessed snapshots**: Compressed `.npz` files (see §6.2)
- **No data versioning needed**: Historical data doesn't change
- **MLFlow artifacts**: Sample visualizations (heatmaps, PNGs), preprocessing configs, query metadata
- **Reproducibility**: MLFlow tags/params store exact queries, time ranges, preprocessing parameters

**Critical Multi-Scale Challenge**: You correctly identified need for **two temporal resolutions**

**Dual-Channel Architecture**:
```
Short-term Channel (ST):  [24h window @ 10s resolution]
  ├─ Input shape: (N, 8640 timesteps, depth, 4, assets)
  ├─ Purpose: Capture immediate price dynamics
  └─ Processing: CNN → LSTM

Long-term Channel (LT):  [90-day window @ 1-day resolution]
  ├─ Input shape: (N, 90 timesteps, summary_stats, assets)
  ├─ Purpose: Market regime context, calibration
  ├─ Features: [mean, std, skew, kurtosis, volume] per day
  └─ Processing: Separate CNN

              ┌────────────┐
    ST ──────►│  CNN+LSTM  │
              └──────┬─────┘
                     │
              ┌──────▼─────┐
    LT ──────►│  CNN       │
              └──────┬─────┘
                     │
              ┌──────▼─────────┐
              │  Concatenate   │
              │  Dense Layers  │
              └──────┬─────────┘
                     │
              ┌──────▼─────────┐
              │  Softmax (8)   │
              └────────────────┘
```

**Long-term preprocessing** (daily aggregation from short-term data):
```python
for each day:
    daily_stats = {
        'mean': np.mean(orderbook_snapshots, axis=time),
        'std': np.std(orderbook_snapshots, axis=time),
        'skew': scipy.stats.skew(orderbook_snapshots, axis=time),
        'kurtosis': scipy.stats.kurtosis(orderbook_snapshots, axis=time),
        'total_volume': np.sum(volumes, axis=time)
    }
```

**Time-Weighted Sampling** (handles non-stationarity):
```yaml
training:
  sample_weighting:
    enabled: true
    method: "exponential_decay"
    half_life_days: 90  # Hyperparameter (tunable via Optuna)
    apply_to: "loss_function"  # Deterministic weighting
```

**Implementation** (single approach - weighting in loss function): 
```python
# Compute sample weights based on age
sample_age_days = (current_date - sample_date).days
weights = np.exp(-sample_age_days * np.log(2) / half_life_days)

# Apply in loss function
loss = weighted_categorical_crossentropy(y_true, y_pred, weights)
```

**Benefits of this approach**:
- Preserves entire dataset (no resampling)
- Deterministic (reproducible)
- Half-life tunable as Optuna hyperparameter
- Can use multiple half-lives for multi-scale regime detection

**Multi-Scale Regime Detection** (enhancement):
```yaml
training:
  multi_scale_weighting:
    enabled: true
    time_scales:
      - name: "short_term_memory"
        half_life_days: 7      # Weekly patterns
      - name: "medium_term_memory"  
        half_life_days: 30     # Monthly patterns
      - name: "long_term_memory"
        half_life_days: 90     # Quarterly patterns
    # Weighted combination or separate regime models
```

**Regime-Aware Training** (addressing regime change challenge): 

**Problem**: Exponential decay assumes recent data always better → fails during regime shifts (bull→bear)

**Solution**: Regime detection via long-term channel
```yaml
training:
  regime_detection:
    enabled: true
    source: "long_term_channel"  # Outputs regime indicator
    regimes: ["bull", "bear", "sideways", "high_volatility"]
    
    strategy: "regime_as_feature"  # Or "regime_specific_models"
    
    # Option A: Use regime as additional input feature
    regime_as_feature:
      encoding: "one_hot"  # [1,0,0,0] for bull, etc.
      
    # Option B: Train separate models per regime (future)
    regime_specific_models:
      enabled: false
      train_per_regime: true
      ensemble_strategy: "regime_weighted"
```

**Benefits**:
- Model can learn regime-dependent patterns
- Addresses sudden regime changes
- Long-term channel provides regime context
- Compatible with multi-scale time weighting

**Monitoring**: Track temporal degradation metrics per regime (see §6.7)

**Configuration**:
```yaml
data:
  multi_scale:
    short_term:
      window_hours: 24
      resolution_seconds: 10
    long_term:
      window_days: 90
      resolution_days: 1
      aggregation: ["mean", "std", "skew", "kurtosis", "volume"]
```

---

### 6.7 Model Training Strategy with Temporal Degradation Testing

**Decision**: Expanding-window validation + multi-horizon test splits

**Your Temporal Degradation Testing Insight**: Excellent—this validates the exponential decay assumption empirically.

**Training/Validation Strategy**:
```
Expanding Window Cross-Validation (for hyperparameter tuning):

Window 1:  |████████████ train ████████████|──val──|
             (months 0-12)                   (12-14)

Window 2:  |████████████████ train ████████████████|──val──|
             (months 0-14)                           (14-16)

Window 3:  |████████████████████ train ████████████████████|──val──|
             (months 0-16)                                   (16-18)
```

**Rationale**: Prevents overfitting to single validation period, tests generalization across time

**Multi-Horizon Test Strategy** (validates degradation hypothesis):
```yaml
testing:
  recent_performance:     # Should match validation performance
    time_range: ["24_months - 2_months", "24_months"]
    expected_performance: "~val_metrics"
    
  moderate_age:           # Tests 6-month degradation
    time_range: ["18_months", "20_months"]
    hypothesis: "Slight degradation if market regime changed"
    
  old_data:               # Tests 12-month degradation  
    time_range: ["12_months", "14_months"]
    hypothesis: "Significant degradation if non-stationary"
    
  very_old_data:          # Tests 18-month degradation
    time_range: ["6_months", "8_months"]
    hypothesis: "Validates exponential decay assumption"
```

**MLFlow Metrics**:
```python
for test_split in ["recent", "6mo_old", "12mo_old", "18mo_old"]:
    mlflow.log_metric(f"test_accuracy_{test_split}", accuracy)
    mlflow.log_metric(f"test_f1_{test_split}", f1)
    # Plot degradation curve
    mlflow.log_artifact(f"degradation_curve.png")
```

**Key Insight**: If old-data performance degrades exponentially, validates your weighting strategy. If it doesn't, suggests market is more stationary than assumed → adjust half-life.

**Rolling Window Validation during Optuna** (configurable):
```yaml
hyperparameter_optimization:
  rolling_window_validation:
    enabled: true  # YAML-configurable boolean
    n_windows: 3   # Number of validation windows
    # Prevents hyperparameter overfitting to single time period
    # Tunes models for robustness across time
```

When enabled, Optuna evaluates each trial across multiple validation windows and uses average performance.

---

### 6.8 Hyperparameter Optimization

**Decision**: Comprehensive Optuna-based optimization (following training-core pattern)

**Implementation**: Identical to `training-core/ML/strategy_2DCNNLSTM.py::OptimizeHyperparameters`

**Search Space** (expanded for multi-scale architecture):
```yaml
hyperparameter_optimization:
  enabled: true
  framework: "optuna"
  n_trials: 30  # Configurable
  direction: "maximize"
  metric: "val_f1_weighted"  # Multi-class F1
  
  search_space:
    # Short-term CNN parameters
    st_cnn_filters_1: [32, 128]
    st_cnn_filters_2: [64, 256]
    st_lstm_units: [64, 256]
    
    # Long-term CNN parameters
    lt_cnn_filters: [16, 64]
    lt_dense_units: [32, 128]
    
    # Shared parameters
    learning_rate: [0.0001, 0.01, "log"]
    batch_size: [16, 32, 64]
    dropout_rate: [0.1, 0.5]
    
    # Time-weighting parameter
    sample_weight_half_life_days: [30, 180]
```

**MLFlow Structure** (from training-core):
```
Experiment: "binance-price-prediction"
  ├─ Parent Run: "hyperparameter_optimization"
  │   ├─ Nested Run: "trial_0"
  │   ├─ Nested Run: "trial_1"
  │   └─ ... (30 trials)
  └─ Final Run: "best_model_training"
```

**Computational Cost**: Managed via
- Early stopping per trial (patience=5)
- Pruning poor trials (Optuna median pruner)
- Parallel trials if multi-GPU (2x RTX 3090 → 2 parallel trials)

**No manual tuning**—fully automated as in training-core.

---

### 6.9 MLFlow Experiment Organization

**Decision**: Use one MLFlow experiment per logical pipeline stage (data collection → DataObject construction → preprocessing → dataset/feature preparation → hyperparameter optimization → model training), mirroring the modular pipeline defined in §4.3 and the lineage pattern from `training-core`.

**Experiment Hierarchy** (per stage and, where relevant, per model implementation):

```
Experiment 1: "binance-data-collection"
  ├─ Run: "raw_ingestion_2024-01-15"
  │   ├─ Artifacts: Raw query manifests, basic diagnostics
  │   ├─ Params: Source URI, time range, asset list
  │   └─ Tags: {stage: data_collection}
  │
Experiment 2: "binance-data-object"
  ├─ Run: "data_object_BTCUSDT_2024-01-15"
  │   ├─ Artifacts: Sample `DataObject` snapshots
  │   ├─ Params: DataObject schema version, aggregation config
  │   └─ Tags: {stage: data_object}
  │
Experiment 3: "binance-preprocessing"
  ├─ Run: "BTCUSDT_preprocessing_2024-01-15"
  │   ├─ Artifacts: Preprocessed .npz files (samples only)
  │   ├─ Params: Query, time range, depth strategy
  │   ├─ Metrics: Compression ratio, gap statistics
  │   └─ Tags: {asset: BTCUSDT, stage: preprocessing}
  │
  └─ Run: "multi_asset_alignment_2024-01-15"
      ├─ Artifacts: Alignment quality plots
      ├─ Params: Tolerance, resampling method
      └─ Metrics: % samples discarded, alignment errors

Experiment 4: "binance-dataset-preparation"
  ├─ Run: "BTCUSDT_30min_dataset_v1"
  │   ├─ Artifacts: Train/val/test splits metadata and feature tensors
  │   ├─ Params: Δt, class boundaries, split ratios, feature_pipeline_id
  │   ├─ Metrics: Class distributions, dataset statistics
  │   └─ Tags: {stage: dataset_preparation, source_preprocessing_run_id: <uuid>}
  │
  └─ (Lineage tracking to preprocessing and data-object runs)

Experiment 5: "binance-hyperparameter-optimization"
  ├─ Parent Run: "BTCUSDT_30min_opt_2024-01-20"
  │   ├─ Nested Run: "trial_0" (params, metrics, artifacts)
  │   ├─ Nested Run: "trial_1"
  │   └─ ... (30 trials)
  │   ├─ Params: {n_trials, search_space, model_key}
  │   └─ Tags: {stage: hyperparameter_optimization, source_dataset_run_id: <uuid>}
  │
  └─ (Links to dataset preparation run)

Experiment 6: "binance-model-training"
  ├─ Run: "BTCUSDT_30min_cnn_lstm_v1"
  │   ├─ Artifacts: Model, plots, calibration curves
  │   ├─ Params: Best hyperparameters from Optuna, model_key
  │   ├─ Metrics: Train/val/test metrics, degradation tests
  │   └─ Tags: {stage: model_training, best_trial_number, optimization_run_id}
  │
  └─ (Full lineage: data_collection → data_object → preprocessing → dataset_preparation → optimization → training)
```

**Lineage Tracking** (critical for reproducibility across stages and models):
```python
with mlflow.start_run(experiment_id=training_exp) as run:
    # Link to upstream runs
    mlflow.set_tag("preprocessing_run_id", preproc_run_id)
    mlflow.set_tag("dataset_run_id", dataset_run_id)
    mlflow.set_tag("optimization_run_id", opt_run_id)
    
    # Asset and configuration metadata
    mlflow.set_tag("asset", "BTCUSDT")
    mlflow.set_tag("prediction_horizon", "30min")
```

**Model Registry**: Not implemented initially (manual model selection via experiment comparison).

**Future**: Can add MLFlow Model Registry later for production deployment workflow.

---

### 6.10 Evaluation Metrics and Calibration Strategy

**Decision**: Comprehensive evaluation suite with mandatory calibration analysis

**Standard Multi-Class Metrics**:
```yaml
evaluation:
  standard_metrics:
    - "accuracy"
    - "precision_per_class"
    - "recall_per_class"
    - "f1_per_class"
    - "weighted_f1"  # Primary metric for optimization
    - "confusion_matrix"
```

**Domain-Specific Metrics**:

**1. Calibration Analysis** (CRITICAL)

**Your Calibration Hypothesis: PARTIALLY CORRECT—Needs Refinement**

You conflate two distinct concepts:

**A) Scale Context** (what long-term channel provides):
- ✅ **Correct**: Long-term statistics help model understand current price regime
- ✅ **Correct**: A 5% move means different things in different volatility regimes
- ✅ **Correct**: Time-dependent context is essential

**B) Probability Calibration** (separate statistical property):
- ⚠️ **Your assumption**: Long-term channel automatically calibrates probabilities
- ❌ **Reality**: Calibration is about **frequency matching**, not scale preservation

**Calibration Definition**: 
If model predicts 70% probability for class C_i, then **across all predictions with ~70% probability**, the true class should be C_i approximately 70% of the time.

**Why long-term channel helps but isn't sufficient**:
- Long-term channel provides **regime context** (bull/bear, high/low volatility)
- This helps model understand **what price movements are typical**
- **BUT**: Neural networks often produce overconfident or underconfident probabilities
- **Example**: Model might predict [0.1, 0.05, 0.8, 0.05] when true calibrated probabilities are [0.2, 0.1, 0.5, 0.2]

**Solution: Two-Stage Approach**

**Stage 1: Long-term channel for regime context** (your idea ✓)
```python
# Model learns scale-dependent patterns
long_term_features = [mean, std, skew, kurtosis, volatility]
model_output = CNN_LSTM(short_term_data, long_term_features)
# → Raw probabilities (may be miscalibrated)
```

**Stage 2: Post-hoc calibration** (additional step needed)
```python
# Temperature scaling on validation set
calibrated_probs = softmax(logits / T)  # T is learnable parameter
# OR
# Isotonic regression (non-parametric calibration)
calibrated_probs = isotonic_calibrator.transform(raw_probs)
```

**Why this matters for trading**:
- Uncalibrated: Model says "90% up" but it's actually 60% → overconfident → bad position sizing
- Calibrated: Model probabilities directly usable for Kelly criterion, risk management

**Implementation**:
```yaml
evaluation:
  calibration:
    enabled: true
    
    # Calibration metrics
    metrics:
      - "expected_calibration_error"  # ECE: average calibration gap
      - "maximum_calibration_error"   # MCE: worst-case gap
      - "brier_score"                 # Probability accuracy
    
    # Calibration visualization
    plots:
      - "reliability_diagram"  # Per-class calibration curves
      - "confidence_histogram" # Distribution of predicted probabilities
    
    # Post-hoc calibration
    post_hoc:
      enabled: true
      method: "temperature_scaling"  # Simple, effective
      validation_set: "separate"     # 10% holdout from validation
      
  # 2. Directional Accuracy
  directional_metrics:
    enabled: true
    classes:
      up: ["+0 to +2%", "+2 to +5%", "+5 to +10%", ">+10%"]
      down: ["<-10%", "-10 to -5%", "-5 to -2%", "-2 to 0%"]
    metric: "binary_accuracy"  # Correct up/down prediction
  
  # 3. Class-Specific Analysis  
  per_class_metrics:
    enabled: true
    metrics: ["precision", "recall", "f1", "support"]
    min_support_threshold: 100  # Warn if class has < 100 samples
  
  # 4. Confidence Analysis
  confidence_metrics:
    enabled: true
    compute_entropy: true      # Prediction uncertainty
    top_k_accuracy: [1, 2, 3]  # Top-1, top-2, top-3 accuracy
```

**MLFlow Logging**:
```python
# Calibration metrics
mlflow.log_metric("ECE", expected_calibration_error)
mlflow.log_metric("MCE", maximum_calibration_error)
mlflow.log_artifact("reliability_diagram.png")
mlflow.log_artifact("calibration_curve_per_class.png")

# Calibrated model (if post-hoc enabled)
mlflow.log_param("temperature_scaling_T", T_optimal)
mlflow.tensorflow.log_model(calibrated_model, "calibrated_model")
```

**Critical Recommendation**: 
- **Keep long-term channel** for regime context ✓
- **Add post-hoc calibration** as separate step (non-negotiable for trading)
- **Monitor calibration degradation** over time (include in temporal testing §6.7)

**Your preservation of y-axis scale is correct BUT insufficient for calibration.** Both are needed.

---

## 7. Success Criteria

### 7.1 Platform Success

- Reproducible model training with full MLFlow experiment tracking
- Extensible architecture supporting new input sources with minimal code changes
- Comprehensive configuration validation preventing runtime errors
- Clear documentation and maintainability

### 7.2 Model Success

- Probability predictions that outperform baseline (e.g., random, naive persistence)
- Well-calibrated probabilities (predicted probabilities match observed frequencies)
- Generalizable across different market conditions (tested on held-out time periods)
- Actionable predictions that could inform trading strategies (evaluated by downstream applications)

---

## 8. Non-Goals (Out of Scope)

1. **Real-time inference:** This platform trains models; inference is handled elsewhere
2. **Signal generation:** Producing buy/sell signals from model outputs is a separate application
3. **Execution/Trading:** No order placement or portfolio management
4. **Live data streaming:** Training uses historical data; real-time data ingestion is out of scope
5. **Automated retraining:** No scheduled model updates; manual execution only

---

## 9. Critical Architecture Decisions Summary

All 10 key questions have been resolved. Critical decisions requiring implementation:

### 9.1 Data Pipeline (HIGH PRIORITY)
1. **Preprocessing pipeline** must be implemented FIRST
   - GreptimeDB → snapshot extraction → adaptive depth compression → `.npz` storage
   - Time alignment and gap handling is NON-NEGOTIABLE
   - Without this, training is infeasible (I/O bottleneck)

2. **Dual-channel architecture** requires careful design
   - Short-term: 24h @ 10s resolution
   - Long-term: 90-day @ 1-day resolution (statistical aggregation)
   - Merging strategy must be validated early

### 9.2 Model Architecture (MEDIUM PRIORITY)
3. **Multi-input CNN+LSTM** with two branches
   - Stacked channels for multi-asset (requires synchronized timestamps)
   - Separate CNN paths for ST and LT channels
   - Concatenation before final dense layers

4. **Calibration is mandatory**
   - Long-term channel provides regime context
   - Post-hoc calibration (temperature scaling) required for probability accuracy
   - Both are essential, complementary

### 9.3 Training Strategy (MEDIUM PRIORITY)  
5. **Optuna hyperparameter optimization** (following training-core)
   - Expanded search space for dual-channel architecture
   - Includes time-weighting half-life as hyperparameter

6. **Temporal degradation testing** validates assumptions
   - Multi-horizon test splits (recent, 6mo, 12mo, 18mo old)
   - Empirically validates exponential decay weighting

### 9.4 Infrastructure (LOW PRIORITY)
7. **MLFlow experiment structure** mirrors training-core
   - 4 experiments: preprocessing, dataset prep, hyperopt, training
   - Full lineage tracking
   - No model registry initially

## 10. Next Steps

### Phase 1: Deep Analysis of training-core (Week 1)
**Action Required**: Comprehensive code study of training-core architecture
- Focus on: MLFlow integration, Optuna setup, preprocessing patterns, podObject structure
- Extract reusable patterns
- Identify adaptations needed for Binance use case

### Phase 2: Preprocessing Pipeline Development (Weeks 2-4)
1. **GreptimeDB data extraction**
   - Query optimization for bulk retrieval
   - Batch processing by date/symbol

2. **Snapshot generation and compression**
   - Adaptive depth detection
   - Smart aggregation (high-res near market, aggregated deep)
   - `.npz` storage with compression

3. **Time alignment and quality control**
   - Multi-asset synchronization
   - Gap detection and handling
   - Quality filtering

4. **Long-term channel generation**
   - Daily statistical aggregation
   - Regime indicator computation

### Phase 3: Data Object and Configuration (Week 5)
1. **DataObject class** (analogous to podObject)
   - Schema definition
   - Serialization/deserialization
   - Validation methods

2. **YAML configuration system**
   - Complete schema definition
   - Validation at startup (fail-fast)
   - Environment variable handling

### Phase 4: Model Architecture (Weeks 6-7)
1. **Dual-channel CNN+LSTM implementation**
   - Short-term branch
   - Long-term branch  
   - Merging strategy

2. **Custom layers and metrics**
   - Temporal encoding layers
   - Calibration metrics
   - Multi-class loss with sample weighting

### Phase 5: Training Pipeline (Weeks 8-9)
1. **Training loop with MLFlow**
   - Experiment tracking
   - Artifact logging
   - Lineage tracking

2. **Optuna integration**
   - Search space configuration
   - Nested runs
   - Pruning strategies

### Phase 6: Evaluation and Calibration (Week 10)
1. **Metrics suite**
   - Standard classification metrics
   - Calibration analysis
   - Temporal degradation testing

2. **Post-hoc calibration**
   - Temperature scaling implementation
   - Calibration validation

### Phase 7: Testing and Documentation (Weeks 11-12)
1. **Integration testing** with small dataset
2. **User and developer documentation**
3. **Configuration examples and templates**

## 11. Critical Risks and Mitigations

### Risk 1: Data Quality Issues
**Risk**: Gaps, misalignments in GreptimeDB data corrupt training
**Mitigation**: Robust preprocessing with quality metrics logged to MLFlow

### Risk 2: Memory Constraints
**Risk**: Large tensors exceed GPU memory (2x RTX 3090 = 48GB total)
**Mitigation**: Adaptive depth compression, batch size tuning, gradient checkpointing

### Risk 3: Training Time
**Risk**: Hyperparameter search takes weeks
**Mitigation**: Early stopping, Optuna pruning, parallel trials across GPUs

### Risk 4: Poor Calibration
**Risk**: Model probabilities unusable for trading decisions
**Mitigation**: Mandatory calibration analysis + post-hoc correction, monitored per-class

### Risk 5: Temporal Overfitting
**Risk**: Model only works on recent data, fails on older test sets
**Mitigation**: Expanding-window CV, temporal degradation testing, adaptive time weighting

---

## 12. References

### Code References
- **training-core:** `/Users/obenomar/Trade/BinanceAlgo/Pipeline-MVP/training-core`
  - **Model:** `training-core/ML/strategy_2DCNNLSTM.py` (architecture pattern)
  - **Config:** `training-core/config/configAnomalies.yaml` (YAML structure)
  - **MLFlow:** `training-core/Common/ioMLFlowArtifacts.py` (experiment tracking)
  - **Preprocessing:** `training-core/PreProc/DefineMLsets.py` (dataset preparation)
  - **Main:** `training-core/main.py` (pipeline orchestration)

- **Data Infrastructure:**
  - **Fetching:** `python-datafeed-v2/v2/depth_channel_v2_greptime.py` (GreptimeDB schema)
  - **Quality:** `python-datafeed-v2/v2/quality_index_builder.py` (data access patterns)

### Technical Specifications
- Companion document: `technical-specifications.md` (to be updated based on this vision)

---

## Document Change Log

**Version 0.3 (2025-11-03 23:00 UTC+09:00)**
- **Clarified NPZ storage strategy**: Batch-by-batch temporary cache, not persistent storage
  - Process 1 day → cache → use → discard (keep 1% for diagnostics)
  - Storage: ~5-10 GB rolling cache (not full TB)
  - Long-term aggregations: ~20-50 GB persistent
- **Fixed time-weighting redundancy**: Removed sampling_strategy, kept only sample_weighting
  - Single approach: deterministic weighting in loss function
  - Preserves full dataset, tunable as hyperparameter
- **Enhanced regime detection**: Multi-scale time weighting + regime-aware training
  - Multiple half-lives (7d, 30d, 90d) for different patterns
  - Regime detection via long-term channel (bull/bear/sideways/high-vol)
  - Option: regime as feature OR separate models per regime
- **Made rolling window CV configurable**: YAML boolean option
- Status: **Finalized - Ready for Implementation**

**Version 0.2 (2025-11-03 20:58 UTC+09:00)**
- Resolved all 10 key open questions with user input
- Added critical analysis and constructive criticism
- Identified preprocessing as highest priority (I/O bottleneck)
- Designed dual-channel architecture for multi-scale temporal modeling
- Clarified calibration strategy (regime context + post-hoc correction)
- Established temporal degradation testing methodology
- Defined MLFlow experiment hierarchy
- Created detailed implementation roadmap
- Status: **Ready for Implementation Planning**

**Version 0.1 (2025-11-03 17:53 UTC+09:00)**
- Initial draft with 10 open questions
- High-level architecture and design principles
- Status: Draft for Review

---

**End of Vision Document**
