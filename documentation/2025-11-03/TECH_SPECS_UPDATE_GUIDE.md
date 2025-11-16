# Technical Specifications v0.3 Update Guide

**Purpose**: Guide for updating technical-specifications.md from v0.1 → v0.3

**Status**: implementation-review.md documents all resolutions. This guide provides the concrete updates needed.

---

## OPTION 1: Use Vision.md as Primary Reference (RECOMMENDED)

**Recommendation**: Since vision.md v0.3 is comprehensive and finalized, use it as the authoritative architectural reference during implementation.

**Why**: 
- Vision.md v0.3 contains all finalized decisions
- Technical specs would essentially duplicate vision content
- Reduces maintenance burden (single source of truth)
- Vision.md is detailed enough for implementation

**Approach**:
1. Use vision.md v0.3 for architectural decisions
2. Use current technical-specifications.md v0.1 for:
   - Module structure (§3)
   - Dependencies (§6)  
   - Testing strategy (§8)
   - Logging/monitoring (§9)
3. Refer to implementation-review.md for specific configurations

---

## OPTION 2: Full Technical Specs Update (If Needed)

If you prefer a standalone technical specs document, here are the key sections to add/update:

### Section Updates Required

**§1 - Add to System Architecture**:
```markdown
### 1.3 Preprocessing Pipeline

GreptimeDB → Daily Extraction → Adaptive Compression → Temporary NPZ Cache
                                                              ↓
                                                    Training Sample Generation
                                                              ↓
                                                         GPU Training
                                                              ↓
                                                      Cache Cleanup (keep 1%)

Long-term: Daily Statistics → Persistent Storage (20-50 GB)
```

**§2 - NEW: Preprocessing Pipeline Architecture**:
- Copy from vision.md §6.2
- Add module specifications
- Add configuration YAML

**§3 - Update DataObject**:
```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class DataObject:
    """Unified data container for Binance order book data."""
    
    # Metadata
    asset_pairs: List[str]
    time_range: Dict
    
    # Short-term data (24h @ 10s)
    short_term_data: Dict[str, np.ndarray]
    
    # Long-term data (90d @ 1d)
    long_term_data: Dict[str, np.ndarray]
    
    # Temporal features (cyclical, trend, volatility)
    temporal_features: Dict[str, np.ndarray]
    
    # Targets
    targets: Dict
    
    # Preprocessing metadata
    preprocessing_metadata: Dict
    
    def validate(self): ...
    def to_dict(self): ...
    @classmethod
    def from_dict(cls, data): ...
```

**§4 - Update Model Architecture to Dual-Channel**:
```yaml
model:
  architecture: "DualChannel_CNN_LSTM_MultiClass"
  
  short_term_channel:
    input_shape: [8640, 100, 4, 7]  # (timesteps, depth, features, assets)
    cnn:
      num_layers: 2
      filters: [64, 128]
      kernel_sizes: [[3, 3], [3, 3]]
      pooling: "max"
      pool_sizes: [[2, 2], [2, 2]]
      activation: "relu"
      dropout_rates: [0.2, 0.3]
    lstm:
      units: 96
      dropout: 0.3
      recurrent_dropout: 0.2
  
  long_term_channel:
    input_shape: [90, 5, 7]  # (days, stats, assets)
    cnn:
      num_layers: 1
      filters: [32]
      kernel_sizes: [[3, 1]]
      pooling: "max"
      pool_sizes: [[2, 1]]
      activation: "relu"
      dropout: 0.2
    dense:
      units: 64
      dropout: 0.2
  
  merge_strategy: "concatenate"
  
  final_dense:
    layers: [64]
    dropout_rates: [0.2]
  
  output:
    num_classes: 8
    activation: "softmax"
```

**§5 - Add Training Configurations**:

**§5.3 - Time-Weighted Sampling**:
```yaml
training:
  sample_weighting:
    enabled: true
    method: "exponential_decay"
    half_life_days: 90  # Tunable hyperparameter
    apply_to: "loss_function"
  
  multi_scale_weighting:
    enabled: false  # Optional
    time_scales:
      - {name: "short_term_memory", half_life_days: 7}
      - {name: "medium_term_memory", half_life_days: 30}
      - {name: "long_term_memory", half_life_days: 90}
```

**§5.4 - Regime Detection**:
```yaml
training:
  regime_detection:
    enabled: true
    source: "long_term_channel"
    regimes: ["bull", "bear", "sideways", "high_volatility"]
    strategy: "regime_as_feature"
    regime_as_feature:
      encoding: "one_hot"
```

**§5.6 - Temporal Degradation Test Splits**:
```yaml
preprocessing:
  test_splits:
    recent:
      start_date: "2024-10-01"
      end_date: "2024-12-31"
    moderate_age:
      start_date: "2024-04-01"
      end_date: "2024-06-30"
    old:
      start_date: "2023-10-01"
      end_date: "2023-12-31"
    very_old:
      start_date: "2023-04-01"
      end_date: "2023-06-30"
```

**§6.2 - Add Rolling Window Validation**:
```yaml
hyperparameter_optimization:
  enabled: true
  framework: "optuna"
  
  rolling_window_validation:
    enabled: true  # YAML-configurable boolean
    n_windows: 3
    window_expansion_months: 2
```

**§8 - Update Evaluation**:

**§8.2 - Post-Hoc Calibration**:
```yaml
evaluation:
  calibration:
    enabled: true
    
    metrics:
      - "expected_calibration_error"
      - "maximum_calibration_error"
      - "brier_score"
    
    plots:
      - "reliability_diagram"
      - "confidence_histogram"
    
    post_hoc:
      enabled: true
      method: "temperature_scaling"
      validation_set: "separate"
```

**§9 - Update Module Structure**:
Add preprocessing submodules:
```
BinanceBot/
├── preprocessing/
│   ├── greptime_extraction/
│   │   ├── __init__.py
│   │   ├── query_builder.py
│   │   └── batch_extractor.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── adaptive_depth.py
│   │   └── npz_manager.py
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── time_aligner.py
│   │   └── gap_handler.py
│   ├── aggregation/
│   │   ├── __init__.py
│   │   └── long_term_stats.py
│   ├── temporal_features.py     # Cyclical encoding
│   └── regime_classifier.py     # Regime detection
```

**§10 - Update Environment Variables**:
```yaml
security:
  environment_variables:
    # GreptimeDB
    - "GREPTIME_HOST"
    - "GREPTIME_PORT"
    - "GREPTIME_DATABASE"
    - "GREPTIME_USER"
    - "GREPTIME_PASSWORD"
    
    # MLFlow
    - "MLFLOW_TRACKING_URI"
    - "MLFLOW_TRACKING_USERNAME"
    - "MLFLOW_TRACKING_PASSWORD"
    
    # Paths
    - "CACHE_DIR"
    - "DATA_DIR"
    - "LOG_DIR"
```

**§3.3 - Update Temporal Features**:
```yaml
data:
  temporal_features:
    cyclical:
      - name: "hour_of_day"
        encoding: "sin_cos"  # → 2 features
        period: 24
      - name: "day_of_week"
        encoding: "sin_cos"  # → 2 features
        period: 7
    
    trend:
      - name: "days_since_start"
        encoding: "normalized_linear"
      - name: "market_regime"
        encoding: "from_long_term_channel"
    
    volatility_context:
      - name: "rolling_vol_24h"
        encoding: "normalized"
      - name: "volume_percentile"
        encoding: "percentile_rank"
```

**§3.4 - Update Storage Strategy**:
```yaml
data:
  source:
    type: "greptime_db"
    connection:
      host: "${GREPTIME_HOST}"
      port: "${GREPTIME_PORT}"
      database: "${GREPTIME_DATABASE}"
      user: "${GREPTIME_USER}"
      password: "${GREPTIME_PASSWORD}"
    table_pattern: "orderbook_{symbol}"
  
  preprocessing_cache:
    type: "temporary_npz"
    cache_dir: "${CACHE_DIR}/daily_batches"
    retention_policy: "discard_after_use"
    diagnostic_retention_fraction: 0.01
    max_cache_size_gb: 10
  
  long_term_storage:
    type: "persistent_npz"
    storage_dir: "${DATA_DIR}/long_term_aggregations"
    estimated_size_gb: 50

# Storage Estimates
# - GreptimeDB: 1-2 TB (immutable, source of truth)
# - Temporary cache: 5-10 GB (rolling, 1-2 days)
# - Long-term aggregations: 20-50 GB (persistent)
# - MLFlow diagnostic samples: 1-2 GB (random 1%)
```

---

## Summary

**Recommendation**: Use vision.md v0.3 as primary reference + implementation-review.md for specifics.

**If full tech specs update needed**: Apply sections above to technical-specifications.md.

**All resolutions documented in**: `implementation-review.md`

---

**Status**: ✅ All issues resolved. Ready for implementation.
