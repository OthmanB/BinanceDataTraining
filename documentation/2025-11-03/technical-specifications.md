# Binance ML Training Platform - Technical Specifications

**Date:** 2025-11-15  
**Time:** 22:45 UTC+09:00  
**Version:** 0.4 - Implementation-Aligned Refinements  
**Status:** In Sync with Current Implementation

---

## Change Log

**Version 0.4 (2025-11-15 22:45 UTC+09:00)**
- ✅ Clarified snapshot construction for targets: snapshots are grouped strictly by timestamp `ts` (database `batch_id` is treated as a coarse time bucket and is not used as a unique identifier).
- ✅ Introduced snapshot-level top-of-book features (`snapshot_features`) for the target asset, aligned 1:1 with target labels.
- ✅ Formalized the hybrid filtering rule: coarse validity filters (e.g., `bid_price > 0 AND ask_price > 0`) run inside GreptimeDB, while complex quality checks and feature construction run on the training machine.
- ✅ Added a pre-training data diagnostics stage that operates on the ML-ready snapshots and logs spread-related metrics and artifacts to MLflow.

**Version 0.3 (2025-11-03 23:16 UTC+09:00)**
- ✅ Added preprocessing pipeline architecture (§2: batch-by-batch NPZ caching)
- ✅ Updated to dual-channel (short-term + long-term) model architecture (§4)
- ✅ Added time-weighted sampling configuration (§5.3)
- ✅ Added regime detection configuration (§5.4)
- ✅ Updated temporal encoding to cyclical (§3.3)
- ✅ Added post-hoc calibration (temperature scaling) (§8.2)
- ✅ Updated storage strategy to NPZ batch caching (§3.4)
- ✅ Added rolling window validation for Optuna (§6.2)
- ✅ Added temporal degradation test splits (§5.6)
- ✅ Resolved DataObject design to dataclass (§3.1)
- ✅ Updated module structure with preprocessing modules (§9)
- ✅ Added GreptimeDB environment variables (§10.1)
- Aligned with vision.md v0.3 decisions
- Removed all "QUESTION" tags - all decisions finalized

**Version 0.1 (2025-11-03 17:53 UTC+09:00)**
- Initial draft with open questions

---

## 1. System Architecture

### 1.1 High-Level Component Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Binance ML Training Platform                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐             │
│  │ Data Loader  │───▶│ Data Object   │───▶│ Preprocessing   │             │
│  │              │    │ Builder       │    │ Pipeline        │             │
│  │ - Order Book │    │ - Model-agn.  │    │ - Gap/NaN       │             │
│  │ - Timestamps │    │ - Multi-Asset │    │ - Time Align    │             │
│  │ - External   │    │ - Time Series │    │ - Sample Slicing│             │
│  └──────────────┘    └────────┬──────┘    └────────┬───────┘             │
│                               │                    │                     │
│                               ▼                    ▼                     │
│                     ┌──────────────────────┐   ┌──────────────────────┐  │
│                     │ Model-Specific       │   │ Model Training        │  │
│                     │ Feature Pipelines    │   │ Strategies            │  │
│                     │ (per model family)   │   │ (per model family)    │  │
│                     └──────────┬───────────┘   └──────────┬───────────┘  │
│                                │                          │              │
│                                ▼                          ▼              │
│                     ┌──────────────────────────────┐                     │
│                     │   MLFlow Experiments         │                     │
│                     │   (per stage + per model)    │                     │
│                     └──────────────────────────────┘                     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ Pipeline Orchestrator (reads YAML, selects functions per stage,     │ │
│  │ enforces ordering, and routes logs to MLFlow experiments)          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

#### Data Loader
- Load historical order book data from source (database/files)
- Parse and structure multi-asset time series
- Handle missing data and quality checks
- Extract timestamp information (local/global features)
- Interface with external data sources (when added)

#### Preprocessor (Model-Agnostic Pipeline)
- Validate data integrity and schema
- Apply general transformations (normalization, scaling)
- Perform gap interpolation, NaN handling, and time alignment
- Slice data into training samples
- Create temporal embeddings
- Prepare train/validation/test splits

#### Data Object
- Unified data container (inspired by podObject from training-core)
- Strictly model-agnostic structure shared by all downstream models
- Multi-asset order book representation
- Temporal metadata
- Configuration-driven structure
- Serialization for caching

#### Model-Specific Feature Pipelines
- Convert the generic `DataObject` into tensors suitable for a given model family (e.g., CNN+LSTM, Transformer)
- Apply model-specific feature engineering and reshaping
- Be fully configurable via YAML by function name
- Log dataset/feature preparation artifacts and metrics to dedicated MLFlow experiments

#### Model Training Strategies
- Implement pluggable training strategies identified by a model key
- First canonical implementation: CNN+LSTM multi-class classifier
- Optionally run hyperparameter optimization (Optuna) per model
- Train with callbacks (early stopping, etc.)
- Evaluate on test sets and log metrics/artifacts
- Compute model complexity metrics (FLOPs, parameters)

#### Pipeline Orchestrator
- Read YAML configuration and resolve stage and model function names
- Orchestrate the full workflow: data collection → DataObject construction → preprocessing → feature pipeline → training
- Ensure that each stage logs to its designated MLFlow experiment and that run IDs are linked for lineage
- Enforce configuration validation before executing stages

#### MLFlow Integration
- Provide helpers for starting/ending runs per stage
- Centralize parameter and metric logging
- Store model and dataset artifacts
- Maintain dataset and feature pipeline versioning
- Maintain lineage and provenance across stages and model implementations

---

## 2. Data Schema and Structures

### 2.1 Data Object Structure

**Core Abstraction:** `DataObject` (replaces podObject from training-core)

```python
DataObject = {
    'metadata': {
        'asset_pairs': List[str],           # e.g., ['BTCUSDT', 'ETHUSDT', ...]
        'time_range': {
            'start': datetime,
            'end': datetime,
            'cadence_seconds': int          # e.g., 10
        },
        'num_samples': int,
        'order_book_depth': int,            # e.g., 1000
        'version': str                      # Dataset version
    },
    'order_books': {
        'BTCUSDT': {
            # Implementation note (v0.4): the initial implementation uses
            # GreptimeDB "tall" rows plus snapshot-level top-of-book features.
            #
            # - 'rows': raw GreptimeDB rows for the symbol
            # - 'snapshot_features': List[[best_bid_price, best_bid_qty,
            #                             best_ask_price, best_ask_qty]]
            # - 'snapshot_timestamps': List[ts] aligned with snapshot_features
            #
            # This is compatible with, and can be extended towards, a full
            # depth tensor representation when needed.
            'rows': List[List[Any]],
            'snapshot_features': List[List[float]],
            'snapshot_timestamps': List[Any],
        },
        # ... other asset pairs
    },
    'temporal_features': {
        'local': np.ndarray,                 # (N, local_feature_dim)
        'global': np.ndarray                 # (N, global_feature_dim)
    },
    'targets': {
        'asset': str,                        # Asset being predicted
        'labels': np.ndarray,                # (N,) class labels
        'price_changes': np.ndarray,         # (N,) actual price changes
        'delta_t_seconds': int               # Prediction horizon
    },
    'external_data': {
        # Extensibility placeholder
    }
}
```

**Decision:** `DataObject` is represented as a Python dictionary with a strict, validated schema.

- Flexible and easy to extend with new fields as the platform evolves
- Simple to serialize/deserialize for caching and debugging
- Consistent with the dictionary-based pattern already used in `training-core`

**Future Option:** If stronger typing is needed, a thin `DataObject` class or `TypedDict` wrapper can be introduced around the dictionary without changing the external format or on-disk representation.

---

### 2.2 Configuration Schema

**File:** `config/training_config.yaml`

```yaml
# Data Source Configuration
data:
  source_type: "database"  # or "file"
  connection:
    database_uri: "${DATABASE_URI}"  # Environment variable
    table_name: "order_book_snapshots"
  
  asset_pairs:
    target_asset: "BTCUSDT"
    correlated_assets: ["ETHUSDT", "BNBUSDT", "ADAUSDT", "ETHBTC", "BNBBTC", "ADABTC"]
  
  time_range:
    start_date: "2022-01-01"
    end_date: "2024-12-31"
    cadence_seconds: 10
  
  order_book:
    depth_levels: 1000
    representation: "full"  # QUESTION: "full", "aggregated", or "hybrid"
  
  temporal_features:
    local:
      - "hour_of_day"
      - "day_of_week"
      - "minute_of_hour"
    global:
      - "days_since_start"
      - "market_session"  # Asian/European/American
  
  validation:
    check_missing_data: true
    max_gap_seconds: 60
    fail_on_invalid: true

# Target Configuration
targets:
  prediction_horizon_seconds: 300  # QUESTION: What is optimal Δt?
  
  price_classes:
    definition_type: "percentage"  # or "absolute", "adaptive"
    boundaries: [-10, -5, -2, 0, 2, 5, 10]  # Defines class ranges
    # Results in classes: (<-10%, -10 to -5%, -5 to -2%, -2 to 0%, 0 to 2%, 2 to 5%, 5 to 10%, >10%)
  
  labeling:
    use_midpoint: true  # Use mid-price or last trade price
    handle_gaps: "interpolate"  # or "skip", "forward_fill"

# Preprocessing Configuration
preprocessing:
  normalization:
    method: "min_max"  # or "standard", "robust"
    per_asset: true
    fit_on_train_only: true
  
  feature_engineering:
    order_book_features:
      - "bid_ask_spread"
      - "volume_imbalance"
      - "depth_imbalance"
      - "weighted_mid_price"
    
    derived_features:
      - "price_momentum"
      - "volume_momentum"
  
  train_test_split:
    method: "chronological"  # QUESTION: or "walk_forward"
    train_ratio: 0.70
    validation_ratio: 0.15
    test_ratio: 0.15
  
  class_balancing:
    enabled: true
    method: "class_weights"  # or "oversampling", "undersampling"

# Model Architecture Configuration
model:
  framework: "keras"  # tensorflow.keras
  backend: "tensorflow"
  architecture: "CNN_LSTM_MultiClass"
  
  input_representation:
    strategy: "stacked_channels"  # QUESTION: or "separate_branches", "attention"
  
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
  
  dense:
    layers: [64]
    dropout_rates: [0.2]
  
  output:
    num_classes: 8  # Based on price_classes definition
    activation: "softmax"
  
  compilation:
    optimizer: "adam"
    learning_rate: 0.001
    loss: "categorical_crossentropy"
    metrics: ["accuracy", "precision", "recall"]

# Training Configuration
training:
  epochs: 50
  batch_size: 32
  validation_split: 0.0  # Using explicit validation set from preprocessing
  
  callbacks:
    early_stopping:
      enabled: true
      monitor: "val_loss"
      patience: 10
      restore_best_weights: true
    
    reduce_lr:
      enabled: true
      monitor: "val_loss"
      factor: 0.5
      patience: 5
      min_lr: 0.00001
  
  class_weights:
    compute_from_train: true

  fine_tuning:
    enabled: false              # When true, resume from a previous model instead of training from scratch
    base_model_run_id: null     # Required when enabled: MLFlow run_id or model identifier to load
    base_model_stage: "binance-model-training"  # MLFlow experiment/stage where the base model is stored

# Hyperparameter Optimization Configuration
hyperparameter_optimization:
  enabled: false  # QUESTION: Include or not?
  framework: "optuna"
  n_trials: 30
  direction: "maximize"
  metric: "val_accuracy"
  
  search_space:
    cnn_filters_1: [16, 128]
    cnn_filters_2: [64, 256]
    lstm_units: [32, 256]
    learning_rate: [0.0001, 0.01]
    batch_size: [16, 32, 64]

# MLFlow Configuration
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"  # Environment variable
  experiment_name: "binance-price-prediction"
  
  run_naming:
    pattern: "{asset}_{model}_{timestamp}"
  
  artifact_logging:
    model_architecture_plot: true
    training_plots: true
    confusion_matrix: true
    class_distribution: true
    feature_importance: false  # For future
  
  model_registry:
    register_model: true
    model_name_pattern: "{asset}_predictor"

# Evaluation Configuration
evaluation:
  metrics:
    - "accuracy"
    - "precision_per_class"
    - "recall_per_class"
    - "f1_per_class"
    - "confusion_matrix"
    - "calibration_error"  # QUESTION: Include advanced metrics?
  
  calibration_analysis:
    enabled: true
    n_bins: 10
  
  backtesting:
    enabled: false  # QUESTION: Include simulated P&L?
    initial_capital: 10000
    transaction_cost: 0.001

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  colored_output: true
  log_function_names: true
  
  colors:
    function_names: "cyan"
    parameter_names: "yellow"
    parameter_values: "green"
    info: "white"
    warning: "yellow"
    error: "red"
    debug: "blue"

# Security Configuration
security:
  environment_variables:
    - "DATABASE_URI"
    - "MLFLOW_TRACKING_URI"
    - "MLFLOW_TRACKING_USERNAME"
    - "MLFLOW_TRACKING_PASSWORD"
  
  validation:
    check_env_vars_at_startup: true
    fail_if_missing: true

### 2.3 Configuration Snapshot and Reproducibility

To guarantee consistent preprocessing, target definition, temporal encoding, and model settings across base and fine-tuning runs, the pipeline uses configuration snapshots:

- At the start of each training run, the **effective configuration** (after environment-variable resolution and validation) is saved as an artifact in MLFlow (e.g., `training_config.yaml` under the corresponding experiment run).
- The saved configuration snapshot represents the exact parameters and options used to generate datasets, build the model, and train it.
- When performing **fine-tuning**, the pipeline:
  - Loads the configuration snapshot from the **base model run** identified by `training.fine_tuning.base_model_run_id`.
  - Applies a **restricted set of overrides** coming from the current YAML (e.g., `data.source_name`, fine-tuning time range, and `training.fine_tuning` block).
  - Validates the merged configuration before executing the fine-tuning pipeline.

This mechanism ensures that base and fine-tuned models share identical data representation choices unless explicitly and safely overridden, and that every run remains fully reproducible from its MLFlow artifacts.

---
## 3. Module Structure

```
BinanceBot/
├── config/
│   ├── training_config.yaml
│   └── validation_schema.yaml
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # Load order book data
│   ├── data_object.py          # DataObject class definition
│   └── external_sources.py     # Extensibility for external data
├── preprocessing/
│   ├── __init__.py
│   ├── validator.py            # Data validation
│   ├── transformer.py          # Normalization, feature engineering
│   ├── temporal_features.py    # Time-based feature extraction
│   └── train_test_split.py     # Dataset splitting logic
├── diagnostics/
│   ├── __init__.py
│   └── data_diagnostics.py     # Pre-training data quality diagnostics
├── models/
│   ├── __init__.py
│   ├── cnn_lstm_multiclass.py  # Main model architecture
│   ├── layers.py               # Custom layers (e.g., MaxRescaling)
│   └── hyperparameter_tuning.py # Optuna integration
├── training/
│   ├── __init__.py
│   ├── pipeline.py             # Main training pipeline
│   ├── callbacks.py            # Custom callbacks
│   └── metrics.py              # Custom metrics
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py            # Model evaluation
│   ├── calibration.py          # Calibration analysis
│   └── visualization.py        # Plotting utilities
├── mlflow_integration/
│   ├── __init__.py
│   ├── experiment_tracker.py   # MLFlow logging
│   ├── model_registry.py       # Model versioning
│   └── artifact_manager.py     # Artifact handling
├── utils/
│   ├── __init__.py
│   ├── config_loader.py        # YAML config loading and validation
│   ├── colored_logging.py      # Colored logging setup
│   └── env_validator.py        # Environment variable validation
├── main.py                     # Entry point
└── README.md
```

---

## 4. Key Technical Decisions Pending

### 4.1 Model Input Representation

**Decision Required:** How to structure multi-asset order book inputs?

**Options:**
- **Stacked Channels:** Single tensor with asset order books as channels
- **Separate Branches:** Multi-input model with per-asset CNN branches
- **Attention Mechanism:** Dynamic asset weighting

**Impact:** Model architecture complexity, parameter count, training time

---

### 4.2 Order Book Tensor Shape

**Decision Required:** How to represent order book depth?

**Current Assumption:** Full 1000-level depth

**Options:**
- Full: `(N_samples, N_timesteps, 1000, 4, N_assets)` → Very large
- Aggregated: `(N_samples, N_timesteps, 50, 4, N_assets)` → Reduced
- Hybrid: High-res near market + low-res deep levels

**Impact:** Memory usage, training speed, information preservation

---

### 4.3 Prediction Horizon

**Decision Required:** Target Δt for price prediction?

**Options:** 1min, 5min, 15min, 1hour, multi-horizon

**Impact:** Model use case, data volume, label distribution

---

### 4.4 Class Definition Strategy

**Decision Required:** How to define price range classes?

**Current Proposal:** Fixed percentage ranges

**Alternatives:**
- Adaptive classes based on volatility
- Asset-specific class definitions
- Dynamic boundaries

**Impact:** Model performance, generalization, actionability

---

### 4.5 Hyperparameter Optimization Inclusion

**Decision Required:** Include Optuna-based optimization?

**Considerations:**
- Training cost: Large models + large data = expensive
- Benefit: Automated tuning vs manual effort

**Options:**
- Full Optuna integration (like training-core)
- Manual tuning only
- Optional flag in config

---

### 4.6 Data Storage and Versioning

**Decision Required:** Where and how to store/version large datasets?

**Options:**
- MLFlow artifacts (simple but potentially expensive for large data)
- DVC for data versioning (separate system)
- Custom database with snapshots
- Hybrid approach

**Impact:** Reproducibility, storage costs, complexity

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Project structure setup
- Configuration loading and validation
- Logging infrastructure
- Environment setup (MLFlow, dependencies)

### Phase 2: Data Pipeline (Weeks 3-4)
- DataObject implementation
- Data loader for order book data
- Preprocessing and validation
- Train/test split logic

### Phase 3: Model Development (Weeks 5-6)
- CNN+LSTM architecture implementation (Keras/TensorFlow)
- Custom layers using tf.keras.layers.Layer
- Loss functions and metrics
- Training pipeline with tf.keras.Model
- MLFlow integration (mlflow.tensorflow.autolog)

### Phase 4: Evaluation and Optimization (Weeks 7-8)
- Evaluation metrics
- Visualization tools
- Hyperparameter tuning (if included)
- Model registry integration

### Phase 5: Testing and Documentation (Weeks 9-10)
- Unit tests
- Integration tests
- User documentation
- API documentation

---

## 6. Dependencies

### Core Dependencies
```
tensorflow>=2.13.0  # Includes Keras API (tensorflow.keras)
numpy>=1.24.0
pandas>=2.0.0
mlflow>=2.8.0
optuna>=3.4.0
pyyaml>=6.0
termcolor>=2.3.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### Framework: Keras with TensorFlow Backend

**Architecture Framework**: Keras (via `tensorflow.keras`)

**Rationale**:
- High-level API for rapid development
- Seamless integration with TensorFlow 2.x
- Native support for custom layers, metrics, and losses
- Excellent MLFlow integration
- Follows training-core pattern

**Import Convention**:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Model definition
model = keras.Sequential([
    layers.Conv2D(...),
    layers.LSTM(...),
    layers.Dense(...)
])
```

### Optional Dependencies
```
dvc>=3.0.0  # If using DVC for data versioning
```

---

## 7. Security Considerations

### Environment Variables (Required)
```bash
# MLFlow Authentication
export MLFLOW_TRACKING_USERNAME="<username>"
export MLFLOW_TRACKING_PASSWORD="<password>"
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"

# Database Connection (if applicable)
export DATABASE_URI="postgresql://user:pass@host:port/dbname"
```

### Validation at Startup
- Check all required environment variables are set
- Fail fast if any are missing
- Never hardcode credentials

---

## 8. Testing Strategy

### Unit Tests
- Configuration loading and validation
- Data transformations
- Feature engineering
- Model architecture building

### Integration Tests
- End-to-end pipeline with small dataset
- MLFlow logging verification
- Data loader → preprocessing → training flow

### Model Tests
- Architecture correctness (input/output shapes)
- Forward pass with dummy data
- Gradient computation

---

## 9. Monitoring and Logging

### Training Logs
- Epoch-level metrics (loss, accuracy)
- Batch-level progress (optional, DEBUG mode)
- Configuration parameters
- Data statistics

### MLFlow Metrics
- Training/validation loss and accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Calibration metrics
- Model complexity (parameters, FLOPs)
- Data diagnostics metrics (e.g., spread statistics, fraction of suspicious spreads)

### Artifacts
- Trained model
- Training plots (loss, accuracy)
- Confusion matrix heatmap
- Model architecture diagram
- Configuration file used
- Dataset version/hash
- Data diagnostics artifacts (e.g., sampled CSVs, future plots/heatmaps derived from snapshot_features)

---

## 10. Technical Design Notes and Open Questions

1. **Memory Management**  
   **Scope:** Managing memory both at the training stage (GPU/CPU) and during preprocessing/dataset construction.  
   **Decision (v1):**  
   - At training time, use mini-batch training with batch size treated as a hyperparameter defined in YAML.  
   - During preprocessing and dataset construction, control chunk sizes (e.g., number of days per run, number of samples per cached file) via YAML parameters.  
   - For large datasets, require streaming or chunked dataset loading (e.g., via `tf.data` pipelines or equivalent generators) so that host RAM and device memory usage remain bounded by explicit YAML configuration parameters (for example, a future `training.memory` and `evaluation.memory` section). If a projected allocation would exceed these budgets, the pipeline must fail fast with a clear error instead of relying on operating-system swap or uncontrolled out-of-memory conditions.  
   **Open Point:** Future versions may introduce auto-tuning of batch size, chunk size, and streaming buffer sizes based on available physical memory and configured budgets.  

2. **GPU Utilization**  
   **Decision (v1):** Target CPU and single-GPU training only. Multi-GPU or distributed training strategies are considered future enhancements. Early tests may run entirely on CPU, with appropriately small batch sizes and model configurations.  

3. **Incremental Training (Fine-Tuning)**  
   **Decision (v1):** Incremental training / fine-tuning from an existing model is supported as an optional mode. The training pipeline can restart from previously trained weights and refine them on new datasets instead of always training from scratch.  
   **Mechanism:**  
   - Fine-tuning is controlled via the `training.fine_tuning` section in the YAML configuration.  
   - When `fine_tuning.enabled: true`, the pipeline first retrieves the configuration snapshot (`training_config.yaml`) stored as an artifact of the base model run. This snapshot defines the effective preprocessing, target definition, temporal encoding, and model settings used for the base training.  
   - The pipeline then applies a **restricted set of overrides** from the current YAML (typically `data.source_name`, `data.time_range.start_date`, `data.time_range.end_date`, and `training.fine_tuning` itself).  
   - The merged configuration is validated and becomes the effective configuration for the fine-tuning run, ensuring representation consistency between base and fine-tuned models.  
   - All parameters in `training.fine_tuning` must be provided when fine-tuning is enabled.  
   **Use Cases:**  
   - Training on very long historical datasets (e.g., 5 years at 10-second cadence) where full retraining is infeasible.  
   - Adapting to changing market regimes by fine-tuning on more recent segments, potentially triggered by a higher-level regime detection algorithm operating on coarser data.  

4. **Model Ensemble**  
   **Decision (v1):** Ensemble methods (e.g., combining multiple models or architectures) are not implemented in the initial version. The architecture is designed so that a future ensemble stage can consume predictions from multiple model runs (e.g., via MLFlow artifacts) and produce aggregated outputs.  
   **Open Point:** Decide on ensemble strategies (e.g., simple averaging of probabilities, weighted averaging based on validation performance, or more advanced meta-models).  

5. **Explainability**  
   **Decision (v1):** Include diagnostic tools to understand what the model learns.  
   **Initial Scope:**  
   - Saliency or activation visualizations over temporal/depth/asset dimensions for CNN+LSTM-based models.  
   - Basic importance-style diagnostics for key input channels or engineered features.  
   - Logging of all diagnostics as artifacts in MLFlow for later analysis.  
   **Future:** Deeper integration with explainability frameworks (e.g., SHAP, attention visualizations) as more complex model families are introduced.  

---

**End of Technical Specifications Document**
