# Richer Feature Representation Plan

*Created: 2025-11-17 22:40 (local time)*

## 0. Goals and constraints

- **Goal**
  - Replace the current minimal 2×2 top-of-book patch with a richer, more expressive feature representation for the CNN+LSTM model.
  - Preserve chronological splitting and all existing C2 two-headed label semantics.
- **Constraints**
  - All hyperparameters and structural choices must be configurable via YAML.
  - No hard-coded defaults in code; missing config keys must raise clear errors.
  - Logging must identify the module/function and follow existing logging style.
  - Changes should be incremental and testable in isolation.

## 1. Feature-space design

### 1.1. Levels and sides

- **Decide which order-book levels to include** (config-driven):
  - `num_levels`: number of bid and ask levels to encode (e.g. 1, 5, 10).
  - For each level `L`, include at least: price, quantity.
- **Decide on sides**:
  - Encode `bid` and `ask` separately.
  - Optional future extension: include `mid`, `spread` as derived features.

### 1.2. Feature types per level

- Candidate per-level raw features (each configurable on/off):
  - `price`: absolute price or normalized price.
  - `quantity`: order size at this level.
  - `price_delta_mid`: price minus mid-price.
  - `cumulative_qty`: cumulative quantity up to this level.
  - `relative_depth`: cumulative qty / total depth in book.
- Derived scalar features at snapshot level:
  - `spread`: best ask - best bid.
  - `mid_price`: (best ask + best bid)/2.
  - `imbalance`: (bid_qty - ask_qty)/(bid_qty + ask_qty).
- Represent feature selection in YAML, e.g.:

  ```yaml
  data:
    snapshot_features:
      levels: 10
      per_level_features: ["price_delta_mid", "quantity", "cumulative_qty"]
      aggregate_features: ["spread", "imbalance"]
      normalization: "zscore"  # or "minmax", "none"
  ```

### 1.3. Normalization / scaling

- Choose normalization scheme (YAML):
  - `none`: use raw values.
  - `zscore`: mean/std computed over training set.
  - `minmax`: min/max scaling per feature.
- Plan to implement a small statistics pass in preprocessing:
  - Compute feature-wise stats on training indices only.
  - Store stats in `metadata` and re-use for validation/test.

## 2. Tensor layout for CNN input

### 2.1. Spatial arrangement

- Define a consistent mapping from logical features to the CNN grid.

- **Option A (levels × feature-channels)**:
  - Height: number of levels (e.g. 10).
  - Width: 2 (bid/ask) or small factor for extra structure.
  - Channels: per-level features (price_delta_mid, quantity, cumulative_qty, ...).

- **Option B (levels × features, scalar channels)**:
  - Height: number of levels.
  - Width: number of per-level features.
  - Channels: 1 (single channel).

- **Planned choice** (for clarity and simplicity):
  - Height: `num_levels`.
  - Width: `num_sides * num_per_level_features` (e.g. bid/ask × features).
  - Channels: 1.
  - Example mapping for each level row:
    - columns 0..(k-1): bid features.
    - columns k..(2k-1): ask features.

### 2.2. Integration with existing `height`, `width`, `channels`

- Training pipeline currently calculates a minimal `(height, width, channels)` from CNN kernel and pooling config.
- Plan:
  - Introduce a new `data.snapshot_features.spatial_layout` config section that defines the intended `height` and `width`.
  - Enforce consistency: computed feature grid size must be **at least** the model-required minimal size.
  - If model's minimal size is larger, pad feature grid with zeros.
  - If feature grid is larger than model minimal size, it is allowed (CNN operates on full grid).

## 3. Config changes (YAML + schema)

### 3.1. New YAML keys

- Under `data` section:

  ```yaml
  data:
    snapshot_features:
      enabled: true
      levels: 10
      sides: ["bid", "ask"]
      per_level_features: ["price_delta_mid", "quantity", "cumulative_qty"]
      aggregate_features: ["spread", "mid_price", "imbalance"]
      normalization: "zscore"  # ["none", "zscore", "minmax"]
      spatial_layout:
        layout_type: "levels_x_features"  # future-proof
        height: 10
        width: 6          # e.g. 2 sides × 3 features
  ```

- Enforce via `validation_schema.yaml`:
  - Types: `levels` int, arrays of strings, `normalization` string with allowed values.
  - `enabled` must be boolean.
  - `spatial_layout.height` and `width` positive ints.

### 3.2. Backward compatibility / missing keys

- Since your global rule is "no defaults in code":
  - All new keys must be **mandatory** once the richer feature representation is turned on.
  - If `data.snapshot_features.enabled` is true but any required sub-key is missing:
    - Raise `ValueError` during config validation.

## 4. Preprocessing and feature builder

### 4.1. New feature construction function

- Create a dedicated module/function, e.g. `preprocessing/snapshot_features.py`:
  - `build_snapshot_feature_grid(config, order_books, indices) -> np.ndarray`.
  - Responsibilities:
    - Read `data.snapshot_features` config.
    - For each sample index:
      - Extract up to `levels` levels from bid/ask sides.
      - Compute per-level features according to `per_level_features`.
      - Compute aggregate features as extra columns or channels if needed.
    - Apply normalization using precomputed stats from `metadata`.
    - Return `X` with shape `(num_samples, height, width, channels)`.

- This function should **not** know about labels or model; only feature construction.

### 4.2. Normalization statistics pipeline

- Extend preprocessing pipeline:
  - On first run (training phase):
    - On training indices only, build raw feature tensor.
    - Compute feature-wise means/stds or min/max (depending on config).
    - Save stats into `data_object["metadata"]["snapshot_feature_stats"]`.
  - On subsequent phases (validation/test/evaluation):
    - Load stats from metadata.
    - Apply the same normalization without recomputing.

- Logging:
  - Log number of samples used for stats, and a summary of feature ranges.

## 5. Integration with training pipeline

### 5.1. Replace 2×2 patch builder

- In `training/pipeline.py`, where we currently build `x_train`/`x_val` from minimal 2×2 patch:
  - Replace the inline feature mapping with calls to the new snapshot feature builder.
  - Key steps:
    - Collect `train_indices` and `val_indices` (already done).
    - Call:
      - `x_train = build_snapshot_feature_grid(config, order_books, train_indices)`.
      - `x_val = build_snapshot_feature_grid(config, order_books, val_indices)` if needed.

- Ensure consistency checks:
  - Verify returned `height`, `width`, `channels` match or exceed the model requirements computed from `kernel_sizes` and `pool_sizes`.
  - If mismatch, raise a clear error, suggesting to adjust either the CNN config or snapshot layout config.

### 5.2. Synthetic input handling

- Preserve `missing_snapshot_strategy` semantics:
  - If `snapshot_features.enabled` is true but order book data is missing or incomplete:
    - `fail` → raise.
    - `skip` → skip training stage.
    - `synthetic` → use synthetic tensor with same `(height, width, channels)` as configured layout.

## 6. Integration with evaluation pipeline

### 6.1. Consistent feature construction

- In `evaluation/evaluator.py`:
  - Replace the existing snapshot to 2×2 mapping with calls to `build_snapshot_feature_grid` for the evaluation indices.
  - Ensure the same config and normalization stats are used (from `metadata`).

- Checks:
  - Confirm that the constructed feature grid matches the shape used during training.

## 7. Diagnostics and sanity checks

### 7.1. Feature distribution diagnostics

- Extend `diagnostics/data_diagnostics.py`:
  - Add summaries for new features:
    - Histograms / quantiles for price deltas, quantities, imbalance, spread.
    - Per-level depth profiles.
  - Optionally write CSVs with per-feature summary statistics.

### 7.2. Visual inspection of feature grids

- Add an optional diagnostic artifact:
  - Take a small sample of feature grids (e.g. 10 snapshots).
  - Save as images or CSV heatmaps for manual inspection.

## 8. Testing and rollout plan

### 8.1. Unit / integration tests

- Create tests for:
  - Config validation of `data.snapshot_features` section.
  - `build_snapshot_feature_grid` behavior on:
    - Complete order book data.
    - Missing levels (less than `levels` available).
    - Different normalization schemes.
  - Consistency of normalization stats between training and evaluation.

### 8.2. Dry runs

- Run pipeline in a small debug mode:
  - `debug_max_samples` small (e.g. 1000).
  - Confirm that:
    - Training runs end-to-end with new features.
    - Evaluation runs and logs metrics.

### 8.3. Hyperparameter tuning and refinement

- Once the richer features are stable:
  - Experiment with:
    - Different `levels` (e.g. 5 vs 10).
    - Different per-level feature combinations.
    - Normalization schemes.
    - Adjusting CNN kernel sizes to exploit the spatial structure.

## 9. Implementation order

1. **Config & schema**: Add `data.snapshot_features` section and validation.
2. **Feature builder module**: Implement `build_snapshot_feature_grid` with normalization support.
3. **Preprocessing integration**: Add stats computation and storage in metadata.
4. **Training pipeline integration**: Replace 2×2 patch with feature builder.
5. **Evaluation integration**: Use feature builder for evaluation features.
6. **Diagnostics**: Extend diagnostics for new features and optional visualization.
7. **Tests & dry runs**: Validate behavior and run small-scale experiments.
