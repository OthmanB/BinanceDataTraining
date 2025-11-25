# Phase 2 Plan: Time Alignment & Long-Term Channel

*Created: 2025-11-19 16:30 (local time)*

This note describes the next implementation phase in two parts:

- (A) Wire `data.validation` and time alignment/gap handling into the preprocessing pipeline.
- (B) Design and implement a first simple long-term/global temporal channel that complements the existing short-term temporal window CNN+LSTM.

The goal is to stay close to the current codebase and configuration while moving toward the multi-scale, memory-aware architecture described in `vision.md` and `technical-specifications.md`.

---

## A. Time Alignment & `data.validation` Wiring

### A1. Clarify configuration semantics

1. **Inventory existing config keys**
   - `data.validation.*` (gap checks, max_gap_seconds, fail_on_invalid, etc.).
   - `data.time_range.cadence_seconds` (target cadence for snapshots).
   - Any time-alignment-related keys already described in `vision.md` (e.g., tolerance_ms, resampling method) and reconcile with actual YAML.

2. **Define effective validation contract**
   - What does it mean for a row/snapshot to be "valid" from the perspective of:
     - Timestamps (monotonicity, maximum gap).
     - Order book values (non-negative prices/quantities, sane spreads).
   - Decide which checks must run **before** target/sequence construction, and what behaviour is expected when checks fail:
     - Fail-fast for gross schema or timestamp violations.
     - Sample- or day-level skipping for local problems.

3. **Update `validation_schema.yaml` if needed**
   - Ensure `data.validation` keys are explicitly typed and required/optional status is clear.
   - Add any missing keys that the implementation will rely on (e.g., `max_gap_seconds`).

### A2. Implement timestamp-driven snapshot validation

4. **Centralise timestamp parsing and checks**
   - Add a small utility (e.g., in `preprocessing/validator.py` or a new helper) that:
     - Converts raw timestamp fields to a consistent `datetime64` or numeric epoch representation.
     - Checks monotonicity and maximum gap per asset according to `data.validation`.

5. **Integrate into `transformer.py` before target construction**
   - For the target asset:
     - Apply timestamp checks on the reconstructed snapshot timeline before computing mid prices and labels.
     - If gaps exceed `max_gap_seconds`:
       - Either drop the offending spans (if allowed by config), or
       - Abort with a clear `ValueError` when `fail_on_invalid` is set.

6. **Ensure joint treatment of timestamps and order book values**
   - When discarding snapshots due to timestamp issues, also discard corresponding `snapshot_features`, mid prices, and any derived labels.
   - Keep `metadata.time_range` consistent with the final snapshot set (start, end, cadence).

### A3. Prepare for multi-asset time alignment (without fully implementing it yet)

7. **Design the reference-asset alignment contract**
   - Document (in code comments and `technical-specifications.md`) that:
     - A single reference asset timeline (e.g., BTCUSDT) defines the master grid.
     - Other assets must be interpolated/resampled to this grid before stacking.
   - This phase focuses on the target asset only, but the alignment logic should be written so it can be extended to multiple assets.

8. **Add stubs or hooks for future multi-asset alignment**
   - Where appropriate (e.g., in `transformer.py` or a new `temporal_alignment` module), define functions that will later handle multi-asset resampling, but for now operate on the single target asset.

### A4. Diagnostics and tests

9. **Extend data diagnostics to monitor gaps**
   - In `diagnostics/data_diagnostics.py`, add metrics derived from the timestamp sequence:
     - Distribution of inter-snapshot intervals.
     - Fraction of gaps exceeding a warning threshold.
   - Log these to MLFlow for each run.

10. **End-to-end checks**
    - With a small date range and `debug_max_samples`:
      - Run preprocessing and ensure that:
        - Timestamp-based discards behave as specified by `data.validation`.
        - `metadata.num_samples`, `anchor_indices`, and timestamps remain consistent.
      - Verify logs clearly report any discarded spans or failures.

---

## B. First Simple Long-Term / Global Channel

The aim is to introduce a minimal "global context" channel that is consistent with the vision (local vs global state) but does not yet implement the full dual-scale architecture. This should:

- Be driven by configuration.
- Use coarse temporal aggregation (e.g., daily or multi-minute bins).
- Produce a compact feature vector per sample capturing longer-term behaviour before the local visible window.

### B1. Choose an initial long-term timescale and representation

1. **Decide baseline global horizon and resolution**
   - Example v1 choice (configurable):
     - Horizon: 30 days prior to the visible window anchor.
     - Resolution: 1 day.
   - Represent as daily aggregates of simple statistics over mid prices or returns.

2. **Define configuration keys**
   - Under `data.multi_scale` or a new `long_term` block, add (in YAML + schema):
     - `long_term.window_days` (e.g., 30).
     - `long_term.resolution_days` (e.g., 1).
     - `long_term.features`: list of statistics (e.g., `mean_return`, `volatility`, `volume_proxy`).

### B2. Implement long-term feature computation (offline, CPU-friendly)

3. **Construct coarse-grained series from snapshots**
   - Starting from the target-asset snapshot timeline (already validated):
     - Compute daily (or multi-minute) aggregates of mid-price returns and/or simple order-book-derived proxies.
     - Store as a separate array in the `DataObject`, e.g., `data_object["long_term_features"]` with shape `(N_days, feature_dim)` and associated timestamps.

4. **Align long-term features with anchors**
   - For each sample anchor index:
     - Determine the corresponding calendar day (or coarse bin) and the preceding `window_days` of long-term features.
     - Aggregate them into a fixed-size vector per sample, e.g.:
       - Exponentially weighted averages of daily volatility.
       - Simple statistics over the last `window_days` of daily returns.
   - Store the resulting per-sample global vector in `DataObject["temporal_features"]["global"]` or a new `long_term_channel` field.

### B3. Integrate the long-term vector into the model

5. **Model input changes (minimal concatenation design)**
   - Keep the existing short-term CNN+LSTM branch unchanged.
   - Add a **small MLP branch** for the long-term/global vector:
     - Input: `global_input` of shape `(global_feature_dim,)`.
     - A few dense layers with dropout as configured.
   - Concatenate the CNN+LSTM output with the global branch output before the final dense block.

6. **Configuration for the long-term branch**
   - Under `model.long_term` (or similar), add:
     - `enabled: true/false`.
     - `dense.layers`, `dense.dropout_rates` for the long-term MLP.
   - Ensure validation enforces that all required keys are present when `enabled: true`.

7. **Training & evaluation pipeline adjustments**
   - In `training/pipeline.py` and `evaluation/evaluator.py`:
     - Build `x_global_train`, `x_global_val`, `x_global_eval` from the per-sample long-term vectors.
     - Pass them as a second input to the Keras model when `model.long_term.enabled` is true.

### B4. Testing and diagnostics for the long-term channel

8. **Unit-level checks**
   - Verify that for a small synthetic dataset:
     - Long-term aggregates are computed correctly from snapshots.
     - Per-sample long-term vectors align with anchor timestamps.

9. **End-to-end run with long-term channel enabled**
   - Run the full pipeline with modest data and `model.long_term.enabled: true`:
     - Confirm input shapes for both branches are correctly inferred.
     - Confirm training completes without memory issues.

10. **Extend diagnostics to include long-term context summaries**
    - Log basic statistics of long-term features (means, standard deviations, min/max) to MLFlow.
    - Optionally export a small sample of `(anchor_time, long_term_vector)` pairs for manual inspection.

---

## C. Dependencies and Ordering

1. Implement and stabilise (A) time alignment and `data.validation` wiring for the target asset.
2. Once timestamp handling is robust, implement (B) long-term feature computation on top of the validated snapshot timeline.
3. Only after both are working reliably, consider integrating more advanced elements from the vision:
   - Multi-asset alignment and stacking.
   - More complex multi-scale regime detection.
   - Streaming/tf.data pipelines under explicit memory budgets.
