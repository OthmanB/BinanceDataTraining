# Temporal Window CNN+LSTM Realignment Plan

*Created: 2025-11-18 07:35 (local time)*

This note describes the steps required to realign the implementation with the original vision: each training sample is a **temporal window of T snapshots**, where \(T\) is derived from `targets.visible_window_seconds` and `data.time_range.cadence_seconds`, and the CNN+LSTM architecture uses a CNN to encode each snapshot and an LSTM to model the evolution of these spatial patterns over time.

The plan is split into documentation alignment, configuration and validation, preprocessing and indexing, feature tensor construction, model architecture changes, training and evaluation integration, diagnostics, and testing.

## 1. Documentation alignment (design already decided)

1. Update the LaTeX note `Notes/tex/cnn_lstm_feature_representation.tex`:
   - Make explicit that each sample is a sequence of \(T\) snapshots over a visible window.
   - Define \(T = \texttt{targets.visible\_window\_seconds} / \texttt{data.time\_range.cadence\_seconds}\).
   - Describe the per-snapshot CNN block and the time-axis LSTM over frame embeddings.

2. Update `documentation/2025-11-03/vision.md`:
   - In §5.1, clarify that the canonical CNN+LSTM works on temporal windows, with T derived as above.
   - State that CNN handles spatial microstructure per snapshot and LSTM models how these patterns evolve over the visible window.

3. Check `documentation/2025-11-03/technical-specifications.md`:
   - Ensure the description of tensor shapes and the CNN+LSTM architecture is consistent with the temporal-window design.
   - Keep the document high level; avoid binding it to implementation details that may still evolve.

## 2. Configuration and validation

1. Confirm that the current `config/training_config.yaml` includes:
   - `targets.prediction_horizon_seconds` (already used in target construction).
   - `targets.visible_window_seconds` (already present and used in diagnostics).
   - `data.time_range.cadence_seconds`.

2. Enforce consistency of visible window and cadence:
   - At every place where temporal semantics are needed (preprocessing, training, evaluation, sequence building), compute:
     - `cadence_seconds = data.time_range.cadence_seconds`.
     - `visible_window_seconds = targets.visible_window_seconds`.
     - Require `visible_window_seconds > 0`.
     - Require `visible_window_seconds % cadence_seconds == 0`.
     - Define `T = visible_window_seconds // cadence_seconds` and require `T >= 1`.
   - If these conditions are violated, raise a clear `ValueError` at startup.

3. Keep the design simple for v1:
   - Use a **backwards-aligned** window `[t - T + 1, ..., t]` for each anchor time `t`.
   - Do not add extra YAML options for window alignment or padding policies yet; document the chosen behaviour in the technical docs and LaTeX instead.

## 3. Preprocessing: anchor indices and label construction

1. Extend `_build_targets_from_order_book` in `preprocessing/transformer.py`:
   - Read `targets.visible_window_seconds` and `data.time_range.cadence_seconds`.
   - Compute `T` as described above and validate.
   - After reconstructing snapshot-level mid prices and ensuring `horizon_steps > 0`, compute the set of **valid anchor snapshot indices**:
     - Let `num_snapshots = mid_prices.shape[0]`.
     - Let `anchor_start = T - 1` (first index that has a full past window).
     - Require that there is enough future horizon for each anchor, i.e. `anchor_idx + horizon_steps <= num_snapshots - 1`.
     - This yields an effective number of samples:
       ```
       effective_samples = num_snapshots - horizon_steps - (T - 1)
       ```
   - If `effective_samples <= 0`, log and set `metadata.num_samples = 0` and return.

2. Construct anchor indices and labels:
   - Build an integer list `anchor_indices` of length `effective_samples` where
     ```
     anchor_indices[i] = anchor_start + i
     ```
     so that sample index `i` corresponds to snapshot index `anchor_indices[i]`.
   - For each sample `i`:
     - Let `anchor_idx = anchor_indices[i]`.
     - Use `p0 = mid_prices[anchor_idx]` as the reference price.
     - Define the future window as `mid_prices[anchor_idx + 1 : anchor_idx + horizon_steps + 1]`.
     - Compute maximum upward/downward moves and the two-head intensity labels exactly as in the current implementation, but using this anchor index.
   - Set `metadata["num_samples"] = effective_samples`.
   - Store `metadata["anchor_indices"] = anchor_indices` in the `DataObject`.
   - Store label arrays in `data_object["targets"]` as currently done.

3. Maintain backward compatibility in index semantics:
   - The **sample index space** for labels and splits is now `0, 1, ..., effective_samples - 1`.
   - Each sample index `i` maps to a specific snapshot index `anchor_indices[i]` and corresponding mid-price and timestamp.
   - Downstream code must consistently use sample indices for labels and `anchor_indices` to reach back to snapshot-level inputs and timestamps.

## 4. Snapshot sequence tensor construction (top-of-book baseline)

1. Introduce a small, focused helper in a new module `preprocessing/snapshot_sequence_builder.py` (no additional configuration keys):
   - Function signature (conceptual):
     ```python
     build_top_of_book_sequence_tensor(
         config,
         snapshot_features,
         anchor_indices,
         sample_indices,
         height,
         width,
         channels,
     ) -> np.ndarray
     ```
   - Inputs:
     - `snapshot_features`: list of per-snapshot top-of-book features for the target asset (`[best_bid_price, best_bid_qty, best_ask_price, best_ask_qty]`).
     - `anchor_indices`: list of snapshot indices for each sample (from metadata).
     - `sample_indices`: indices in sample space (train, val, test) for which to build tensors.
     - `height`, `width`, `channels`: spatial dimensions required by the CNN (already computed in the training and evaluation pipeline).
   - Behaviour:
     - Compute `T` from configuration and validate.
     - Allocate a tensor `X_seq` of shape `(len(sample_indices), T, height, width, channels)` filled with zeros.
     - For each sample `s` with sample index `i` in `sample_indices`:
       - Retrieve `anchor_snapshot_idx = anchor_indices[i]`.
       - For each temporal position `τ = 0, ..., T-1`:
         - Compute `snapshot_idx = anchor_snapshot_idx - (T - 1 - τ)`.
         - If `snapshot_idx` is within bounds, read `snapshot_features[snapshot_idx]` and parse the four baseline features.
         - Map them into the **top-left 2×2 patch** of the spatial grid at time `τ`:
           - `(0,0)` → best bid price.
           - `(0,1)` → best bid quantity (if `width > 1`).
           - `(1,0)` → best ask price (if `height > 1`).
           - `(1,1)` → best ask quantity (if `height > 1` and `width > 1`).
         - Leave all other cells as zero for this baseline implementation.
     - Return `X_seq` as the input tensor for the model.

2. This helper will be used in both training and evaluation to ensure the sequence construction logic is centralized and consistent.

## 5. Model architecture changes (time-axis LSTM)

1. Update `models/cnn_lstm_multiclass.py` so that `build_cnn_lstm_model` expects an input shape `(T, H, W, C)`:
   - Validate `len(input_shape) == 4` and raise a clear error otherwise.
   - Keep the existing configuration-driven CNN parameters (`num_layers`, `filters`, `kernel_sizes`, `pool_sizes`, `activation`, `dropout_rates`).

2. Apply the CNN per time step using a `TimeDistributed` wrapper:
   - Starting from `inputs` of shape `(T, H, W, C)`, apply for each convolutional layer:
     - `TimeDistributed(Conv2D(..., padding="same"))`.
     - `TimeDistributed(MaxPooling2D(...))`.
     - Optionally `TimeDistributed(Dropout(...))` when dropout rate is positive.
   - After all conv+pool layers, apply `TimeDistributed(Flatten())` to obtain a tensor of shape `(T, D)` where `D = H' × W' × filters_last`.

3. Replace the current spatial LSTM with a time-axis LSTM:
   - Feed the sequence `(T, D)` into `LSTM(units, dropout, recurrent_dropout, return_sequences=False)`.
   - Interpret the resulting vector as a learned summary of how the spatial patterns of the order book evolve over the visible window.

4. Keep the dense and output heads as they are structurally:
   - Dense block as configured in `model.dense.layers` and `model.dense.dropout_rates` (e.g. one dense layer with 64 units and dropout 0.2 in the baseline).
   - Two softmax heads `up_intensity` and `down_intensity` with `model.output.num_classes` units each and activation `model.output.activation`.
   - Compilation logic (optimizer, learning rate, loss, metrics) remains as already implemented for the two-head model.

## 6. Training pipeline integration

1. In `training/pipeline.py`, after reading `metadata.num_samples` and performing the chronological split:
   - Read `metadata["anchor_indices"]` and validate:
     - It must exist when using snapshot features.
     - Its length must equal `n_samples`.
   - Compute `T` from configuration (visible window and cadence) and validate.

2. Adjust the input shape passed to `build_cnn_lstm_model`:
   - Compute `height`, `width`, and `channels` exactly as today from the CNN kernel and pooling configuration.
   - Define `input_shape = (T, height, width, channels)`.

3. Replace the current 2×2 patch per-sample construction with a call to the sequence builder:
   - Obtain `snapshot_features` for the target asset as today.
   - If `snapshot_features` is non-empty:
     - Use `build_top_of_book_sequence_tensor` with `sample_indices = train_indices` to build `x_train` of shape `(effective_train_n, T, H, W, C)`.
     - If `val_indices` is non-empty, call the same helper with `sample_indices = val_indices` to build `x_val`.
   - If `snapshot_features` is empty:
     - Respect `training.missing_snapshot_strategy` exactly as now:
       - `fail` → raise.
       - `skip` → log and return.
       - `synthetic` → build synthetic tensors of shape `(effective_train_n, T, H, W, C)` and, if applicable, `(len(val_indices), T, H, W, C)` using random noise.

4. Labels:
   - Keep the current logic for retrieving `labels_up_intensity` and `labels_down_intensity` from `data_object["targets"]`.
   - Ensure that their length is at least `n_samples = metadata.num_samples`.
   - One-hot encode them into `y_up_train`, `y_down_train` and `y_up_val`, `y_down_val` using `train_indices` and `val_indices` exactly as now (sample indices remain unchanged).

5. Fit the model:
   - Call `build_cnn_lstm_model(config, input_shape=input_shape)` to construct the updated architecture.
   - Call `model.fit` with `x_train` of shape `(N_train, T, H, W, C)` and the two-head labels.
   - Keep callbacks, MLFlow logging, and model artifact logging behaviour unchanged.

## 7. Evaluation pipeline integration

1. In `evaluation/evaluator.py`:
   - After computing `test_idx` and `eval_n`, read `metadata["anchor_indices"]` and validate it as in the training pipeline.
   - Compute `height`, `width`, `channels` from CNN config as currently done.
   - Retrieve `snapshot_features` for the target asset.

2. Build evaluation inputs:
   - Define `eval_indices = test_idx[:eval_n]` (sample indices).
   - If `snapshot_features` is non-empty, call `build_top_of_book_sequence_tensor` with `sample_indices = eval_indices` to obtain `x_eval` of shape `(eval_n, T, H, W, C)`.
   - If `snapshot_features` is empty:
     - Respect `evaluation.missing_snapshot_strategy` exactly as now:
       - `fail` → raise.
       - `skip` → log and return.
       - `synthetic` → build synthetic tensors of shape `(eval_n, T, H, W, C)` using random noise.

3. Predictions and metrics:
   - Keep the existing logic for:
     - Getting `labels_up_intensity` and `labels_down_intensity` from `data_object["targets"]`.
     - Restricting to `eval_indices` for `y_true_up` and `y_true_down`.
     - Calling `model.predict(x_eval)` and checking that it returns two heads.
     - Computing per-class confusion matrices, macro metrics, and calibration metrics.
   - Only the input shape changes; the metric definitions do not.

## 8. Data diagnostics alignment

1. In `diagnostics/data_diagnostics.py`, align the notion of index with anchor indices:
   - The `train_idx`, `val_idx`, and `test_idx` arguments are in **sample index space**.
   - Read `metadata["anchor_indices"]` if present.
   - When sampling indices for diagnostic records:
     - Use sample indices for labels (`targets["labels"]`).
     - Map sample indices to snapshot indices via `anchor_indices` when accessing:
       - `snapshot_features`.
       - `snapshot_timestamps` and normalized timestamps.
   - This ensures that the diagnostics operate on the same anchor snapshots that the model uses for labels and sequences.

2. Keep all existing diagnostic metrics and artifacts intact:
   - Spread statistics, anomaly exports, plots, and heatmaps continue to work on the underlying snapshots.
   - Only the mapping from sample indices to snapshots changes.

## 9. Testing and verification

1. Unit-level checks:
   - Add simple tests or manual runs to confirm that:
     - `_build_targets_from_order_book` produces `metadata.num_samples`, `targets.*` arrays, and `metadata.anchor_indices` with consistent shapes and index mappings.
     - `build_top_of_book_sequence_tensor` returns tensors of expected shape and correctly maps top-of-book values into the temporal grid.

2. End-to-end dry run:
   - Use a small date range and small `debug_max_samples` to:
     - Run preprocessing.
     - Run training with the updated CNN+LSTM architecture.
     - Run evaluation.
     - Run diagnostics.
   - Inspect logs to verify that:
     - The same `T` is used consistently.
     - No index-out-of-bounds errors occur.
     - MLFlow receives metrics and artifacts as before.

3. Consistency with documentation:
   - Cross-check that:
     - The LaTeX note, `vision.md`, and `technical-specifications.md` all describe the temporal-window CNN+LSTM design.
     - The actual tensor shapes and index semantics used in code match these descriptions.
