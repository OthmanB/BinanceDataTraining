# Training Usability and Workstation Capability Assessment

**Date:** 2026-09-05

**Scope:** Practical usability, experiment design, hardware fit, and parameter selection.

**Out of scope:** A code-correctness review and any claim that the model has a profitable market edge.

## Executive Assessment

This project is already a capable experimentation framework. It can ingest and align multi-asset order-book data, build compact restartable snapshot stores, train a configurable CNN-LSTM, run parallel Optuna studies, track experiments in MLflow, and perform classification, calibration, temporal, and signal-backtest evaluation.

It is not practical as configured in `config/e2e_fullscale_production_2022to2024.yaml`.

The main conclusions are:

1. **The workstation is sufficient.** Three RTX 3090 GPUs, a 32-core Threadripper, and 141 GiB RAM are enough for serious experiments with this model family. More compute hardware is not the first need.
2. **The checked-in production workload is dominated by tensor representation and experiment multiplication, not model size.** The effective model has only about 241,000 parameters, but one short-term sample is 8.79 MiB and retains 720 time steps through every CNN layer and into the LSTM.
3. **Three independent GPU jobs are substantially more useful than one three-GPU mirrored job.** Measured aggregate throughput for three concurrent one-GPU jobs was about 42.4 samples/s. A three-GPU mirrored run at a comparable global batch measured only 9.7 samples/s and was slower than one GPU at the same global batch.
4. **The full production shape is usable only at small batch sizes.** Batch 8 reached about 16.0 samples/s end to end and used 20.5 GiB peak VRAM. Batch 8 is the practical ceiling for the tested baseline architecture on a 24 GiB card. The inherited batch 12 is not safe for this shape.
5. **A parameter-only efficiency profile can make the project practical.** A tested `(256,20,4,10)` profile with batch 80 reached 241 samples/s end to end and used 13.1 GiB peak VRAM. This is about 15 times the measured throughput of the current production shape while retaining two assets, confidence masks, six engineered features, and the same CNN-LSTM family.
6. **The current `auto` undersampling setup is an immediate operational blocker.** On an existing production-shaped 20-day snapshot it attempted to reduce 47,534 training samples to 24 and then failed the configured 500-sample minimum. Disable coarse undersampling for the initial practical profile.
7. **The configured 10-second cadence does not match the cached data.** The measured median interval is 14 seconds and only about 0.2% of frame intervals are exactly 10 seconds. Since the code derives step counts from `cadence_seconds` but does not resample source rows, nominal time windows are not the actual time windows.
8. **The learning target is better described as future excursion intensity than direct direction or endpoint return.** The two heads independently classify the maximum upward and downward excursions anywhere in the future horizon. This is useful for barrier-touch probability, adverse/favorable excursion, volatility, and risk decisions. It is not directly aligned with a fixed-horizon entry/exit backtest.
9. **The largest remaining usability gains require small architectural additions rather than more hardware.** A configurable sample stride, separate vector branch for auxiliary features, a purged walk-forward split, parallel input preparation, and epoch-level recovery would each materially improve practical use.

### Overall Verdict

| Capability | Assessment |
|---|---|
| Short smoke tests | Good |
| One-GPU model iteration | Good after reducing the input shape |
| Parallel architecture or parameter experiments | Good; this is the best use of all three GPUs |
| Current three-GPU mirrored final training | Poor efficiency |
| Current three-year production config | Not operationally practical |
| Reduced-profile three-year training | Practical on this machine |
| Experiment tracking and diagnostics | Broad, but operational controls need simplification |
| Statistically credible market validation | Incomplete without purge/embargo and walk-forward changes |
| Direct trading-strategy validation | Exploratory only in the current form |

## Workstation Inventory

The following was measured on the machine during this assessment.

| Resource | Observed configuration | Practical implication |
|---|---|---|
| CPU | AMD Ryzen Threadripper 3970X, 32 cores / 64 threads | More than sufficient, but the current batch materializer uses approximately one CPU core per training process. Most CPU capacity is idle. |
| RAM | 141 GiB total, about 79 GiB available during inspection | Sufficient for three independent training workers and shared page cache. Three 28 GiB HPO RSS limits leave little formal headroom, but measured batch preparation was much smaller. |
| GPU 0 | RTX 3090, 24 GiB, 300 W cap | Suitable for one full-shape batch-8 worker or a much larger reduced-shape batch. |
| GPU 1 | RTX 3090, 24 GiB, 275 W cap | Concurrent benchmark throughput was effectively the same as the other cards. |
| GPU 2 | RTX 3090, 24 GiB, 350 W cap | Concurrent benchmark throughput was effectively the same as the other cards. |
| GPU topology | No active NVLink path; PCIe topology is `NODE`/`PHB` and the driver reports PCIe generation 3 maximum | Synchronous gradient exchange is relatively expensive for such a small model. Independent workers avoid this cost. |
| Primary storage | 1 TB Samsung SSD 860 SATA; 504 GiB free | Capacity is adequate. Cold snapshot creation and multi-worker reads would benefit from NVMe, but storage is not the primary hot-loop bottleneck once data are cached. |
| Secondary storage | 500 GB WDC SATA SSD mounted at `/mnt/model-store`; 246 GiB free | Useful for artifact isolation, but not an NVMe throughput upgrade. |
| Python | 3.12.3 | Within the repository's supported range. |
| TensorFlow | 2.16.2, CUDA-enabled, all three GPUs visible | Runtime installation is usable. `python -m pip check` reported no broken requirements. |

The three GPUs must be treated as dedicated training resources. Before the resident `llama-server` was stopped, GPUs 1 and 2 had almost no free VRAM and a production-shaped model could not be initialized on GPU 2. The installed aggregate of 72 GiB VRAM is not pooled memory, and unrelated resident processes can make an individual 24 GiB card unavailable.

## What The Algorithm Actually Does

### Inputs

The production lineage uses:

- BTCUSDT as the target asset and ETHUSDT as a correlated asset.
- Up to 100 order-book levels per side from the source.
- A hybrid representation retaining 20 raw levels and aggregating the remainder into 20 bins.
- Four values per level: bid price, bid quantity, ask price, and ask quantity.
- One data channel per asset plus one confidence-mask channel per asset.
- Six engineered features and ten temporal/session values.
- An optional compact long-term branch.

The hybrid representation is implemented in `preprocessing/depth_aggregator.py:101-121` and `preprocessing/depth_aggregator.py:181-313`.

### Labels

For each anchor, the code examines every future midpoint in the horizon, calculates the maximum upward excursion and maximum downward excursion, and discretizes them independently (`training/snapshot_dataset.py:414-427`, `training/snapshot_dataset.py:3730-3750`).

This means:

- Both heads can legitimately predict a high intensity when the future path moves strongly in both directions.
- The label is not the return at the end of the horizon.
- The label does not encode which barrier occurs first.
- The label does not encode whether a predicted excursion was executable after spread, fees, latency, or slippage.

The model is therefore naturally useful as an **excursion or path-risk forecaster**. It can potentially inform take-profit barriers, stop distance, order placement, trade filtering, or volatility/range forecasts. Converting the two heads directly into one long/short decision throws away part of this meaning.

The current signal backtest resolves conflicting up/down predictions by choosing the stronger probability and exits at a fixed future index (`evaluation/backtesting.py:253-269`, `evaluation/backtesting.py:301-399`). That execution rule is not aligned with maximum-excursion labels. Backtest output should be treated as a signal sanity check, not profitability evidence.

### Model

Each time slice passes through TimeDistributed Conv2D and pooling layers. The spatial result is flattened, the complete temporal sequence goes through an LSTM, and the representation feeds two softmax heads (`models/cnn_lstm_multiclass.py:102-194`, `models/cnn_lstm_multiclass.py:399-469`).

For the effective full-scale production lineage:

```text
Input: (batch, 720, 40, 4, 20)
CNN:   16 -> 32 -> 64 filters
LSTM:  96 units
Long-term branch: 8 values -> Dense 32
Shared dense: 64 units
Outputs: two heads x 5 classes
Parameters: 241,434
```

The model is small. Input activations, host copies, and the long sequence are expensive; weights are not.

### Storage And Streaming

The `frame_store_v1` snapshot format stores base frames, masks, labels, anchors, and compact auxiliary vectors once, then reconstructs overlapping windows from memory-mapped arrays (`training/snapshot_dataset.py:815-905`, `training/snapshot_dataset.py:921-924`). This is a sound storage design. An existing 20-day snapshot held 67,907 samples in about 132 MiB instead of persisting hundreds of gigabytes of expanded sample tensors.

The cost is paid again during every batch. The batch path reconstructs windows, broadcasts masks and auxiliary values, concatenates arrays, strips masks for normalization, normalizes flattened positions, and concatenates masks back (`training/snapshot_dataset.py:815-905`, `training/snapshot_dataset.py:1648-1694`, `training/snapshot_dataset.py:2049-2094`).

## Current Production Shape

The effective short-term dimensions are:

| Dimension | Value | Derivation |
|---|---:|---|
| Time `T` | 720 | `7200 / 10` configured seconds |
| Height `H` | 40 | 20 raw levels + 20 aggregate bins |
| Width `W` | 4 | bid price/quantity + ask price/quantity |
| Channels `C` | 20 | 2 assets + 2 masks + 6 engineered + 10 temporal |
| Values/sample | 2,304,000 | `720 * 40 * 4 * 20` |
| Bytes/sample | 9,216,000 decimal / 8.79 MiB | Float32 host tensor |

Sixteen of the 20 channels are anchor-level auxiliary values; another two are confidence masks. The six engineered values and ten temporal values are calculated once at the anchor and broadcast across all `T * H * W` positions (`training/snapshot_dataset.py:3753-3893`). Thus 80% of the channel width contains repeated anchor values rather than time-varying depth data.

The stored snapshot is compact, but each batch of eight logically expands to about 70.3 MiB before normalization and temporary arrays. Device-side backpropagation is much larger because intermediate activations and gradients must be retained.

## Measured Benchmarks

### Method

Benchmarks used:

- TensorFlow 2.16.2 with `mixed_float16` and float32 output heads.
- GPU memory growth enabled.
- The effective production CNN-LSTM and its actual tensor shape.
- Existing `frame_store_v1` arrays under `snapshots/fullscale_trial_2022to2024_b3fcda89` for host and end-to-end tests.
- Warmed step measurements after TensorFlow graph tracing.

These are short engineering microbenchmarks, not full-epoch model-quality experiments. They are suitable for bottleneck and capacity decisions. Long jobs will also include snapshot preparation, normalization-stat passes, validation, diagnostics, MLflow, and evaluation.

### Full Production Shape: One GPU

Model-only training results for `(720,40,4,20)`:

| Batch | Samples/s | Peak TensorFlow device allocation |
|---:|---:|---:|
| 2 | 6.27 | 5.16 GiB |
| 4 | 10.89 | 10.27 GiB |
| 6 | 14.65 | 15.35 GiB |
| 8 | 18.85 | 20.54 GiB |

The actual snapshot generator plus model at batch 8 measured:

| Batch | End-to-end samples/s | Peak device allocation |
|---:|---:|---:|
| 8 | 15.98 | 20.46 GiB |

This end-to-end test omitted the tiny eight-value long-term branch so it could consume the existing short-term generator directly. The tested short-term model had 239,098 parameters rather than 241,434; this difference is too small to change the throughput conclusion. The reduced-shape end-to-end tests included the eight-value long-term input.

The standalone host reconstruction and normalization path measured approximately 19 to 22 samples/s at batches 2 through 12. Process CPU time matched wall time, indicating one effective CPU core. Batch size did not materially improve host throughput.

Practical conclusions:

- Batch 8 is the highest reasonable full-shape batch for the baseline model.
- Batch 9 would leave too little headroom, and batch 12 is outside the measured scaling envelope.
- The current HPO range `[2,4]` is safe but slow. In this syntax it means every integer from 2 through 4, not two categorical values (`models/hyperparameter_tuning.py:1999-2008`).
- A larger architecture can require reducing batch 8 to 6 or 4.

### Mirrored Training Versus Independent Jobs

The repository defines `training.batch_size` as per-replica under MirroredStrategy (`training/distributed.py:135-137`, `training/pipeline.py:606-625`). It wraps a Python generator with `Dataset.from_generator()` and prefetch but does not introduce parallel mapping (`training/distributed.py:240-302`).

Measured three-GPU mirrored result:

| Mode | Per-GPU batch | Global batch | Global samples/s | Peak per GPU |
|---|---:|---:|---:|---:|
| 3-GPU MirroredStrategy | 2 | 6 | 9.74 | about 5.15 GiB |

One GPU at batch 6 measured 14.65 model-only samples/s. The mirrored job was therefore slower than putting the same global batch on one card. Graph initialization for the mirrored test took about 8.7 minutes, compared with roughly 1 to 4 minutes in the one-GPU tests depending on shape and data adapter.

Measured concurrent independent result:

| GPU | Batch | End-to-end samples/s |
|---:|---:|---:|
| 0 | 8 | 14.05 |
| 1 | 8 | 14.28 |
| 2 | 8 | 14.12 |
| Aggregate | 24 across jobs | 42.44 |

Three independent jobs retained about 89% of the isolated per-job throughput and delivered 2.66 times the isolated single-job rate. This is the correct scheduling model for this workstation.

### Tested Efficiency Shapes

The first reduced test used:

```text
T=360, H=20, W=4, C=10
```

This represents a 3,600-second visible window only if source cadence is truly 10 seconds. It measured 178.5 end-to-end samples/s at batch 64 and 20.6 GiB peak VRAM.

The source cache is actually close to a 14-second cadence. A cadence-aligned test therefore used:

```text
cadence_seconds=14
visible_window_seconds=3584  # 256 frames, about 59m 44s
prediction_horizon_seconds=1792  # 128 frames, about 29m 52s
raw_levels=10
aggregated_bins=10
temporal channel broadcasting disabled
T=256, H=20, W=4, C=10
```

This still includes:

- BTCUSDT and ETHUSDT base channels.
- Two confidence-mask channels.
- Six engineered features.
- The dual-input long-term branch.

Measured results:

| Shape | Batch | End-to-end samples/s | Peak VRAM |
|---|---:|---:|---:|
| `(256,20,4,10)` | 80 | 241.17 | 13.09 GiB |
| `(256,20,4,10)` | 128 | 242.86 | 20.85 GiB |

Host preparation for this shape measured about 282 to 289 samples/s. Batch 128 provided no meaningful throughput gain over batch 80 because the producer was already close to saturation. Batch 80 is therefore the recommended operating point: nearly identical throughput with about 7.8 GiB more VRAM headroom.

The reduced benchmark reconstructed real frame-store data with 256-frame windows, 20 levels, two assets, masks, and the six existing engineered features. It used representative normalization arithmetic rather than rebuilding a new snapshot with newly fitted statistics. A real pilot snapshot must confirm the final number, but the performance difference is large enough that the direction is not ambiguous.

## Cadence Is A Prerequisite, Not A Cosmetic Parameter

`cadence_seconds` is used to derive window and horizon step counts (`training/snapshot_dataset.py:293-320`). Samples still advance one incoming snapshot at a time (`training/snapshot_dataset.py:379-388`). The SQL and ingestion path does not resample rows to that cadence.

Measured on the existing full-scale cache:

| Statistic | Value |
|---|---:|
| Median frame interval | 14 seconds |
| 90th percentile frame interval | 14 seconds |
| Exactly 10-second intervals | about 0.2% |
| Maximum observed interval in this cache | 28 seconds |

Consequences of retaining `cadence_seconds: 10` on these data:

- The nominal 720-frame 2-hour input spans about 2.8 hours at the median source interval.
- The nominal 180-frame 30-minute horizon spans about 42 minutes.
- Momentum step counts are also interpreted against the wrong interval.
- Changing cadence to improve performance without changing source rows silently changes the real duration in the opposite way from what a user may expect.

Immediate parameter-only action:

- If 14 seconds is the intended collector cadence across the target data, set `cadence_seconds: 14` and use values divisible by 14, such as 3,584 and 1,792 seconds.
- If the intended model cadence is 10 seconds, resample or bucket the source explicitly before training. Do not merely leave the YAML at 10.
- Re-measure cadence by asset and date regime before committing to a multi-year snapshot.

Date bounds also require care: `end_date` is inclusive through 23:59:59 (`data/greptime_client.py:115-176`). For example, January-only data end on `2024-01-31`; the checked-in month profile's `2024-02-01` includes February 1.

Required improvement:

- Add explicit source resampling/bucketing with a declared aggregation policy and store observed cadence statistics in the snapshot manifest.

## Data Volume And Runtime Projection

Observed cached data density varies materially:

- One two-day January cache contains 11,733 total samples and 8,213 training samples.
- One 20-day May cache contains 67,907 total samples and 47,534 training samples.
- This corresponds to roughly 2,377 to 4,106 training samples per calendar day after the 70% split.

For 1,096 calendar days from 2022 through 2024, a rough observed-density range is 2.6 to 4.5 million training samples before balancing. The theoretical maximum at a true 10-second cadence is higher. Actual planning must be based on a preflight manifest, not date range alone.

### Current Shape Projection

At the measured 15.98 samples/s:

| Work | Optimistic observed-density estimate |
|---|---:|
| One full-range epoch | about 45 to 78 hours |
| Ten final-training epochs | about 19 to 33 days |
| Six HPO trials on three independent GPUs | about 38 to 65 days |
| HPO plus one final run | about 57 to 98 days |

These are optimistic for the checked-in profile because its HPO batch range is only 2 to 4, which measured slower than batch 8. The checked-in HPO resource list also names only GPUs 0 and 1, making six trials require three waves rather than two (`config/e2e_fullscale_production_2022to2024.yaml:35-51`).

The current configuration also trains ten epochs in each of about 110 sequential 10-day windows. Early stopping patience is inherited as 10, so it cannot usefully shorten a fit capped at 10 epochs. The callback state is recreated for each window (`training/callbacks.py:13-95`, `training/pipeline.py:1434-1523`).

### Efficiency Shape Projection

At the measured 241 samples/s:

| Work | Observed-density estimate |
|---|---:|
| One full-range epoch | about 3.0 to 5.2 hours |
| Eight final-training epochs | about 1.0 to 1.7 days of hot-loop time |
| Practical allowance including validation, cache effects, diagnostics, and evaluation | about 2 to 4 days |

This is a realistic workstation-scale workload. It is also small enough that non-sequential final training becomes possible, which improves evaluation semantics.

## Immediate Configuration Blockers

### 1. Coarse `auto` Undersampling

The current full-scale production overlay enables undersampling with:

```yaml
labeling_criteria: "max_intensity"
target_distribution: "auto"
min_samples_after_balance: 500
```

`auto` assigns an equal target share to every present class and is limited by the rarest class (`training/sample_balancing.py:145-213`). On the inspected 20-day training split:

| Max-intensity class | Before | Selected target |
|---:|---:|---:|
| 0 | 6 | 6 |
| 1 | 2,719 | 4 |
| 2 | 7,399 | 4 |
| 3 | 25,653 | 5 |
| 4 | 11,757 | 5 |
| Total | 47,534 | 24 |

The run then fails the 500-sample minimum. This is expected from the chosen criterion: class 0 requires both the up and down excursions to remain below the first threshold, which is rare across a 30 to 42 minute horizon.

The individual heads are much less imbalanced:

| Head | Class counts 0 through 4 |
|---|---|
| Up | 10,046, 9,123, 6,832, 15,228, 6,305 |
| Down | 9,172, 8,983, 7,466, 16,174, 5,739 |

Recommendation:

- Set `preprocessing.class_balancing.enabled: false`.
- Start with `training.class_weights.compute_from_train: true` and evaluate per-class precision, recall, calibration, and signal frequency.
- Also run an unweighted baseline. The head imbalance is moderate enough that weighting may not improve calibration.
- Do not combine inverse-frequency class weights, coarse undersampling, and 90-day recency weights until each has been independently justified.
- Note that duty-cycle sample weights are always applied even when recency weighting is disabled (`training/snapshot_dataset.py:1245-1315`).

### 2. HPO Objective

The month production profile uses `metric: "loss"` and the three-year overlay inherits it (`config/e2e_fullscale_production_month1.yaml:81-86`). This optimizes training loss, not generalization. The implementation extracts the last history value rather than the best epoch (`training/pipeline.py:552-564`).

Recommendation:

- Use `direction: "minimize"` and `metric: "val_loss"`.
- Keep `restore_best_weights: true` for final training.
- Treat HPO rankings as approximate until the objective reports the best validation epoch directly.

### 3. HPO And Final Batch Semantics

Each parallel HPO worker is a one-GPU job, while final MirroredStrategy interprets the selected batch as per-replica. A winning HPO batch of 8 becomes global batch 24 on three GPUs. It no longer represents the optimization regime that produced the trial score.

Recommendation:

- Separate HPO and final training into different runs.
- Keep final training on one GPU for this model.
- Use the other two GPUs for concurrent ablations or independent seeds.

### 4. Sequential Evaluation

Sequential training carries one model from window to window, but each window independently creates a 70/15/15 split (`training/pipeline.py:1434-1523`). After all windows, the final model is evaluated on every historical window (`main.py:224-250`).

This has several practical consequences:

- Only the first 70% of each window is used for gradient updates.
- Validation and test portions interrupt the chronology of training data at every window.
- Earlier-window evaluation occurs after the model has trained on later windows.
- Long-term features are recomputed from each window's local snapshot history, so long windows such as 30 or 90 days may be mostly zero in short sequential windows.

Recommendation:

- Use sequential mode for online-adaptation experiments and recovery, not as the primary final holdout evaluation.
- Use non-sequential training for the reduced final profile.
- Add proper expanding-window or rolling walk-forward evaluation before treating temporal results as evidence.

### 5. No Purge Or Embargo

The chronological split consists of adjacent index ranges with no gap (`preprocessing/train_test_split.py:16-45`). Adjacent full-shape inputs share 719 of 720 frames, and adjacent labels share most of their 180-step horizon.

At minimum, validation and test boundaries need a label-horizon purge. A stricter independence gap is approximately input window plus prediction horizon.

For the cadence-aligned efficiency profile:

- Label-only purge: 128 frames, about 30 minutes.
- Full input-plus-label separation: 384 frames, about 90 minutes.

This requires implementation; there is no current YAML parameter for it.

## Recommended Parameter Strategy

Parameter search should be staged. Data meaning and representation should be stabilized before spending trials on filter counts.

### Stage 1: Data And Label Preflight

Use a 2 to 7 day range and no HPO.

| Parameter | Recommended starting value |
|---|---|
| `run_mode.mode` | `production` because `trial` means HPO-only |
| `training.epochs` | 1 |
| `training.batch_size` | 16 to 32 for the reduced shape; 2 for the current full shape |
| `hyperparameter_optimization.enabled` | `false` |
| `data.validation.fail_on_invalid` | `false` for discovery, then fix the causes |
| `diagnostics.enabled` | `true` |
| `diagnostics.sampling.num_samples` | 50 |
| `diagnostics.visualization.enabled` | `false` for repeated preflights |
| `preprocessing.class_balancing.enabled` | `false` |
| `training.class_weights.compute_from_train` | `false` |
| `training.sample_weighting.enabled` | `false` |

Success criteria:

- Confirm source cadence per asset.
- Confirm aligned-frame density, gaps, confidence masks, and duty cycle.
- Confirm per-head class counts and joint up/down counts.
- Confirm a complete train/validation/evaluation pass.
- Record actual batch throughput and peak VRAM.

### Stage 2: Practical Efficiency Baseline

This is the recommended first serious model profile if the observed 14-second cadence is representative.

| Area | Parameter | Recommended value |
|---|---|---|
| Assets | target / correlated | BTCUSDT / `[ETHUSDT]` |
| Cadence | `data.time_range.cadence_seconds` | 14, after confirming source cadence |
| Ingestion | chunk hours / delay / concurrency | 12 / 0 / 1 for an offline historical database |
| Input window | `targets.visible_window_seconds` | 3,584, giving 256 frames |
| Horizon | `targets.prediction_horizon_seconds` | 1,792, giving 128 frames |
| Classes | count / boundaries | 5 / `[0.15, 0.35, 0.55, 1.5]` for the first comparable baseline |
| Raw depth | `data.order_book.depth_levels` | 100 |
| Hybrid depth | raw / bins | 10 / 10 |
| Alignment | method / policy | `interpolate` / `forward_fill` |
| Alignment stale limit | `alignment.max_gap_seconds` | 84 as an initial six-step cap |
| Validation | max gap / fail | 56 seconds / `true` after preflight cleanup |
| Masks | `include_mask_channel` | `true` |
| Normalization | method / fitting | `min_max` / `fit_on_train_only: true` |
| Split | train / validation / test | 0.70 / 0.15 / 0.15, pending purge support |
| Engineered features | enabled | `true` with the current six features |
| Temporal channels | integration | `none` until a separate vector branch exists |
| Long-term | windows | `[7, 30]` |
| CNN | filters | `[16, 32, 64]` |
| CNN | pools | `[[2,2], [2,2], [2,1]]` |
| LSTM | units / recurrent dropout | 96 / 0.0 |
| Dense | units / dropout | 64 / 0.2 |
| Precision | `mixed_precision` | `float16` |
| Distribution | enabled | `false` |
| Batch | `training.batch_size` | 80; test 64, 80, and 96 |
| Optimizer | Adam learning rate | 0.0008 fallback; otherwise freeze HPO winner |
| Epochs | maximum | 8 |
| Early stopping | monitor / patience | `val_loss` / 2 or 3 |
| LR reduction | factor / patience / minimum | 0.5 / 1 / `1e-5` |
| Class balancing | coarse undersampling | disabled |
| Class weights | compute from train | enabled for one branch of the experiment matrix |
| Recency weighting | enabled | disabled initially |
| Sequential training | enabled | false for final fit |
| HPO | enabled | false in final fit |
| Diagnostics | enabled | false in repeated final fits after a successful snapshot preflight |

The tested short-term shape for this profile is `(256,20,4,10)`. Batch 80 leaves substantial VRAM headroom while matching batch-128 throughput.

### Stage 3: HPO Profile

Run HPO separately from final training. A useful first study is 90 calendar days split into three 30-day windows, three epochs per window, and 12 to 18 trials.

| Parameter | Recommended value |
|---|---|
| `run_mode.mode` | `trial` |
| `training.sequential_training.enabled` | `true`, required by current snapshot trial mode |
| `training.sequential_training.window_days` | 30 |
| `training.epochs` | 3 |
| `hyperparameter_optimization.n_trials` | 12 initially, then 18 after telemetry is stable |
| `hyperparameter_optimization.direction` | `minimize` |
| `hyperparameter_optimization.metric` | `val_loss` |
| `parallel.enabled` | `true` |
| `parallel.resources` | `["gpu:0", "gpu:1", "gpu:2"]` |
| `parallel.max_trials_per_worker_process` | 1 for isolation |
| `parallel.resume_study` | `true` with a unique persistent study name |
| `regime.retry_on_oom` | `true` |
| `regime.max_vram_fraction` | 0.90 |
| `regime.safe_envelope.enabled` | `true` |
| `trial_model_logging.enabled` | `false` |
| Diagnostics | disabled after the preflight snapshot passes |
| Final model registry | disabled for HPO runs |

Recommended first search space:

```yaml
hyperparameter_optimization:
  direction: "minimize"
  metric: "val_loss"
  n_trials: 12
  parallel:
    enabled: true
    resources: ["gpu:0", "gpu:1", "gpu:2"]
    max_trials_per_worker_process: 1
    resume_study: true
  search_space:
    cnn:
      - filters: [16, 24]
      - filters: [32, 48]
      - filters: [64, 96]
    lstm:
      - units: [64, 128]
    learning_rate: [0.0002, 0.002, "log"]
    batch_size: [64, 80, 96]
```

Notes:

- Lists with three batch values are categorical; two values are interpreted as an integer range.
- The log-scaled learning-rate range avoids oversampling the high end of a 10x range.
- Architecture ranges are deliberately narrower than the checked-in stress profiles. Market-data and representation uncertainty is currently much larger than capacity uncertainty.
- If larger first-layer filters are sampled, allow the regime controller to back batch size down.
- Use a second focused study after the first one identifies a stable architecture and learning-rate region.

### Stage 4: Final Training

Use a new production run with HPO disabled and the selected parameters explicitly frozen.

Recommended execution model:

1. Run one final model per GPU as three independent seeds or three controlled ablations.
2. Keep the same frozen train/validation/test boundaries for all three runs.
3. Report mean and spread across seeds rather than selecting the best test result.
4. Use one GPU per model and `distributed.enabled: false`.
5. Set `gpu_visible_devices` separately for each run.

The repository does not currently expose a global model/data seed, so reproducible independent-seed training needs an implementation change before this protocol is rigorous.

### Full-Fidelity Alternative

If preserving `(720,40,4,20)` is required for a direct comparison:

| Parameter | Recommended value |
|---|---|
| Batch | 6 for architecture HPO, 8 for the fixed baseline |
| GPUs | One GPU per job |
| HPO resources | All three GPUs as independent workers |
| Epochs | 3 for HPO, maximum 6 for a first final run |
| HPO range | batch `[4,6,8]`, LR `[0.0001,0.002,"log"]` |
| Sequential window | 20 to 30 days for recovery, not 10 |
| Undersampling | disabled |
| Recency weighting | disabled |
| HPO objective | `val_loss` |

Do not combine six full-range HPO trials and ten-epoch final training in one invocation. Use a bounded HPO range, freeze the winner, and only then launch final training.

## Price-Class Parameters

The checked-in fixed boundaries are:

```yaml
boundaries: [0.15, 0.35, 0.55, 1.5]
```

On the inspected 20-day cache, per-head 20/40/60/80 percentiles of positive excursion magnitude were:

| Head | 20% | 40% | 60% | 80% |
|---|---:|---:|---:|---:|
| Up | 0.155% | 0.331% | 0.575% | 0.987% |
| Down | 0.167% | 0.347% | 0.579% | 0.966% |

The first three checked-in boundaries are therefore plausible for balancing the individual heads on this sample. The 1.5% final boundary deliberately leaves a rarer extreme class. It should not be changed solely to make every class equal.

Recommended policy:

- Retain `[0.15, 0.35, 0.55, 1.5]` as the first benchmark so results remain comparable.
- Recompute excursion quantiles over several volatility regimes before changing it.
- Freeze one boundary set across train, validation, and test so class meaning remains stable.
- Consider separate up/down boundary fitting in a future implementation if asymmetry is material.
- Avoid using `max_intensity` quantiles to define boundaries for two individual heads. On the same sample, max-intensity quantiles were much larger: `[0.465, 0.661, 0.908, 1.428]`.

For trading decisions, class 1 begins at only 0.15%, while the sample backtest uses a 0.1% cost per side, or 0.2% round trip. A strategy based on the current boundaries should generally require probability mass at class 2 or above, corresponding to at least roughly 0.35%, and must still model spread and execution. This does not repair the maximum-excursion versus endpoint-exit mismatch.

## Parameters That Currently Have Limited Or Misleading Effect

| Parameter or behavior | Practical warning |
|---|---|
| `data.time_range.cadence_seconds` | Derives step counts but does not resample source data. |
| `preprocessing.normalization.per_asset` | The snapshot normalization path flattens non-mask tensor positions and has no separate per-asset branch (`training/snapshot_dataset.py:1697-1772`). |
| `model.long_term.resolution_days` | Long-term features directly slice days; this value does not control the calculation in the current path (`preprocessing/long_term_features.py:445-467`). |
| `model.long_term.summary_method` / `ewma_halflife_days` | The current feature computation directly computes window statistics and does not apply these summary controls. |
| `data.ingestion.max_concurrent_chunk_fetches` | Must equal 1; other values are rejected (`data/greptime_client.py:522-539`). |
| `training.debug_max_samples` | Production mode rejects a cap below the available sample count; do not treat it as a production runtime throttle. |
| `hyperparameter_optimization.search_space.batch_size: [2,4]` | Means integer 2, 3, or 4, not categorical 2 or 4. |
| `training.callbacks.early_stopping.patience: 10` with `epochs: 10` | Provides little or no opportunity to stop a window early. |
| `training.sample_weighting.half_life_days: 90` in a 10-day sequential window | Has very little effect within each window. In a non-sequential three-year run it nearly removes the earliest years, which wastes their ingestion cost. |
| `training.runtime.distributed.enabled` | Works technically, but is slower than one GPU for the tested model and input regime. |
| XLA/JIT | Not exposed. The model is compiled without `jit_compile` (`models/cnn_lstm_multiclass.py:449-469`). |

## Improvement Margin

### P0: Highest Value

#### 1. Add A Configurable Sample Stride

Current samples advance one frame at a time (`training/snapshot_dataset.py:379-388`). With a 256-frame input and 128-frame label horizon, adjacent examples are nearly duplicates.

Add a parameter such as:

```yaml
training:
  sample_stride_steps: 4
```

Useful pilot values are 4, 5, or 8 at a 14-second source cadence, producing predictions approximately every 56, 70, or 112 seconds while preserving 14-second detail inside each input window.

Expected gain:

- Approximately 4x to 8x fewer training examples and host reconstructions.
- Much lower redundant compute.
- Better correspondence between nominal sample count and effective independent information.

This is likely the single highest-return implementation change.

#### 2. Move Auxiliary Values To A Separate Vector Branch

The current six engineered and ten temporal values are repeated over every time, level, and book column. Pass them as a compact vector branch and merge them after the LSTM, similar to the existing long-term branch.

Expected gain for the full production representation:

- CNN input channels fall from 20 to 4 while preserving all auxiliary information.
- Host expansion and PCIe transfer shrink by up to 5x for the short-term input.
- First-layer CNN work falls materially.

This is preferable to permanently disabling potentially useful temporal features.

#### 3. Add Purged Walk-Forward Splits

Add explicit purge and embargo durations plus expanding-window or rolling-window evaluation. Store every split boundary in MLflow.

Expected gain:

- No speed improvement.
- Large improvement in the credibility and usability of validation results.

#### 4. Add Epoch-Level Checkpoint And Resume

Current sequential resume saves only after a complete window (`training/pipeline.py:1526-1542`), and non-sequential training has no interruption recovery. Add Keras `BackupAndRestore` or equivalent epoch/batch recovery.

Expected gain:

- Makes one-to-four-day final runs operationally safe.
- Allows non-sequential final training without accepting total restart risk.

#### 5. Align Labels And Trading Evaluation

Choose and expose one of these explicit tasks:

- Barrier-touch classification with first-hit timing and barrier-aware execution.
- Maximum favorable/adverse excursion for risk and order management.
- Endpoint return classification for fixed-horizon direction.
- Multi-task prediction of endpoint return plus up/down excursions.

Expected gain:

- Converts the current broad signal backtest into an evaluation tied to the trained target.

### P1: High Value

#### 6. Parallelize The Input Pipeline

The Threadripper has 32 physical cores, but each measured producer used about one. Replace or augment the Python generator with a native `tf.data` pipeline or a bounded multiprocessing producer pool. Avoid repeating float64 normalization copies where possible.

Expected gain:

- Necessary to feed three GPUs efficiently after tensor-size reductions.
- Likely 2x or more host throughput depending on memory bandwidth and copy reduction.

#### 7. Add Explicit Resampling And Cadence Validation

Store median, mode, percentiles, and irregularity statistics in each manifest. Fail or warn when configured cadence and observed cadence differ beyond tolerance.

Expected gain:

- Prevents silent changes to the model's real input and prediction duration.

#### 8. Add A Run Planner

Before training, report:

- Effective inherited configuration.
- Input shape and bytes per sample.
- Date count, window count, and observed sample count.
- Train/validation/test counts after purge and balancing.
- HPO trials, waves, and devices.
- Estimated epoch and total duration from a short benchmark.
- Snapshot and artifact paths.

The CLI currently exposes only `--config` and `--schema` (`main.py:209-221`). A `plan` or `dry-run` stage would prevent expensive configuration mistakes.

#### 9. Add Reproducibility Controls

Add one global seed and record:

- Python, NumPy, TensorFlow, and Optuna seeds.
- Determinism settings.
- Git commit and dirty state.
- Python/dependency versions.
- GPU/driver/runtime metadata.
- Effective config and data-query hashes.

#### 10. Fix Snapshot Lifecycle Management

The workspace currently contains 81 snapshot directories totaling about 3.4 GiB. Fixed names plus configuration mismatches create many suffixed directories (`training/snapshot_store.py:140-153`). Eviction only considers names prefixed by `root_name` (`training/snapshot_store.py:205-240`), while the existing fixed names use a different prefix.

Use run-specific roots, explicit retention, and post-evaluation cleanup. Be careful with `max_snapshots` during sequential evaluation because evicting early windows can force them to be rebuilt.

### P2: Useful After P0/P1

#### 11. Add NVMe Scratch Storage

An NVMe drive would improve cold snapshot construction, concurrent uncached reads, Optuna SQLite activity, and MLflow artifact staging. It will not solve the current one-core broadcast/normalization bottleneck by itself.

#### 12. Benchmark XLA And Compiled LSTM Alternatives

XLA, a fused recurrent implementation, temporal downsampling before the LSTM, or a temporal-convolution alternative may help. These should be measured only after input representation and stride are fixed.

#### 13. Improve HPO Pruning

Report validation metrics to Optuna each epoch/window and enable pruning. Current trials generally pay for the complete configured workload unless Keras early stopping ends a fit.

## Recommended Use Of The Three GPUs

### Best Current Allocation

```text
GPU 0: HPO trial / final seed A
GPU 1: HPO trial / final seed B
GPU 2: HPO trial / final seed C
CPU:   one producer per GPU, plus snapshot preparation
```

Use all three GPUs for:

- Parallel Optuna trials.
- Representation ablations.
- Different date-regime experiments.
- Independent seeds and ensemble members after seed controls exist.

Do not use all three GPUs by default for:

- One MirroredStrategy final fit of this 0.16 to 0.24 million parameter model.

Revisit mirrored training only if the model becomes much larger, per-replica batch remains substantial, the input pipeline becomes parallel, and a fresh one-versus-three GPU benchmark demonstrates a gain.

## Suggested Experiment Sequence

1. **Cadence audit:** Measure source interval distributions for BTCUSDT and ETHUSDT over several months. Decide whether 14 seconds is canonical or implement 10-second resampling.
2. **Reduced-shape preflight:** Build `(256,20,4,10)`, run one epoch, and verify the measured throughput, VRAM, masks, and labels.
3. **Weighting matrix:** Compare no weights against per-head class weights. Keep undersampling and recency weighting off.
4. **Representation matrix:** Compare 10+10 depth against 20+20, and temporal channels off against on. Use bounded ranges and fixed model capacity.
5. **Bounded HPO:** Run 12 trials over 90 representative days on three independent GPU workers using `val_loss`.
6. **Frozen final fit:** Disable HPO, use one GPU per independent run, maximum eight epochs, and early-stopping patience 2 or 3.
7. **Evaluation gate:** Require per-class metrics, calibration, temporal slices, and simple baselines. Do not promote from accuracy alone.
8. **Trading-task gate:** Use a barrier-aware backtest for excursion labels or retrain on endpoint labels before interpreting PnL.
9. **Scale-up:** Only after the above succeeds, extend from months to the complete 2022-2024 range.

## Final Judgment

The project direction is viable on this workstation. The machine has enough GPU compute, CPU capacity, and RAM. The practical problem is that the current configuration spends most of those resources expanding and repeatedly processing highly overlapping tensors, then multiplies the work across HPO trials and sequential windows.

The checked-in three-year production profile should not be launched unchanged. It combines a cadence mismatch, an unusable undersampling rule, a training-loss HPO objective, only two HPO GPUs, a slow small-batch search, ten epochs per 10-day window, and mirrored final training that does not fit the measured hardware/model scaling.

The recommended immediate path is:

- Correct cadence semantics.
- Use `(256,20,4,10)` with batch 80.
- Disable coarse undersampling and recency weighting.
- Use per-head class weights only as a measured ablation.
- Run `val_loss` HPO as three independent one-GPU workers.
- Freeze HPO results before a non-sequential final fit.
- Treat outputs as excursion forecasts until label and execution evaluation are aligned.

With those parameter choices, serious multi-year experiments move from an estimated many-week or multi-month workload to a roughly two-to-four-day workstation workload. Adding sample stride and a separate auxiliary branch should reduce that further and, more importantly, make the compute spent per unit of independent market information much more defensible.
