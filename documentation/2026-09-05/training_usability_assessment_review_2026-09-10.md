# Code Review of the 2026-09-05 Training Usability Assessment

**Date:** 2026-09-10

**Scope:** Verification of every code-level claim in `training_usability_assessment_2026-09-05.md` against the current source tree; identification of imprecisions, gaps, and additional improvements to the proposed plan.

**Method:** Each `file:line` citation was checked in the source. All re-derivable numbers (parameter counts, label distributions, undersampling results, cadence statistics, excursion percentiles, snapshot sizes, window counts, projection arithmetic) were recomputed from the actual snapshot `snapshots/fullscale_trial_2022to2024_b3fcda89` (20-day May-2022 cache), the model builder, and the checked-in configs.

---

## Overall Verdict

The assessment is **highly accurate**. Every verifiable behavioral claim and every `file:line` citation checked out (within a few lines). No materially wrong claim was found. Specifically re-verified by re-computation:

| Claim | Verdict | Evidence |
|---|---|---|
| Model architecture, 241,434 params (full) / 239,098 (no LT branch) | **Exact** | Built with TF 2.16.2 in the repo venv: `241,434` / `239,098`; reduced `(256,20,4,10)+LT` = `163,706` |
| Max-excursion labels, discretized independently | **Exact** | Recomputed all 67,907 labels from raw `frames_base` midpoints (180-frame horizon): 100.0% (up) / 99.997% (down, boundary float ties) match stored `y_up`/`y_down` |
| Per-head 20/40/60/80 percentiles table | **Exact** | Match when computed over **positive-only** excursions on the full cache: Up `[0.155, 0.331, 0.575, 0.987]`, Down `[0.167, 0.347, 0.579, 0.966]` |
| Max-intensity quantiles `[0.465, 0.661, 0.908, 1.428]` | **Exact** | Recomputed from stored labels |
| Undersampling: 47,534 train samples → 24 kept → fails 500 minimum | **Exact** | See imprecision 1.1 below for the mechanism |
| Cadence: median 14 s, 90th pct 14 s, ~0.2% at 10 s, max 28 s | **Exact** | Measured on `frames_ts`: median 14.0 s, 0.167% exactly 10 s, max 28 s (mode 14 s at 56%, 13 s at 43% — see gap 2) |
| 67,907 samples in ~132 MiB; 11,733-sample two-day January cache | **Exact** | `du` = 132 M; manifest totals match |
| 720x40x4x20 shape, 16 aux + 2 mask channels | **Exact** | Chunk `x_shape` = `[1176, 720, 40, 4, 20]`, `aux_dim` = 16 |
| Batch `[2,4]` = integer range, not categorical | **Correct** | `models/hyperparameter_tuning.py:1999-2008` (2 values → `suggest_int`; 3+ → categorical) |
| `metric: "loss"`, last-history-value extraction, no best-epoch | **Correct** | `config/e2e_fullscale_production_month1.yaml:85-86`; `training/pipeline.py:552-564` (`series[-1]`) |
| Per-replica batch under MirroredStrategy; generator wrapped without parallel mapping | **Correct** | `training/distributed.py:135-137, 240-302` (no `map`/`num_parallel_calls`) |
| No HPO pruning; no global seed; CLI only `--config`/`--schema` | **Correct** | No `trial.report`/pruner in `hyperparameter_tuning.py`; no `tf.random.set_seed` in train path; `main.py:209-221` |
| `end_date` inclusive through 23:59:59; `max_concurrent_chunk_fetches` must be 1 | **Correct** | `data/greptime_client.py:155, 534-539` |
| No purge/embargo; adjacent split ranges | **Correct** | `preprocessing/train_test_split.py:16-45` |
| Backtest: stronger-probability conflict resolution, fixed-index exit, 2x per-side cost | **Correct** | `evaluation/backtesting.py:253-269, 301-399` |
| Snapshot lifecycle: fixed names + suffixes; eviction prefix never matches | **Correct** | `training/snapshot_store.py:140-153, 205-240`; 81 dirs / 3.4 GiB on disk |
| LT `resolution_days` unused; `summary_method`/`ewma_halflife_days` not applied | **Correct** | `_apply_summary_method` is defined but never called (`preprocessing/long_term_features.py:346`); daily aggregation hardcoded to UTC days |
| No `jit_compile`; duty-cycle weights always applied; per-asset normalization is a no-op in the snapshot path | **Correct** | `models/cnn_lstm_multiclass.py:469`; `training/snapshot_dataset.py:1248-1249, 1674-1675`; `:1697-1772, 2049-2081` |
| Sequential: per-window 70/15/15, callbacks recreated per window, resume only after a full window, final model evaluated on every past window | **Correct** | `training/pipeline.py:1434-1550, 1062, 1202-1243`; `main.py:224-250` |
| Trial-mode HPO requires bounded sequential windows; HPO workers are single-GPU with distributed disabled | **Correct** | `main.py:419-429`; `models/hyperparameter_tuning.py:1008-1024` |
| Projection arithmetic (45-78 h/epoch, 19-33 days/10 epochs, 3.0-5.2 h/epoch at 241 samples/s, 89% / 2.66x concurrency) | **Correct** | All ratios recompute from the doc's own measured rates |
| Workstation (3x RTX 3090 24 GiB, Threadripper 3970X 32c/64t, 141 GiB, Python 3.12.3, TF 2.16.2) | **Correct** | Measured on the machine (GPU power caps now all 300 W — see 1.3) |

---

## 1. Imprecisions (minor)

### 1.1 Undersampling table: "Selected target" is the post-truncation kept set, not the plan

The doc's table shows selected targets `[6, 4, 4, 5, 5]` (total 24). The actual sequence in the current code is:

1. `compute_undersample_counts` with `auto` produces the plan **`[6, 6, 6, 6, 6]` = 30** (limited by the 6 class-0 samples).
2. `training/pipeline.py:343-344` then truncates to whole batches of `batch_size` (12, inherited): `usable = (30 // 12) * 12 = 24`.
3. The class breakdown of those first 24 kept indices is `[6, 4, 4, 5, 5]` — exactly the doc's table (reproduced by running `select_undersampled_indices` on the real snapshot with `uniform_time`).

So the doc's numbers are reproducible, but the column label is misleading: 24 is the batch-truncation result, and the plan itself was 30. The conclusion (fails `min_samples_after_balance: 500`) is unaffected.

Related omission: **`min_fraction_after_balance: 0.05` also fails independently** — 24/47,534 = 0.05% < 5% (`training/pipeline.py:386-408`). The doc only mentions the 500-sample minimum. Both guards are part of the blocker.

### 1.2 "A winning HPO batch of 8 becomes global batch 24 on three GPUs"

The checked-in `training.runtime.distributed.resources` is `["gpu:0", "gpu:1"]` (`config/e2e_fullscale_production_month1.yaml:63-66`) — **two** replicas. A batch-8 winner therefore becomes global batch **16**, not 24, unless the distributed resource list is also changed to three GPUs. The general point (HPO batch is not the final global batch under mirroring) is correct; the concrete number is not, for the checked-in config.

### 1.3 GPU power caps

The doc records per-card caps of 300/275/350 W. On 2026-09-10 all three cards report a **300 W** cap. Either the caps were normalized after the assessment or the reading was transient. No impact on conclusions.

### 1.4 Price-class section presents boundary fitting as future work

The doc says "Consider separate up/down boundary fitting in a future implementation if asymmetry is material." In fact, **quantile-based boundary fitting is already implemented**: `targets.price_classes.boundaries: "auto"` with `targets.price_classes.auto: {method: quantile, fit_on: train|full, labeling_criteria: max_intensity|up_intensity|down_intensity, max_samples, random_seed}` (`utils/config_loader.py:286-409`, `training/auto_boundaries.py`). It fits from the disk-cached series before snapshot hashing. What is genuinely still missing is **per-head (asymmetric) fitting** and **global cross-window fitting** (`per_window` must be `true`; "global fit not implemented yet" — `utils/config_loader.py:330-334`).

**Interaction warning the plan must respect:** in sequential mode the auto boundaries are **refit per window**, so class meanings drift from window to window — this contradicts the doc's own recommendation to "freeze one boundary set across train, validation, and test." For non-sequential training, `fit_on: train` is the correct usage; for sequential runs, keep explicit boundaries.

---

## 2. Gaps (items the assessment missed)

1. **Sequential HPO objective is a cross-window weighted average.** In multi-window (sequential) mode, a trial's reported value is the sample-count-weighted average over all windows of the last-epoch per-window metric (`training/pipeline.py:576-592`, accumulated at `:1509-1523`). The doc's "extracts the last history value rather than the best epoch" note is correct per window but does not mention this aggregation. Additionally, in production mode with the checked-in overlay, **every HPO trial runs the full sequential window loop** (the HPO objective calls `run_training_pipeline_result`, `models/hyperparameter_tuning.py:1709-1748`) — i.e., each of the 6 trials is a ~110-window x 10-epoch sequential run (and each window trains only on its 70% train split). The doc's HPO projection (6 trials ≈ 38-65 days as two waves of final-run-sized workloads) is consistent with this, but it should be stated explicitly.

2. **Source cadence is bimodal, not 14 s.** Measured on the 20-day cache: 14 s intervals are 56% of frames, 13 s intervals 43%; mean ≈ 13.5 s. `cadence_seconds: 14` is a reasonable choice, but a 256-frame "nominal" window actually spans ≈ 3,408-3,584 s (55-60 min), and the cadence audit recommended by the plan should report mode + mean + per-asset, not only the median.

3. **Stage 3 HPO profile omits the runtime settings the benchmarks used.** The production overlay comments out `training.runtime` entirely (`config/e2e_fullscale_production_2022to2024.yaml:28-30`), so `gpu_memory_growth: false` is inherited — while the doc's benchmarks ran with memory growth enabled. For three parallel one-GPU workers, the Stage 3 profile should explicitly set `training.runtime.gpu_memory_growth: true` (the default config's own comment recommends it for parallel workers, `config/training_config_default.yaml:278`).

4. **Long-term windows must be shorter than sequential windows.** Stage 3 uses `window_days: 30`. The *default* LT windows `[7, 30, 90]` (`training_config_default.yaml:237`) would be degenerate inside a 30-day window (30/90-day features zero or truncated for essentially all samples; LT features are computed from each window's local history — `training/long_term_context.py`, `training/pipeline.py:662-677`). The month1-inherited `[3, 7]` is fine. Stage 3 should pin `model.long_term.windows_days` explicitly, with every window strictly below `window_days`.

5. **`training.validation_split` is coupled to the split ratios.** It must exactly equal `preprocessing.train_test_split.validation_ratio` or the pipeline raises (`training/pipeline.py:262-266`). This belongs in the "Parameters That Currently Have Limited Or Misleading Effect" table — it silently constrains any split change in the Stage 2 profile.

6. **`selection_policy: "kmeans"` is advertised but unimplemented.** The config docs list `uniform_time | kmeans | random`, but `kmeans` raises `ConfigError("... not implemented yet")` (`training/sample_balancing.py:442-443`).

7. **Config comment bug:** `data.time_range.end_date` is documented as "exclusive" in `config/training_config_default.yaml:75`, but the code treats it as inclusive through 23:59:59 (`data/greptime_client.py:155`). The assessment's body is correct; the config comment should be fixed.

8. **Training generator drops the trailing partial batch.** `build_training_generator_for_indices` truncates the index list to whole batches (`training/snapshot_dataset.py:1526-1531`). The last `< batch_size` training samples are never used. Minor, but it composes with the undersampling truncation in 1.1.

9. **Duty-cycle weighting impact is negligible on this data.** 95.6% of the 20-day cache's samples have duty exactly 1.0 (min 0.9375). The doc's "always applied even when recency weighting is disabled" note is correct, but the practical effect on this cache is a ≤6.25% down-weight on 4.4% of samples — closer to a no-op than a confounder.

10. **Fine-tuning module not mentioned.** `training/fine_tuning.py` (config `training.fine_tuning`: base run id, layer freezing, LR factor) exists. The assessment's improvement list could mention using it to bridge sequential → non-sequential (warm-start the non-sequential final fit from a sequential model) or to warm-start representation ablations.

11. **`data.multi_database` not mentioned.** Time-split dual GreptimeDB connections exist (`config/training_config_default.yaml:46-61`, `data/greptime_client.py:554+`). Relevant to the 2022-2024 ingestion plan if historical and recent data live in different instances.

12. **Observability dashboard postdates the assessment.** `observability/server.py` (HTMX dashboard + `/metrics`, merged April 2026) and the run-state writer (`main.py:302-318`) now exist. The assessment's "operational controls need simplification" verdict was written before it; the dashboard is a partial answer.

13. **Optuna sampler seed is also unexposed.** `requirements.txt` pins `optuna==3.6.1` and the code creates studies without a sampler seed. P1#9 ("record the Optuna seed") should include seeding the TPE sampler, not just `random.seed`. Note dependencies *are* pinned in `requirements.txt`, so recording them is a one-line `pip freeze` in the run metadata.

---

## 3. Additional improvements to the plan (beyond the doc's P0-P2 list)

1. **Implement sample stride at snapshot-build time, not batch time (extends P0#1).** The doc proposes `training.sample_stride_steps` as a training parameter. The better placement is in snapshot construction (the `StreamingSampleBuilder._next_anchor_idx` advance at `training/snapshot_dataset.py:379-384`, and the frame-store anchor emission in `_build_snapshot_chunks`), and it should join the snapshot config hash (`training/snapshot_store.py:79-84`). Then the stored snapshot is physically smaller (fewer `anchor_local_idx` entries, smaller aux) and training, evaluation, diagnostics, and HPO all inherit the stride consistently. A batch-time filter would still pay full storage and would leave evaluation/diagnostics at stride 1.

2. **The auxiliary vector branch (P0#2) is cheaper than it appears.** The frame store already persists the 16 auxiliary values separately (`aux.npy`, `aux_dim: 16` in each chunk manifest entry); broadcasting to T x H x W happens only at batch materialization (`training/snapshot_dataset.py:897-903`). A second model input needs no re-snapshotting — only a model-input and generator change — and it removes 16/20 of the normalization/PCIe work immediately. This makes the doc's "CNN input channels fall from 20 to 4" achievable without touching the storage format.

3. **Purge/embargo (P0#3) is an index-selection change, not a storage change.** Add it in `compute_split_boundaries` / `_resolve_snapshot_training_indices` (`preprocessing/train_test_split.py:16-45`, `training/pipeline.py:251-284`). The frame store already keeps window+horizon frames per chunk (`overlap_frames` in the manifest), so purged samples remain reconstructable. One extra: make `auto_boundaries` with `fit_on: train` respect the same purge so boundary fitting cannot read the validation region.

4. **Concrete fix for snapshot lifecycle (P1#10).** Each manifest already stores `root_name`. Driving `maybe_evict_snapshots` off the manifest `root_name` (instead of the directory-name prefix, `training/snapshot_store.py:212`) would make the 81 accumulated fixed-name directories (3.4 GiB) reclaimable without changing any checked-in config.

5. **Run planner (P1#8) is largely buildable on existing artifacts.** The manifest already stores `time_range`, per-chunk `num_samples`/`x_shape`, `label_stats`, normalization stats, and `config_hash`. A `plan`/`dry-run` CLI stage (today only `--config`/`--schema`, `main.py:209-221`) can report effective shape, bytes/sample, split counts, window counts, and an epoch-time estimate from a short microbenchmark without new infrastructure.

6. **Weighting-matrix note for experiment 3.** `compute_from_train` is per-head inverse-frequency, computed from the train split only (`training/pipeline.py:799-800`, `training/class_weights.py:108-205`), and it multiplies the (always-on) duty-cycle weight (`training/snapshot_dataset.py:1686-1689`). Since duty is ≈ 1.0 on this cache (gap 9), the class-weight branch is a clean single-variable ablation.

7. **Middle option for coarse balancing if ever re-enabled.** On this cache, `auto` + `max_intensity` keeps 30 (→24) samples, while `auto` + `up_intensity` (or `down_intensity`) would keep ≈ 31.5k samples — passing both the 500-sample and 5%-fraction guards. The doc's "disable" recommendation is correct for the initial profile, but the plan should record this escape hatch instead of treating `labeling_criteria: max_intensity` as the only coarse option.

---

## 4. Items checked and confirmed as stated (no change needed)

- `run_mode.mode: "trial"` = HPO-only; production = HPO + final training + evaluation (`main.py:418-463`).
- `debug_max_samples` is rejected in production below dataset size (`utils/production_checks.py:17-21`); it also caps evaluation (`evaluation/evaluator.py:140-141, 1124-1129`).
- HPO safe-envelope / OOM-regime knobs exist as documented (`regime.*` in `config/training_config_default.yaml:365-383`).
- `momentum_steps = momentum_window_seconds // cadence_seconds` (`training/snapshot_dataset.py:3770`) — cadence mismatch does distort momentum windows, as claimed.
- Temporal vector = 3 cyclic-local (x2) + `days_since_start` + 3 one-hot sessions = 10 values (`training/snapshot_dataset.py:3784-3883`); `integration_mode: "none"` is a valid setting (`:352`), so Stage 2's temporal-off profile is directly expressible.
- Early stopping `patience: 10` with inherited `epochs: 10` cannot shorten a fit; `restore_best_weights: true` is the default.
- `evaluation.backtesting.enabled: false` in the production profile — the backtest-semantics critique is about capability, not an enabled run.
- HPO `batch_size` lists of three values are categorical (`[64, 80, 96]` in Stage 3 is fine); the log-scaled LR syntax `[low, high, "log"]` is supported (`models/hyperparameter_tuning.py:1958-1966`).

---

## 5. Bottom line

No change to the assessment's conclusions is required. The four imprecisions (1.1-1.4) are labeling/scoping issues, not errors in the underlying analysis. The most important additions for the execution plan are:

1. State explicitly that production-mode HPO trials each run the full sequential window loop, and that the trial objective is a cross-window weighted average of last-epoch values (gap 1).
2. Pin `model.long_term.windows_days` below `window_days` in any sequential HPO profile (gap 4), and keep explicit class boundaries in sequential runs if `boundaries: "auto"` is ever used (1.4).
3. Add `training.runtime.gpu_memory_growth: true` to the Stage 3 profile so it matches the benchmark conditions (gap 3).
4. Place sample-stride, the auxiliary vector branch, and purge/embargo as index/build-time changes with concrete insertion points (section 3), and fix snapshot eviction via the manifest `root_name` (section 3.4).
5. Record the 13/14 s cadence bimodality and per-asset cadence in the preflight manifest before freezing `cadence_seconds` (gap 2).
