# Consolidated Execution Plan — Training Usability Remediation

**Date:** 2026-09-10

**Sources:**
- `training_usability_assessment_2026-09-05.md` (the assessment)
- `training_usability_assessment_review_2026-09-10.md` (the code review; all `file:line` citations re-verified)

**Revision:** 2026-09-10 — revised per
`training_usability_execution_plan_final_review_2026-09-10.md`: all its
verified findings (M1–M6, N1–N4, P1–P2) applied.

This plan consolidates the assessment's blockers, parameter strategy, and
P0–P2 improvement list with every correction and gap the review identified,
into a single ordered set of execution phases sized for agent workers.

---

## How To Use This Plan (for worker agents)

**What a phase is.** A phase is a self-contained work package sized to be
completed in **one ~250k-token context session**. A worker should be able to
pick up a phase cold, read only the files listed in its *Context load*, make
the changes in its *Write scope*, and finish by meeting its *Acceptance*
criteria — without needing to read the rest of the repository.

**Rules for every phase:**
1. Complete exactly one phase per session. Do not start the next phase in the
   same session.
2. Scope reading to the phase's *Context load*. A 250k window holds a phase,
   not a codebase — do not read the whole repo.
3. **Do not edit source files without explicit approval** (repo rule). The
   orchestrator/user approves the edit set before a worker modifies code.
4. **Do not commit** unless the orchestrator explicitly asks. Verification
   (tests + a smoke run) is part of the phase; committing is a separate,
   orchestrator-controlled step.
5. Follow repo conventions: YAML-first configuration, fail-fast validation,
   `ConfigError` for config problems, `logging` (no `print`), type hints on
   public functions, `from __future__ import annotations`, module docstring,
   import order stdlib then third-party then local.
6. After any logic change, run the full suite:
   `CI=true python -m unittest discover -s tests -v`.
 7. Each phase lists a **context budget note** so a worker can confirm the work
    fits a single session before starting.
 8. **Execute Gate G0 (GPU availability) before any step that trains or
    benchmarks on the GPUs** (see Gate G0 below). Phases that run live GPU
    work are marked "Gate G0 required". A worker must not launch a live GPU
    run while the LLM-serving service still holds VRAM.

**Milestones and phase map.** Phases are grouped into five milestones. The
recommended order is linear; the two marked *optional-early* phases (2 and 12)
can be pulled forward if a working baseline is wanted before all code lands.

| Milestone | Purpose | Phases |
|---|---|---|
| M1 — Unblock | A practical profile that runs today, no code changes | 1, 2 |
| M2 — Efficiency code (P0) | Cut redundant compute; make final fits safe | 3, 4, 5, 6 |
| M3 — Infrastructure (P1) | Planner, reproducibility, lifecycle, host pipeline | 7, 8, 9, 10 |
| M4 — Alignment + experiments | Label/execution alignment, then the experiment matrix | 11, 12, 13 |
| M5 — Scale-up + P2 | Full 2022–2024 range; remaining nice-to-haves | 14, 15 |

**Global invariants (apply to all phases):**
- The `frame_store_v1` snapshot on-disk format is stable. Phases 3 and 4 add
  *new* fields/inputs but must not break existing snapshots or the manifest
  contract. Where a change invalidates a snapshot, it must join the snapshot
  config hash (`training/snapshot_store.py:79-84`) so a new directory is
  created rather than a cached one being corrupted.
- `cadence_seconds` derives step counts but does **not** resample source rows
  (`training/snapshot_dataset.py:293-320`). Treat cadence as a data-meaning
  decision (Phase 2), not a tuning knob.
- Source cadence is **bimodal** (review gap 2): 14 s is about 56% of frame
  intervals, 13 s about 43%, mean about 13.5 s, max 28 s on the 20-day cache.
  A 256-frame "nominal" window spans about 3,408–3,584 s (55–60 min), not a
  fixed 3,584 s.
- The two labels are **max excursions** (up and down), discretized
  independently, over the horizon — not endpoint returns. All evaluation must
  respect that until Phase 11 re-aligns the task.
- **The three GPUs are shared with a resident LLM-serving service.** The cards
  are preloaded with LLM model weights (served by `llama-server`), so free VRAM
  is near zero while the service runs. Live training requires the service to be
  temporarily stopped. This is **Gate G0** below and is a hard prerequisite for
  every live GPU phase.

---

## Gate G0 — GPU Availability (operational prerequisite for live GPU work)

The three RTX 3090s are not free resources. A resident LLM-serving service
(`llama-server`) holds their VRAM while serving models. The assessment already
observed this: with the service running, GPUs 1 and 2 had almost no free VRAM
and a production-shaped model could not initialize on GPU 2. **No live GPU test
may start until Gate G0 passes.**

**Nature of the gate.** This is an **operational/coordination gate, not a code
change.** Stopping the service takes the LLM offline, so it requires explicit
approval from the orchestrator/user. A worker must **not** silently stop the
service; it requests approval, waits, then proceeds.

**Steps (run before the first live GPU step of a phase):**
1. Identify what is holding GPU memory: `nvidia-smi` (compute-app list) to
   confirm the LLM-serving process and which cards it occupies.
2. **Request approval** from the orchestrator/user to stop the LLM-serving
   service, stating which cards and for how long. Do not stop it without that
   approval.
3. Stop the service (e.g. stop the `llama-server` process / its manager) and
   let the VRAM release.
4. **Verify** with `nvidia-smi` that all three cards show near-zero used VRAM.
   Target free VRAM per card: at least ~20 GiB for a full-shape batch-8 job, or
   at least ~14 GiB for the reduced-shape batch-80 job, plus headroom.
5. **Verify** TensorFlow sees the cards: `tf.config.list_physical_devices('GPU')`
   returns three devices.
6. **Record** the pre-run GPU state (used/free VRAM per card, power caps — all
   report 300 W per review 1.3, driver, TF version) in the run metadata.
7. **After** the live work completes, restart the service per the
   orchestrator's decision (or leave it stopped for the session). Note the
   restart in the phase report.
8. **Multi-day runs (>24 h, e.g. Phases 13/14 final fits):** the upfront
   approval (step 2) must state an end time, and continued exclusive GPU
   occupation must be **re-confirmed at least daily** (or at each window
   boundary). If the LLM service must reclaim the GPUs mid-run, do **not** kill
   the training process: use the Phase 6 checkpoint/resume to pause cleanly at
   the next epoch/window boundary, return the GPUs, and resume once the service
   is stopped again. Record every pause/resume in the run metadata.

**Gate G0 passes when:** the service is stopped (with approval), `nvidia-smi`
shows near-zero used VRAM on the cards the run will use, and TensorFlow lists
all expected GPUs. If any card is still occupied and cannot be freed, the run
must be rescheduled to the free cards only (adjust `parallel.resources` /
`gpu_visible_devices`) or deferred — do not launch into occupied VRAM.

**Phases that require Gate G0:** 2, 4, 6, 10, 12, 13, 14, and 15 (the XLA /
compiled-LSTM benchmark sub-item) — i.e. every phase with a live training or
GPU-benchmark step. Phases 1, 3, 5, 7, 8, 9, and 11 are config/code/test-only
and need no GPU (their unit tests run on CPU); if a worker chooses to
smoke-test them on GPU, Gate G0 still applies.

---

## M1 — Unblock: A Practical, Immediately Runnable Profile

### Phase 1 — Practical efficiency profile config (parameter-only)

**Goal.** Produce a new checked-in config that fixes every immediate blocker
using parameters only (no code). This unblocks all later phases.

**Depends on:** nothing.

**Context load (read):**
- `config/training_config_default.yaml` (full baseline + comments)
- `config/e2e_fullscale_production_month1.yaml`
- `config/e2e_fullscale_production_2022to2024.yaml`
- `config/validation_schema.yaml`
- `training/sample_balancing.py:442-443` (confirm `kmeans` unimplemented)
- `data/greptime_client.py:155` (`end_date` inclusive)

**Write scope:** one new file, e.g. `config/e2e_practical_efficiency_2024.yaml`
(name consistent with the `e2e_*` convention; do **not** edit the existing
production overlays in this phase).

**Steps.**
1. Base the new config on `training_config_default.yaml`.
2. **Blocker 1 — disable coarse undersampling.**
   `preprocessing.class_balancing.enabled: false`.
   - Record in a comment the review's corrected mechanism (review 1.1): the
     `auto`/`max_intensity` plan is 30 samples, truncated to whole batches to
     24 (`training/pipeline.py:343-344`), and it fails **both**
     `min_samples_after_balance: 500` **and** `min_fraction_after_balance:
     0.05` (24/47,534 = 0.05%) (`training/pipeline.py:366-408`).
   - Record the escape hatch (review improvement 7): if coarse balancing is
     ever re-enabled, `labeling_criteria: up_intensity` (or `down_intensity`)
     keeps about 31.5k samples and passes both guards; `max_intensity` does not.
3. **Blocker 2 — HPO objective.** `hyperparameter_optimization.direction:
   "minimize"`, `metric: "val_loss"` (not `loss`). Keep
   `restore_best_weights: true` for final training.
4. **Blocker 3 — separate HPO from final.** This phase's config is the
   *final-fit* profile (HPO off). A separate HPO profile is produced in
   Phase 13. Do not run HPO and final training in one invocation.
   - **Blocker 4 — non-sequential final profile** (assessment blocker 4,
     "Sequential Evaluation"): set `training.sequential_training.enabled:
     false` explicitly. The default (`config/training_config_default.yaml:289`)
     already does this; state it explicitly here per the repo's
     no-hidden-defaults convention. Sequential mode is reserved for
     online-adaptation experiments, not the final fit.
5. **Efficiency shape** (assessment Stage 2): `cadence_seconds: 14`
   (provisional until Phase 2 confirms), `visible_window_seconds: 3584`
   (256 frames), `prediction_horizon_seconds: 1792` (128 frames),
   `depth_levels: 100`, hybrid raw/bins `10/10`, temporal `integration:
   "none"`, `include_mask_channel: true`, normalization `min_max` /
   `fit_on_train_only: true`, split `0.70/0.15/0.15`.
6. **Model:** CNN filters `[16,32,64]`, pools `[[2,2],[2,2],[2,1]]`, LSTM 96
   (recurrent dropout 0.0), dense 64 / dropout 0.2, `mixed_precision:
   float16`.
7. **Runtime:** `distributed.enabled: false`, one GPU per job (set
   `gpu_visible_devices` per launch), `batch_size: 80` (test 64/80/96),
   epochs 8, early stopping `val_loss` patience 2–3, LR reduction
   `0.5 / 1 / 1e-5`, Adam LR `0.0008` fallback.
8. **Weighting defaults:** recency `sample_weighting.enabled: false`;
   `class_weights.compute_from_train` off in this base file (one experiment
   branch enables it — Phase 12). Note duty-cycle weights are always applied
   but are about a no-op on this data (95.6% of samples duty = 1.0, review
   gap 9).
9. **Long-term:** pin `model.long_term.windows_days` explicitly so every LT
   window is **strictly below** `window_days` in any sequential profile
   (review gap 4). For the non-sequential final fit, `[7, 30]` is fine.
10. **Coupling guard (review gap 5):** set `training.validation_split` to
    exactly `preprocessing.train_test_split.validation_ratio` (0.15) — the
    pipeline raises otherwise (`training/pipeline.py:262-266`).
11. **Boundaries:** keep explicit `boundaries: [0.15, 0.35, 0.55, 1.5]`.
    Do **not** use `targets.price_classes.boundaries: "auto"` in any
    sequential run — auto boundaries are refit per window and drift class
    meaning (review 1.4). `fit_on: train` is correct only for non-sequential.
12. **HPO profile runtime (review gap 3):** when the Phase 13 HPO profile is
    written, it must set `training.runtime.gpu_memory_growth: true` (the
    2022–2024 overlay comments out `training.runtime`, so `false` is
    inherited, while the benchmarks ran with growth enabled). Note this here
    so it is not lost.
13. Add a header comment block documenting the corrected facts: HPO batch 8
    under the checked-in 2-GPU list becomes global **16**, not 24 (review
    1.2); `end_date` is inclusive through 23:59:59 (review gap 7);
    `selection_policy: kmeans` is unimplemented (review gap 6); the generator
    drops the trailing partial batch (`training/snapshot_dataset.py:1526-1531`).

**Acceptance.**
- `python main.py --config config/e2e_practical_efficiency_2024.yaml --schema
  config/validation_schema.yaml` passes config load + schema validation (it
  may then fail at a later runtime stage if data/GPU are unavailable; config
  load is the bar for this phase).
- `CI=true python -m unittest discover -s tests -v` still passes (no code
  changed, so this is a regression guard).
- A short comment in the YAML maps each blocker (1–4) to the key that fixes it.

**Deliverable.** One new YAML + header comments. No source edits.

**Context budget note.** About 5 config files (a few hundred lines each) plus
two small code ranges. Comfortably within one session.

---

### Phase 2 — Data & cadence preflight (run + report) *optional-early*

**Goal.** Confirm source cadence and data health on a small range, build the
reduced-shape snapshot, run one epoch, and freeze `cadence_seconds`. This
produces the first real end-to-end signal and a preflight manifest.

**Depends on:** Phase 1.

**Gate G0 required.** The one-epoch preflight is the first live GPU run of the
plan; execute Gate G0 (stop the LLM-serving service, free the VRAM, verify)
before launching it.

**Context load (read):**
- The Phase 1 config.
- `data/greptime_client.py` (cadence, `end_date`, `max_concurrent_chunk_fetches`).
- `training/snapshot_dataset.py:293-320` (cadence to step counts), `:379-388`
  (anchor advance), `:414-427` (labels).
- `training/manifest_utils.py` and an existing manifest under
  `snapshots/fullscale_trial_2022to2024_b3fcda89/manifest.json`.
- `diagnostics/` (what the preflight reports).

**Write scope:** a preflight report under `documentation/` (date-stamped);
optionally a small read-only analysis script under `tools/` (no pipeline code).

**Steps.**
1. **Cadence audit (review gap 2).** Measure source frame-interval
   distributions for BTCUSDT and ETHUSDT over several months / volatility
   regimes. Report **mode + mean + percentiles + per-asset** (not just the
   median). Confirm the 13/14 s bimodality and decide whether 14 s is
   canonical or whether 10 s requires explicit resampling (Phase 9 adds the
   validation; this phase makes the decision).
2. **Preflight run.** Use a 2–7 day range, `training.epochs: 1`,
   `diagnostics.enabled: true` (`num_samples: 50`),
   `data.validation.fail_on_invalid: false` for discovery, the Phase 1
   efficiency shape. Record:
   - actual batch throughput (samples/s) and peak VRAM;
   - aligned-frame density, gaps, confidence masks, duty cycle;
   - per-head class counts (0–4) and joint up/down counts;
   - a complete train/validation/evaluation pass.
3. **Freeze cadence.** Set `cadence_seconds` to the confirmed canonical value
   (14 if representative). Update the Phase 1 config and re-validate.
4. **Preflight manifest.** Record date count, observed sample count,
   train/val/test counts, cadence statistics, and the effective shape into the
   report. Multi-year planning must use this manifest, not the date range
   (assessment "Data Volume").

**Acceptance.**
- Cadence decision is written down with the per-asset evidence.
- One-epoch reduced-shape run completes; throughput/VRAM are recorded and
  compared to the benchmark expectation (about 241 samples/s at batch 80,
  about 13 GiB peak). Pass band: throughput within +/-15% of 241 samples/s
  (i.e. 205-277) and peak VRAM <= 15 GiB. Outside the band,
  record the measured values and the cause (the preflight snapshot's fitted
  statistics differ from the benchmark's representative normalization
  arithmetic) before freezing the profile.
- Preflight manifest committed as a dated doc.

**Deliverable.** Preflight report + manifest + updated `cadence_seconds`.

**Context budget note.** This is a *run + analyze* phase. The agent's session
work is: prepare the run (config already exists), launch it, then read logs +
manifest and write the report. The multi-hour run itself is not an agent task.
Fits one session of agent work.

---

## M2 — Efficiency Code (P0, Highest Value)

### Phase 3 — Configurable sample stride (at snapshot-build time)

**Goal.** Add `sample_stride_steps` so anchors advance by N frames instead of
1, implemented at **snapshot construction** (not batch time) so the stored
snapshot is physically smaller and training/eval/diagnostics/HPO all inherit
the stride consistently (review improvement 1).

**Depends on:** Phase 1 (config shape exists). Independent of 4/5/6.

**Context load (read):**
- `training/snapshot_dataset.py:293-388` (builder, `_next_anchor_idx`,
  `_build_sample`), and the frame-store anchor emission in
  `_build_snapshot_chunks`.
- `training/snapshot_store.py:30-84` (`_snapshot_config_subset`,
  `compute_config_hash`).
- `utils/config_loader.py` + `config/validation_schema.yaml` (new key).
- `training/snapshot_dataset.py:1526-1531` (trailing-batch interaction).

**Write scope:** `training/snapshot_dataset.py`, `training/snapshot_store.py`,
`utils/config_loader.py`, `config/validation_schema.yaml`,
`config/training_config_default.yaml`, a test file.

**Steps.**
1. Add config key `training.sample_stride_steps` (int, default 1 = current
   behavior). Validate: integer >= 1.
2. In the streaming builder, advance
   `self._next_anchor_idx += self._sample_stride_steps` instead of `+= 1`
   (`training/snapshot_dataset.py:384`). Keep the buffer-eviction logic
   correct for the larger step (verify `earliest_needed` still releases the
   right frames).
3. **Verify, do not assume, that the frame-store path inherits the stride:**
   `_build_snapshot_chunks` (`training/snapshot_dataset.py:2972-3400`) drives
   the same `StreamingSampleBuilder` (via `_create_sample_builder`, `:3497`,
   and `add_snapshot`, `:3335`), so the single step-2 edit is expected to cover
   both paths. Confirm with a test that the stored snapshot reflects the stride
   (reduced `anchor_local_idx` entries); if a separate anchor-emission site
   exists, patch it there.
4. **Join the snapshot config hash** (`training/snapshot_store.py:79-84` /
   `_snapshot_config_subset`) so a stride change creates a new snapshot
   directory instead of reusing a stride-1 one.
5. Document pilot values (4, 5, 8 at 14 s cadence, predictions every
   56/70/112 s) and the expected 4x–8x reduction in training examples and host
   reconstructions.
6. Add/extend tests: stride=1 reproduces the current sample count; stride>1
   reduces the count by the expected factor and keeps window/horizon
   integrity.

**Acceptance.**
- `sample_stride_steps: 1` is byte-for-byte the old behavior (regression).
- `sample_stride_steps: 4` on the 20-day cache yields about 1/4 the samples
  and a smaller snapshot directory.
- Full test suite passes.

**Deliverable.** Stride parameter + build-time application + hash join + tests.

**Context budget note.** Two medium files + schema + one test module. One
session.

---

### Phase 4 — Auxiliary values to a separate vector branch

**Goal.** Move the 16 anchor-level auxiliary channels (6 engineered + 10
temporal) out of the 4-D tensor and into a compact second model input merged
after the LSTM (like the existing long-term branch). CNN channels fall 20 to 4
with **no re-snapshotting** — the frame store already persists `aux.npy`
(`aux_dim: 16`) and broadcasting happens only at batch time
(`training/snapshot_dataset.py:897-903`) (review improvement 2).

**Depends on:** Phase 1. Independent of 3/5/6 (can run before or after stride).

**Gate G0 required.** The forward-pass / throughput check runs on a GPU.

**Context load (read):**
- `models/cnn_lstm_multiclass.py:102-194`, `:399-469` (model build, inputs).
- `training/snapshot_dataset.py:897-903` (aux broadcast), `:3753-3893`
  (aux/temporal construction), `:1648-1694` / `:2049-2094` (batch
  materialize, normalize).
- `training/snapshot_dataset.py` generator signatures (what the model input
  tuple must carry).
- `training/pipeline.py:824-891` (`_initialize_snapshot_training_model` — the
  only place the model is built; `build_cnn_lstm_model` at `:881-887`) and
  `:976-1050` (`_build_snapshot_training_data` — chooses the raw generator vs
  `wrap_generator_as_dataset`).
- `training/distributed.py:190-235` (`_build_output_signature_no_lt` /
  `_with_lt` — strictly 1-or-2-tensor input signatures) and `:240-302`
  (`wrap_generator_as_dataset`).
- `utils/config_loader.py:878-1065` + `config/validation_schema.yaml` (the new
  toggle must be registered in the schema's dotted keys).

**Write scope:** `models/cnn_lstm_multiclass.py`, `training/snapshot_dataset.py`,
`training/pipeline.py` (model-init + data-builder argument wiring),
`training/distributed.py` (fail-fast guard), `utils/config_loader.py` /
`config/validation_schema.yaml` (an `aux_vector_branch: true` toggle), a test
file.

**Steps.**
1. Add a model-input flag (e.g. `model.input_representation.aux_vector_branch`)
   defaulting to `false` (current behavior).
2. When enabled, add a second model input of shape `(batch, 16)` (or the
   current aux dim) and merge its embedding with the LSTM output before the
   dense/head stack, mirroring the long-term branch.
3. Change the generator to emit the aux vector as a separate input (no
   broadcast to T x H x W) and shrink the tensor channels from 20 to 4 when the
   branch is on (base + mask channels only).
4. Keep normalization correct: the aux branch is not part of the flattened
   position normalization path.
5. Keep the change model-side (no re-snapshot). If it does alter what is
   consumed from the store, join it into the snapshot/model config hash.
6. **The distributed (mirrored) path is out of scope for this phase:** the
   `tf.data` output signatures (`training/distributed.py:190-235`) have no
   provision for a third aux tensor. Add a fail-fast `ConfigError` when
   `aux_vector_branch: true` combines with
   `training.runtime.distributed.enabled: true` (the practical profile runs
   with `distributed.enabled: false`). Extend the signature builders only if
   mirrored + aux is genuinely needed later.
7. Add tests: branch on, the model builds with 4 CNN channels and a 16-dim
   second input and a forward pass runs; branch off, behavior is unchanged;
   the distributed+aux `ConfigError` fires.

**Acceptance.**
- Branch on: CNN input `(...,4,...)`, aux as a separate `(batch,16)` input,
  forward pass works, and host expansion/PCIe for the short-term input is
  reduced (verify the materializer no longer broadcasts aux to T x H x W).
- Branch off: identical to current behavior.
- `aux_vector_branch: true` + `distributed.enabled: true` raises the
  `ConfigError` guard.
- Full test suite passes.

**Deliverable.** Aux vector branch (toggle) + generator change + tests.

**Context budget note.** Model file + generator ranges + schema + tests. One
session; the largest single P0 phase.

---

### Phase 5 — Purged walk-forward splits

**Goal.** Add explicit **purge/embargo** to the chronological split plus
expanding/rolling walk-forward evaluation, storing every split boundary in
MLflow. This is an **index-selection change, not a storage change** — the
frame store keeps window+horizon frames per chunk (`overlap_frames`), so
purged samples remain reconstructable (review improvement 3).

**Depends on:** Phase 1. Independent of 3/4/6.

**Context load (read):**
- `preprocessing/train_test_split.py:16-45` (`compute_split_boundaries`).
- `training/pipeline.py:251-284` (`_resolve_snapshot_training_indices`).
- `training/auto_boundaries.py` + `utils/config_loader.py:286-409` (so
  `fit_on: train` respects the same purge).
- `utils/config_loader.py:878-1065` + `config/validation_schema.yaml` (the new
  `purge_steps`/`embargo_steps` keys must be registered in the schema's dotted
  keys under the `preprocessing` section).
- `mlflow_integration/` (where split boundaries get logged).

**Write scope:** `preprocessing/train_test_split.py`, `training/pipeline.py`,
`training/auto_boundaries.py`, `utils/config_loader.py` /
`config/validation_schema.yaml`, `mlflow_integration/`, a test file.

**Steps.**
1. Add config keys, e.g. `preprocessing.train_test_split.purge_steps` and
   `embargo_steps` (in frames), default 0 = current behavior. Validate >= 0.
2. In `compute_split_boundaries` / `_resolve_snapshot_training_indices`,
   remove the `purge_steps` samples immediately before the validation and test
   boundaries (label-horizon purge) and apply `embargo_steps` as the
   independence gap. Document the two recommended settings for the
   cadence-aligned profile: label-only purge = 128 frames (about 30 min); full
   input+label separation = 384 frames (about 90 min).
3. **Boundary-fitting purge:** make `auto_boundaries` with `fit_on: train`
   exclude the purged region so boundary fitting cannot read the validation
   region (review improvement 3).
4. Add expanding-window (and optionally rolling) walk-forward evaluation; log
   every split boundary + purge/embargo values to MLflow.
5. Add tests: purge=0 reproduces current boundaries; purge>0 removes exactly
   the expected indices at both boundaries; walk-forward produces
   non-overlapping, chronologically ordered folds.

**Acceptance.**
- Purge=0 is the old behavior (regression).
- With the 128-frame purge, the train/val boundary gap equals 128 frames and
  no train sample's horizon overlaps validation.
- MLflow run records the boundaries and purge/embargo values.
- Full test suite passes.

**Deliverable.** Purge/embargo + walk-forward + MLflow logging + tests.

**Context budget note.** Split util + pipeline range + auto_boundaries +
MLflow + tests. One session.

---

### Phase 6 — Epoch-level checkpoint and resume

**Goal.** Make 1–4 day final runs operationally safe. Current sequential resume
saves only after a complete window (`training/pipeline.py:1526-1542`) and
non-sequential training has **no** interruption recovery. Add Keras
`BackupAndRestore` (or equivalent) epoch/batch recovery for both paths.

**Depends on:** Phase 1. Independent of 3/4/5.

**Gate G0 required.** The interrupt/resume test trains a fit on a GPU.

**Context load (read):**
- `training/pipeline.py:1434-1550` (sequential loop + resume), `:1062`
  (`callbacks = create_callbacks(config)`), `:1250-1319`
  (`_fit_snapshot_model_once`, `model.fit` at `:1304`), `:1320-1345`
  (`_fit_snapshot_model_once_result`), `:1681-1728`
  (`_run_snapshot_training_pipeline_result` — the non-sequential dispatch
  where BackupAndRestore must also be wired).
- `training/callbacks.py:13-95` (callback construction).
- `main.py:224-250` (post-train evaluation).
- `utils/config_loader.py:878-1065` + `config/validation_schema.yaml` (the
  `training.checkpoint.*` keys must be registered in the schema's dotted keys).

**Write scope:** `training/pipeline.py`, `training/callbacks.py`,
`utils/config_loader.py` / `config/validation_schema.yaml` (a
`training.checkpoint.*` block), a test file.

**Steps.**
1. Add a `training.checkpoint` config block (enabled, directory,
   save-every-epochs or per-batch), validated on load.
2. Wire `tf.keras.callbacks.BackupAndRestore` (or an equivalent
   epoch/batch-granular mechanism) into both the sequential per-window fit and
   the non-sequential fit.
3. Ensure resume restores the epoch/batch position, optimizer state, and (where
   applicable) the data index position — not just weights.
4. Keep the existing per-window sequential resume as a fallback; the new
   mechanism must not regress it.
5. Add tests: a fit interrupted mid-epoch resumes from the checkpoint without
   re-running completed batches (mock/cancel the fit to simulate interruption).

**Acceptance.**
- A non-sequential fit can be killed mid-epoch and restarted from the
  checkpoint with no lost progress.
- Sequential per-window resume still works.
- Full test suite passes.

**Deliverable.** Epoch/batch checkpoint+resume (both paths) + config + tests.

**Context budget note.** Pipeline ranges + callbacks + config + tests. One
session.

---

## M3 — Infrastructure (P1)

### Phase 7 — Run planner (`plan` / `dry-run` CLI stage)

**Goal.** Add a `plan`/`dry-run` CLI stage that reports, before training: the
effective inherited config, input shape + bytes/sample, date/window/sample
counts, train/val/test counts **after purge and balancing**, HPO trials/waves/
devices, and an epoch-time estimate from a short microbenchmark — built almost
entirely on existing manifest artifacts (review improvement 5). The CLI today
exposes only `--config` and `--schema` (`main.py:209-221`).

**Depends on:** Phase 5 for the post-purge counts acceptance criterion.
Scaffolding (CLI flag, report skeleton, manifest/config counts) can proceed in
parallel with Phase 5 — report pre-purge counts until Phase 5 lands. 3/4
optional.

**Context load (read):**
- `main.py:209-221` (`_parse_args`), `:224-463` (run flow).
- `training/snapshot_store.py:79-164` (manifest/config subset).
- `training/manifest_utils.py` — generic `load_manifest`/`write_manifest`/
  `sanitize_component` only (39 lines, no schema knowledge).
- The manifest field schema is constructed in `training/snapshot_dataset.py`:
  chunk entries (`num_samples`/`x_shape`/`aux_dim`) at `:600-675` (read/verify)
  and `:3220-3240` (build, inside `_build_snapshot_chunks` `:2972-3400`);
  `label_stats` save/load at `:1956-2020`; top-level keys (`time_range`,
  `normalization_stats`, `config_hash`, `root_name`, ...) per
  `snapshots/fullscale_trial_2022to2024_b3fcda89/manifest.json`.
- `training/pipeline.py:251-284` (split resolution), `:287-419` (undersampling
  resolution) — to reuse for post-purge/post-balancing counts.
- **7b only (optional integration):** `observability/server.py` (HTMX
  dashboard + `/metrics`; 3,400+ lines — read only the `/metrics` + run-state
  routes you need) and the run-state writer (`main.py:302-318`,
  `observability.run_state`) — existing operational context (review gap 12).
  The planner should reuse/align with these rather than duplicate them; the
  dashboard is a partial answer to the assessment's "operational controls need
  simplification" verdict.

**Write scope:** a new `training/run_planner.py` (or `tools/plan.py`),
`main.py` (new `--plan` flag / subcommand), `utils/config_loader.py` /
`config/validation_schema.yaml` if new keys are needed, a test file.

**Steps.**
1. Add `--plan` (or a `plan` subcommand) to `main.py`; it loads + validates
   config, then stops before any GPU/training work.
2. Implement the planner to read the config + (if present) the snapshot
   manifest and emit a structured report: effective config, shape,
   bytes/sample, date count, window count, observed sample count,
   train/val/test counts (post-purge, post-balancing), HPO trials/waves/
   devices, snapshot + artifact paths.
3. Add an optional short microbenchmark (a few warmed steps on a tiny batch)
   to estimate samples/s and thus epoch/total duration; keep it opt-in and
   fast.
4. Add tests: the planner output matches hand-computed values for a known
   config + manifest.

**Acceptance.**
- `python main.py --config <cfg> --plan` prints the full report and exits 0
  without initializing GPUs or training.
- Reported counts match the Phase 5 split + balancing logic.
- Full test suite passes.

**Deliverable.** `plan`/`dry-run` stage + planner module + tests.

**Context budget note.** `main.py` + scoped pipeline/manifest ranges + a new
module + tests. One session for 7a (planner core: shape/bytes/date/window/
sample/split counts from config + manifest). If the 7b dashboard/run-state
integration (which requires deep, non-adjacent reads of
`observability/server.py`, 3,400+ lines) is in scope, split into two sessions:
7a planner core, 7b integration + microbenchmark.

---

### Phase 8 — Reproducibility controls

**Goal.** Add one global seed and record the full provenance set so
independent-seed training (Phase 13/14) is rigorous. This is currently missing
(a P1 item); the review adds that the **Optuna TPE sampler seed must also be
seeded**, not just `random.seed` (review gap 13).

**Depends on:** Phase 1.

**Context load (read):**
- `main.py` (run start, where metadata is gathered).
- `models/hyperparameter_tuning.py:1910, 2184, 2234, 2559` (the four
  `optuna.create_study` sites — **all four** need a seeded TPE sampler for the
  "reproducible trial suggestions" acceptance to hold).
- `utils/config_loader.py:878-1065` + `config/validation_schema.yaml` (the
  `reproducibility` block must be registered: a brand-new top-level section is
  silently unvalidated unless added to the schema's `sections:` map).
- `mlflow_integration/` (where metadata is logged).
- `requirements.txt` (pinned deps — recording is a one-line `pip freeze`).

**Write scope:** `main.py`, `models/hyperparameter_tuning.py`,
`mlflow_integration/`, `utils/config_loader.py` /
`config/validation_schema.yaml` (a `reproducibility` block), a test file.

**Steps.**
1. Add a `reproducibility.seed` (int) config key and register the new
   top-level `reproducibility` section in `config/validation_schema.yaml`
   (add it to the `sections:` map, or nest the key under `training` instead) —
   otherwise it passes schema validation silently unchecked. When set, seed
   Python `random`, NumPy, and TensorFlow (`tf.random.set_seed`). Document that
   full determinism is not guaranteed but the seed is recorded.
2. **Seed the Optuna TPE sampler** in `run_hyperparameter_search` (review gap
   13) so trial suggestions are reproducible.
3. Record in run metadata: Python/NumPy/TF/Optuna versions and seeds,
   determinism settings, git commit + dirty state, `pip freeze` output (deps
   are pinned in `requirements.txt`), GPU/driver/runtime metadata, effective
   config hash and data-query hash.
4. Add tests: setting the seed produces a recorded seed in metadata; the
   Optuna sampler is constructed with the seed.

**Acceptance.**
- A run with `reproducibility.seed` set logs all seed + provenance fields.
- Two runs with the same seed + data produce identical trial suggestions
   (Optuna) — or at minimum, the seed is verifiably applied.
- Full test suite passes.

**Deliverable.** Global seed (incl. Optuna TPE) + provenance metadata + tests.

**Context budget note.** `main.py` + HPO + MLflow + config + tests. One
session.

---

### Phase 9 — Snapshot lifecycle + cadence validation + small config corrections

**Goal.** Bundle the small, high-value correctness fixes:
(a) make snapshot eviction reclaim the accumulated fixed-name directories via
the manifest `root_name` (review improvement 4);
(b) add cadence validation that warns/fails when configured vs observed
cadence diverge (P1#7);
(c) fix the `end_date` "exclusive" config comment (review gap 7);
(d) clarify the unimplemented `kmeans` policy error (review gap 6);
(e) note the trailing-partial-batch drop
(`training/snapshot_dataset.py:1526-1531`).

**Depends on:** Phase 2 (cadence decision) for step (b) only; steps (a), (c),
(d), (e) need no other phase (Phase 1's config is not required).

**Context load (read):**
- `training/snapshot_store.py:140-153`, `:205-240` (naming + eviction, prefix
  at `:212`), manifest `root_name` field.
- `training/manifest_utils.py` — generic load/save/sanitize only (39 lines);
  cadence stats go into the manifest dict constructed in
  `training/snapshot_dataset.py:3220-3240` (chunk build).
- `utils/config_loader.py:878-1065` + `config/validation_schema.yaml` (only if
  the warn/fail cadence tolerance becomes a config key — register it in the
  schema).
- `config/training_config_default.yaml:75` (the wrong comment).
- `training/sample_balancing.py:442-443` (`kmeans` raises).

**Write scope:** `training/snapshot_store.py` (eviction),
`training/snapshot_dataset.py` (cadence stats into the chunk-build manifest at
`:3220-3240`; one-line trailing-batch comment at `:1526-1531`),
`config/training_config_default.yaml` (comment), `training/sample_balancing.py`
(clearer guard/message), `utils/config_loader.py` /
`config/validation_schema.yaml` (only if the cadence tolerance is a config
key), a test file.

**Steps.**
1. **Eviction by `root_name` (review improvement 4):** drive
   `maybe_evict_snapshots` off the manifest `root_name` instead of the
   directory-name prefix (`training/snapshot_store.py:212`) so the 81
   accumulated fixed-name directories (3.4 GiB) become reclaimable without
   changing checked-in configs. Preserve the current-directory exclusion and
   the sequential-evaluation caveat (evicting early windows forces rebuilds —
   keep `max_snapshots` conservative there).
2. **Cadence validation (P1#7):** store median/mode/percentiles/irregularity
   in each manifest; warn (or fail, configurable) when `cadence_seconds`
   differs from observed cadence beyond a tolerance.
3. **Config comment fix (review gap 7):** correct
   `config/training_config_default.yaml:75` to state `end_date` is
   **inclusive** through 23:59:59.
4. **`kmeans` guard (review gap 6):** keep `kmeans` raising, but make the error
   message explicit ("not implemented yet") and document the valid set
   (`uniform_time | random`) in the schema/comment.
5. Add a one-line code comment at `training/snapshot_dataset.py:1526-1531`
   documenting the intentional trailing-partial-batch drop.
6. Add tests: eviction now considers a fixed-name dir whose manifest
   `root_name` matches; cadence validation fires on a synthetic mismatch.

**Acceptance.**
- Eviction reclaims matching-`root_name` fixed-name directories; the active
  snapshot is never evicted.
- A cadence mismatch produces the configured warning/failure.
- The `end_date` comment is correct; the `kmeans` error is clear.
- Full test suite passes.

**Deliverable.** Lifecycle + cadence-validation + doc/guard corrections + tests.

**Context budget note.** Several small edits across a few files + tests. One
session (the "oddments" phase).

---

### Phase 10 — Parallelize the input pipeline

**Goal.** Raise host throughput so three GPUs can be fed efficiently after the
tensor-size reductions. Each measured producer used about 1 CPU core; the
Threadripper has 32. Replace or augment the Python generator with a native
`tf.data` pipeline or a bounded multiprocessing producer pool, and avoid
repeated float64 normalization copies (P1#6).

**Depends on:** Phase 3 and/or 4 (so the pipeline is built on the final tensor
/ aux layout). This is the riskiest P1 phase — sequence it after the P0 tensor
changes are stable.

**Gate G0 required.** The before/after throughput benchmark uses a GPU.

**Context load (read):**
- `training/snapshot_dataset.py:1510-1694` (generator, materialization),
  `:2049-2094` (normalization), `:815-924` (frame-store reconstruction).
- `training/distributed.py:240-302` (how the generator is wrapped — no
  parallel mapping today).
- `training/pipeline.py` (where `steps_per_epoch` / generator are wired).

**Write scope:** `training/snapshot_dataset.py` (or a new
`training/snapshot_tfdata.py`), `training/distributed.py`,
`training/pipeline.py`, a test file.

**Steps.**
1. Profile the current single-core producer to confirm the bottleneck
   (broadcast + float64 normalization copies).
2. Implement a bounded multiprocessing producer pool **or** a native
   `tf.data` pipeline that reads the memmapped frame store, reconstructs
   windows, and emits `(x, aux, y_up, y_down, sample_weight)` batches with
   `num_parallel_calls` and prefetch.
3. Remove redundant float64 normalization copies (normalize in-place / once).
4. Keep the existing Python generator as a fallback behind a config flag.
5. Add tests: the new pipeline yields the same tensors as the old generator
   for the same indices (numeric equivalence); throughput improves.

**Acceptance.**
- On the reduced shape, host throughput roughly doubles (or better) versus the
  single-core baseline, measured on the same snapshot.
- Output tensors match the old generator (regression).
- Full test suite passes.

**Deliverable.** Parallel input pipeline (toggle) + tests + a before/after
throughput note.

**Context budget note.** Generator/materialize ranges + distributed wrapper +
tests. One session; the largest P1 phase — if it overruns, split the
`tf.data` implementation and the benchmark into two sessions.

---

## M4 — Label/Execution Alignment + Experiment Matrix

### Phase 11 — Align labels and trading evaluation (P0#5)

**Goal.** Choose and expose one explicit task so evaluation is tied to the
trained target instead of the current generic signal backtest (which resolves
conflicting up/down predictions by the stronger probability and exits at a
fixed index — `evaluation/backtesting.py:253-269`, `:301-399`, misaligned with
max-excursion labels).

**Depends on:** Phase 5 (purged/walk-forward splits) so the evaluation is
credible.

**Context load (read):**
- `evaluation/backtesting.py:253-269`, `:301-399` (current execution rule).
- `evaluation/evaluator.py:459-580` (metric extraction/computation/payload),
  `:843-1094` (`evaluate_model`), `:1095-2104` (`evaluate_snapshot_model` —
  the snapshot entry point the task evaluation must hook), `:2105-2164`
  (`_compute_class_metrics`, `_finalize_calibration`).
- `training/snapshot_dataset.py:414-427`, `:3730-3750` (label construction).
- `config/validation_schema.yaml` (evaluation section).

**Write scope:** `evaluation/backtesting.py` (or a new
`evaluation/barrier_backtest.py`), `evaluation/evaluator.py`,
`utils/config_loader.py` / `config/validation_schema.yaml`, a test file.

**Steps.**
1. Add an `evaluation.task` config key selecting one explicit task:
   - `barrier_touch` (with first-hit timing + barrier-aware execution),
   - `mfe_mae` (max favorable/adverse excursion for risk/order management),
   - `endpoint_return` (fixed-horizon direction), or
   - `multi_task` (endpoint + up/down excursions).
2. Implement the selected task's evaluation so the backtest/execution rule
   matches the label (e.g. barrier-aware entry/exit for excursion labels, not
   fixed-index exit).
3. Keep the old backtest available but clearly labeled as a signal sanity
   check, not profitability evidence.
4. Add tests: for a synthetic path, the barrier-aware execution hits the
   correct barrier in the correct order; endpoint classification matches the
   label.

**Acceptance.**
- The configured task is used end-to-end and its metrics are logged.
- The old fixed-index backtest is demoted to "sanity check" status.
- Full test suite passes.

**Deliverable.** Task-aligned evaluation + config + tests.

**Context budget note.** Evaluation files + schema + tests. One session. If
`multi_task` is chosen, scope it to the two-head excursion task first and
defer endpoint to a follow-up.

---

### Phase 12 — Weighting + representation experiment matrix (run + report) *optional-early*

**Goal.** Run the two bounded experiment matrices on the practical profile and
report per-class precision/recall, calibration, and signal frequency. These are
parameter-only runs (plus the Phase 11 task if it has landed).

**Depends on:** Phase 1 (profile); Phase 5 (purge) and Phase 11 (task) for
credible metrics; Phase 3 (stride) to cut cost.

**Gate G0 required.** Every matrix arm is a live one-GPU training run.

**Context load (read):**
- The Phase 1 config (+ Phase 11 task config).
- `training/class_weights.py:108-205` (per-head inverse-frequency weights,
  computed from the train split only).
- `training/snapshot_dataset.py:1686-1689` (class weights multiply the always-on
  duty-cycle weight).
- `evaluation/evaluator.py:459-580` (metric definitions: `_extract_eval_targets`,
  `_compute_prediction_metrics`, `_build_evaluation_metrics_payload`);
  `evaluation/backtesting.py:253-269, :301-399` only if the Phase 11 task
  config is in play.

**Write scope:** 2–4 run configs under `config/` (one per matrix arm); a report
under `documentation/`.

**Steps.**
1. **Weighting matrix** (clean single-variable ablation — duty is about 1.0 on
   this cache, review gap 9 / improvement 6):
   - arm A: no class weights (`compute_from_train: false`);
   - arm B: per-head class weights (`compute_from_train: true`);
   - keep undersampling and recency weighting **off** in both arms.
2. **Representation matrix** (bounded ranges, fixed capacity):
   - depth 10+10 vs 20+20;
   - temporal channels off vs on.
3. Run each arm as an independent one-GPU job (use all three GPUs), with the
   same frozen split boundaries.
4. Report per-class precision/recall, calibration, and signal frequency per
   arm; do **not** promote on accuracy alone.

**Acceptance.**
- All matrix arms complete with metrics logged to MLflow.
- A report ranks arms by the Phase 11 task metric and notes calibration.
- (If Phase 8 landed) seeds are recorded per arm.

**Deliverable.** 2–4 configs + a dated experiment-matrix report.

**Context budget note.** *Run + analyze*. Agent work per arm: write the config,
launch, then (in a later session) read MLflow and write the report. Each arm's
agent work fits one session; the multi-hour runs do not.

---

### Phase 13 — Bounded HPO + frozen final fit + evaluation gate (run + report)

**Goal.** Run a bounded HPO study on a representative range, freeze the winner,
then run the non-sequential final fit, and gate promotion on the evaluation.

**Depends on:** Phase 1, Phase 5 (purge), Phase 6 (resume), Phase 8 (seed),
Phase 11 (task); Phase 3/4 to cut trial cost.

**Gate G0 required.** HPO trials and the three final seed fits are live GPU
runs across all three cards.

**Context load (read):**
- The Phase 1 config (base for the HPO profile).
- `models/hyperparameter_tuning.py:1008-1024`, `:1709-1748`, `:1958-1966`,
  `:1999-2008` (trial model, objective, LR syntax, batch encoding).
- `main.py:418-463` (trial vs production mode).
- `training/pipeline.py:576-592`, `:1509-1523` (cross-window weighted-average
  objective — review gap 1).

**Write scope:** one HPO config under `config/` (trial mode), one final-fit
config (production mode, HPO off, winner frozen), a report under
`documentation/`.

**Steps.**
1. **HPO profile** (trial mode). A useful first study: 90 calendar days in
   three 30-day windows, 3 epochs per window, 12 trials (then 18 after
   telemetry is stable).
   - `run_mode.mode: "trial"`, `sequential_training.enabled: true`,
     `window_days: 30`, `epochs: 3`.
   - **Pin `model.long_term.windows_days`** so every LT window is strictly
     below 30 (e.g. `[3, 7]`) (review gap 4).
   - **Set `training.runtime.gpu_memory_growth: true`** to match the benchmark
     conditions (review gap 3).
   - `direction: "minimize"`, `metric: "val_loss"`, `parallel.resources:
     ["gpu:0","gpu:1","gpu:2"]`, `max_trials_per_worker_process: 1`,
     `resume_study: true` (unique study name), `regime.retry_on_oom: true`,
     `max_vram_fraction: 0.90`, `safe_envelope.enabled: true`,
     `trial_model_logging: false`, diagnostics off, final-model registry off.
    - Search space: CNN filters `[16,24]`/`[32,48]`/`[64,96]`, LSTM units
      `[64,128]`, `learning_rate: [0.0002, 0.002, "log"]`, `batch_size:
      [64, 80, 96]` (three values = categorical, not a range).
    - **If Phase 3 (stride) or Phase 4 (aux branch) has landed before this
      profile is written:** stride changes trial wall-clock (re-derive the
      epoch-time estimate); the aux branch changes CNN input channels from 20
      to 4, so re-derive the CNN filter search space for the narrower input
      (capacity-per-channel changes) — run one short sanity benchmark on the
      final shape before committing the 12 trials.
   - **State explicitly** (review gap 1): in production mode each HPO trial
     runs the full sequential window loop, and the trial objective is the
     sample-count-weighted average of the last-epoch per-window metric. Trial
     mode + bounded windows keeps each trial to one window.
2. **Freeze the winner.** Write a final-fit config (production mode, HPO off)
   with the winning parameters explicitly set, `distributed.enabled: false`,
   one GPU per run, non-sequential.
3. **Final fit.** Run one model per GPU as three independent seeds (or three
   controlled ablations) with the same frozen boundaries; report mean + spread
   (requires Phase 8 seed). Keep `restore_best_weights: true`.
4. **Evaluation gate.** Require per-class metrics, calibration, temporal
   slices, and simple baselines. Do not promote from accuracy alone.

**Acceptance.**
- HPO study completes 12 trials; the best trial is identified by `val_loss`.
- The frozen final-fit config reproduces the winner's parameters.
- Three seed runs complete; mean + spread are reported against the gate.

**Deliverable.** HPO config + frozen final-fit config + a dated HPO/final-fit
report.

**Context budget note.** *Run + analyze*. Agent work: write the two configs,
launch HPO, then (later) launch the final fits and write the report. Each
agent session fits; the multi-day runs do not.

---

## M5 — Scale-Up + Remaining P2

### Phase 14 — Scale up to the full 2022–2024 range (run + report)

**Goal.** Only after the Phase 13 evaluation gate passes, extend from the
representative months to the complete 2022–2024 range (1,096 calendar days,
roughly 2.6–4.5 million training samples at observed density).

**Depends on:** Phase 13 gate passed; Phase 3 (stride), Phase 5 (purge),
Phase 6 (resume), Phase 10 (host pipeline) all landed so the full range is a
workstation-scale workload (about 2–4 days, not many weeks).

**Gate G0 required.** The full-range final fit is a multi-day live GPU run —
Gate G0 step 8 (stated end time, daily re-confirmation, checkpoint-based
mid-run pause) applies.

**Context load (read):**
- The frozen final-fit config (Phase 13 winner).
- `data/greptime_client.py:554+` and `config/training_config_default.yaml:46-61`
  (`data.multi_database` — review gap 11; use if historical and recent data
  live in different GreptimeDB instances).
- The Phase 2 preflight manifest (density planning).
- `training/fine_tuning.py` (review gap 10; optionally warm-start the
  non-sequential final fit from a sequential model to bridge the two).

**Write scope:** a full-range config under `config/` (derived from the Phase 13
frozen config, wider `data.time_range`); a report under `documentation/`.

**Steps.**
1. Derive the full-range config from the frozen Phase 13 winner; set
   `data.time_range` to 2022-01-01 through 2024-12-31 (remember `end_date` is
   inclusive). If the data spans instances, enable `data.multi_database`.
2. Re-run the Phase 2-style preflight manifest on the full range to confirm
   density and cadence before committing.
3. Launch the non-sequential final fit (with Phase 6 checkpoint/resume); use
   the other two GPUs for concurrent ablations or seeds.
4. Run the Phase 11 task evaluation + the evaluation gate on the full range.
5. Report the full-range result against the representative-month result.

**Acceptance.**
- Full-range preflight manifest confirms density/cadence.
- Final fit completes (or resumes after an interruption) within the projected
  2–4 day window.
- Evaluation gate is re-evaluated on the full range.

**Deliverable.** Full-range config + a dated scale-up report.

**Context budget note.** *Run + analyze*. Agent work: derive the config, run
the preflight, launch, then report. Multi-day runs are not agent work.

---

### Phase 15 — P2 items (NVMe, XLA/compiled-LSTM benchmark, HPO pruning)

**Goal.** Land the remaining lower-priority items once P0/P1 are stable.

**Depends on:** M2 + M3 complete (input representation and stride fixed first,
so the XLA benchmark measures a stable target).

**Gate G0 required** for the XLA / compiled-LSTM benchmark sub-item (15a); the
HPO-pruning sub-item (15b) is code + synthetic tests and needs no GPU.

**Context load (read):**
- `models/cnn_lstm_multiclass.py:449-469` (model compiled without
  `jit_compile`).
- `models/hyperparameter_tuning.py` (no `trial.report` / pruner today; study
  creation sites at `:1910, 2184, 2234, 2559`).
- `training/pipeline.py:552-592` (per-epoch metric reporting).

**Write scope:** as needed across `models/`, `models/hyperparameter_tuning.py`,
`training/pipeline.py`; an infra note for NVMe.

**Steps.**
1. **NVMe scratch (P2#11):** document and, if procured, wire an NVMe drive for
   cold snapshot construction, concurrent uncached reads, Optuna SQLite, and
   MLflow artifact staging. Note it does not fix the one-core
   broadcast/normalization bottleneck by itself (Phase 10 does).
2. **XLA / compiled LSTM (P2#12):** benchmark XLA (`jit_compile`), a fused
   recurrent implementation, temporal downsampling before the LSTM, or a
   temporal-convolution alternative — only after representation + stride are
   fixed. Record samples/s and VRAM per option.
3. **HPO pruning (P2#13):** report validation metrics to Optuna each
   epoch/window (`trial.report`) and enable a pruner so underperforming trials
   stop early instead of paying the full workload.

**Acceptance.**
- NVMe note documents the target layout (or the drive is wired).
- XLA/compiled-LSTM benchmark reports a measured winner (>=10% throughput
  improvement, or >=2 GiB peak-VRAM reduction at equal throughput, versus the
  non-compiled baseline on the same shape) or explicitly confirms no gain
  beyond that band.
- HPO trials are pruned early on a synthetic failing trial.
- Full test suite passes.

**Deliverable.** Infra note + benchmark report + pruning implementation + tests.

**Context budget note.** Three sub-items; if needed, split into 15a (NVMe +
XLA benchmark, mostly measurement) and 15b (HPO pruning, code). Each fits one
session.

---

## Appendix A — Review Corrections Folded Into This Plan

| # | Review item | Where it is applied |
|---|---|---|
| 1.1 | Undersample plan is 30, truncated to 24; `min_fraction_after_balance` also fails | Phase 1 step 2 |
| 1.2 | HPO batch 8 = global **16** on the checked-in 2-GPU list, not 24 | Phase 1 step 13 |
| 1.3 | GPU power caps now all report 300 W | Global invariants (no action) |
| 1.4 | Quantile boundary fitting already exists; per-head + global cross-window missing; auto boundaries drift per window in sequential | Phase 1 step 11, Phase 5 step 3 |
| gap 1 | Sequential HPO objective = cross-window weighted average; each trial runs the full window loop | Phase 13 step 1 |
| gap 2 | Cadence is bimodal 13/14 s; audit must report mode + mean + per-asset | Global invariants, Phase 2 step 1 |
| gap 3 | Stage 3 HPO profile must set `gpu_memory_growth: true` | Phase 1 step 12, Phase 13 step 1 |
| gap 4 | Pin `model.long_term.windows_days` below `window_days` | Phase 1 step 9, Phase 13 step 1 |
| gap 5 | `training.validation_split` must equal `validation_ratio` | Phase 1 step 10 |
| gap 6 | `kmeans` advertised but unimplemented | Phase 1 step 13, Phase 9 step 4 |
| gap 7 | `end_date` documented "exclusive" but is inclusive | Phase 1 step 13, Phase 9 step 3 |
| gap 8 | Generator drops the trailing partial batch | Phase 1 step 13, Phase 9 step 5 |
| gap 9 | Duty-cycle weighting is a near no-op on this data | Phase 1 step 8, Phase 12 step 1 |
| gap 10 | Fine-tuning module can bridge sequential to non-sequential | Phase 14 context |
| gap 11 | `data.multi_database` for dual GreptimeDB connections | Phase 14 step 1 |
| gap 12 | Observability dashboard now exists (partial answer to "simplify operational controls") | Phase 7 context (reuse `observability/server.py` + run-state writer) |
| gap 13 | Optuna TPE sampler seed unexposed; deps are pinned | Phase 8 step 2–3 |
| imp 1 | Sample stride at snapshot-build time, join the config hash | Phase 3 |
| imp 2 | Aux vector branch is cheap (no re-snapshot) | Phase 4 |
| imp 3 | Purge/embargo is index-selection; boundary fitting must respect it | Phase 5 |
| imp 4 | Snapshot eviction via manifest `root_name` | Phase 9 step 1 |
| imp 5 | Run planner buildable from existing manifest artifacts | Phase 7 |
| imp 6 | Weighting matrix is a clean single-variable ablation | Phase 12 step 1 |
| imp 7 | `up_intensity`/`down_intensity` escape hatch for coarse balancing | Phase 1 step 2 |

## Appendix B — Bottom Line (carried from the review)

1. Production-mode HPO trials each run the full sequential window loop, and the
   trial objective is a cross-window weighted average of last-epoch values
   (gap 1).
2. Pin `model.long_term.windows_days` below `window_days` in any sequential HPO
   profile (gap 4); keep explicit class boundaries in sequential runs if
   `boundaries: "auto"` is ever used (1.4).
3. Add `training.runtime.gpu_memory_growth: true` to the Stage 3 HPO profile so
   it matches the benchmark conditions (gap 3).
4. Place sample-stride, the auxiliary vector branch, and purge/embargo as
   index/build-time changes with concrete insertion points, and fix snapshot
   eviction via the manifest `root_name` (review section 3).
5. Record the 13/14 s cadence bimodality and per-asset cadence in the preflight
   manifest before freezing `cadence_seconds` (gap 2).

## Appendix C — Phase Dependency Graph

```text
Phase 1 (config) ─┬─> Phase 2 (preflight/cadence)
                  ├─> Phase 3 (stride)      ─┐
                  ├─> Phase 4 (aux branch)   ├─> Phase 10 (host pipeline)
                  ├─> Phase 5 (purge)  ──────┤
                  ├─> Phase 6 (resume)       │
                  ├─> Phase 8 (seed)         │
                  └─> Phase 9 (lifecycle) <── (needs Phase 2)
Phase 5 ─> Phase 7 (planner)
Phase 5 + Phase 11 ─> Phase 12 (matrix) <── Phase 3/8
Phase 1+5+6+8+11 ─> Phase 13 (HPO + final fit)
Phase 13 gate ─> Phase 14 (scale-up)
M2 + M3 ─> Phase 15 (P2)
```

Phases 3, 4, 5, 6, 8, and 9 are mutually independent and each needs only Phase
1 (Phase 9 additionally needs Phase 2 for its cadence step only), so they can
be executed in any order or across parallel workers. Phase 7 can be scaffolded
in parallel, but its post-purge counts acceptance requires Phase 5 first. The
Phase 1 fan-out in the graph reflects milestone ordering, not extra hard
dependencies beyond each phase's own "Depends on" line.

