# Final Readiness Review — Training Usability Execution Plan (2026-09-10)

**Reviewer scope:** Independent, adversarial final check of
`training_usability_execution_plan_2026-09-10.md` against the live source
tree (not the two source docs). Read-only: no files modified, no GPU/training
work executed. One CPU-only unit-test run and several CPU-only static
Python checks (model construction, pure-function calls) were used to verify
numeric claims, per the reviewer's constraints.

---

## 1. Verdict

**READY-WITH-MINOR-CHANGES.**

The plan's factual foundation is exceptionally strong: of 25+ `file:line`
citations and numeric claims independently re-verified against the current
source tree (more than double the required 12), all but a handful were
byte-for-byte exact, including three independently rebuilt model parameter
counts (241,434 / 239,098 / 163,706) and the undersampling arithmetic
(30 → truncated to 24). Every Appendix A row is traceable to a real phase,
and every P0–P2 item and review correction/gap is folded in exactly once.
However, the deeper pass found: one flatly wrong context-load citation
(Phase 6), one file mislabeled as containing manifest-schema fields it does
not contain (Phases 7 and 9), one phase (Phase 4) whose Write scope omits
files it actually needs to touch for full correctness, one direct internal
contradiction between a phase's stated "Depends on" and the Appendix C
summary paragraph, and one phase (Phase 1) whose Acceptance criterion
references a 4th "blocker" that Steps never explicitly instantiates. None of
these are fatal to the plan's structure, but all are concrete, fixable, and
should be corrected before phases 1, 4, 6, 7, and 9 are handed to a cold
worker.

---

## 2. Issues

### MAJOR

**M1 — Wrong context-load citation in Phase 6.**
Plan location: Phase 6, "Context load," `training/pipeline.py:1434-1550, :1062,
:1202-1243 (callback creation, non-sequential fit)`.
Evidence: `training/pipeline.py:1202-1243` is the entire body of
`_resolve_sequential_windows` (starts `def _resolve_sequential_windows` at
1202, ends `return windows` at 1243) — a date-range chunking helper, not
callback creation and not the non-sequential fit path. The actual
non-sequential fit dispatch is `_run_snapshot_training_pipeline_result`
(`training/pipeline.py:1681-1730`), which calls `_fit_snapshot_model_once_result`
(`:1320-1345`, itself calling `_fit_snapshot_model_once` at `:1250-1319`,
`model.fit(**fit_kwargs)` at `:1304`). Callback creation is correctly cited
separately at `:1062` (`callbacks = create_callbacks(config)`).
Fix: replace `:1202-1243` with `:1250-1345` (fit-once, non-sequential) and
`:1681-1730` (non-sequential dispatch / where BackupAndRestore must also be
wired for the single-window path).

**M2 — `training/manifest_utils.py` does not contain "manifest fields" (Phases 7, 9).**
Plan location: Phase 7 "Context load" — *"`training/manifest_utils.py`
(manifest fields: `time_range`, per-chunk `num_samples`/`x_shape`,
`label_stats`, normalization stats, `config_hash`)"*; Phase 9 "Context load" —
*"`training/manifest_utils.py` (where cadence stats would be stored)."*
Evidence: `training/manifest_utils.py` is 39 lines total and contains only
`sanitize_component`, `load_manifest`, `write_manifest` — generic JSON
persistence helpers with no schema knowledge. The actual manifest field
schema is constructed in `training/snapshot_dataset.py`: per-chunk
`num_samples`/`x_shape`/`aux_dim` at `:600-675` and in `_build_snapshot_chunks`
(`:2972-3400`); `label_stats` read/write at `save_label_stats_to_manifest`
(`:1956-1994`) / `load_label_stats_from_manifest` (`:1996-2020`). Verified
directly against a real manifest (`snapshots/fullscale_trial_2022to2024_b3fcda89/manifest.json`):
top-level keys are `assets, chunks, complete, config_hash, config_snapshot,
created_at, label_stats, normalization_stats, root_name, series, snapshot_name,
time_range, version`; none of this is defined in `manifest_utils.py`.
Fix: repoint Phase 7/9 Context load to `training/snapshot_dataset.py`
(chunk/label_stats ranges above) and keep `manifest_utils.py` only for the
generic load/save/sanitize calls it actually provides.

**M3 — Phase 4's Write scope/Context load omit files needed for correctness, and misattributes where the model is built.**
Plan location: Phase 4 "Context load" — *"`models/hyperparameter_tuning.py`
(if HPO builds the model, keep in sync)"*; "Write scope" — `models/
cnn_lstm_multiclass.py`, `training/snapshot_dataset.py`, `utils/config_loader.py`
/ `config/validation_schema.yaml`, a test file.
Evidence: `grep -rn "build_cnn_lstm_model"` shows the model is instantiated
in exactly one place, `training/pipeline.py:881-887`
(`_initialize_snapshot_training_model`, `:824-891`), which also assembles the
`input_shape`/`long_term_input_dim` args and (in `_build_snapshot_training_data`,
`:976-1050`) decides whether to hand `model.fit` a raw Python generator
(`dist_ctx is None`, i.e. `distributed.enabled: false`) or a `tf.data.Dataset`
built by `wrap_generator_as_dataset` (`training/distributed.py:240-302`),
whose output signature is fixed by `_build_output_signature_no_lt`/
`_build_output_signature_with_lt` (`training/distributed.py:190-235`) — a
strictly 1-or-2-tensor signature with no provision for a third (aux-vector)
tensor. `models/hyperparameter_tuning.py` never imports or calls
`build_cnn_lstm_model` at all (confirmed by grep) — HPO reuses the pipeline
end-to-end, so the parenthetical is misleading.
Risk: with `training.runtime.distributed.enabled: true` (mirrored path),
turning on `aux_vector_branch` will hit an uncovered code path
(`training/distributed.py`) and fail or silently mis-shape data; Phase 4's
Acceptance criteria never exercise the distributed combination.
Fix: add `training/pipeline.py:824-891, 976-1050` and
`training/distributed.py:190-302` to Context load and Write scope, drop the
`hyperparameter_tuning.py` parenthetical, and add either a third
`_build_output_signature_with_lt_aux` variant or an explicit
`ConfigError` guard rejecting `aux_vector_branch: true` with
`distributed.enabled: true` until that path is implemented.

**M4 — Appendix C directly contradicts Phase 7's own stated dependency.**
Plan location: Phase 7 header, line 509: *"**Depends on:** Phase 5 (so it can
report post-purge counts); 3/4 optional."* vs. Appendix C closing paragraph
(end of file): *"Phases 3, 4, 5, 6, 7, 8, and 9 are mutually independent (each
depends only on Phase 1, with Phase 9 also needing Phase 2 and Phase 7 also
benefiting from Phase 5), so they can be executed in any order or across
parallel workers."*
Evidence: these are two irreconcilable claims about the same edge — Phase 7's
own section says Phase 5 is a hard prerequisite ("Depends on"), the summary
says it is a soft "benefit" and that Phase 7 "can be executed in any order."
An orchestrator following only the summary paragraph (a reasonable thing to
do when parallel-dispatching phases 3–9 to independent workers) could start
Phase 7 before or concurrently with Phase 5, at which point Phase 7's own
Acceptance criterion ("Reported counts match the Phase 5 split + balancing
logic") is unverifiable because Phase 5 doesn't exist yet.
Fix: remove Phase 7 from the "mutually independent" set in the closing
paragraph, or soften Phase 7's own "Depends on" line to match (e.g. "Phase 5
recommended before finalizing acceptance; can be scaffolded in parallel").

**M5 — Phase 1's Acceptance criterion references a 4th blocker that Steps never instantiate.**
Plan location: Phase 1, Acceptance, third bullet: *"A short comment in the
YAML maps each blocker (1–4) to the key that fixes it."* Steps only introduce
"**Blocker 1**" (step 2, undersampling), "**Blocker 2**" (step 3, HPO
objective), "**Blocker 3**" (step 4, HPO/final separation).
Evidence: the source assessment enumerates four "Immediate Configuration
Blockers" (1. coarse undersampling, 2. HPO objective, 3. HPO/final batch
semantics, 4. Sequential Evaluation) plus a separate "No Purge/Embargo" item
deferred to Phase 5. Blocker 4 ("use non-sequential training for the final
profile") is never given an explicit `key: value` step in Phase 1 — it is
only alluded to via the phrase "non-sequential final fit" in steps 9 and 11.
It happens to work by silent inheritance: `training/training_config_default.yaml:289`
already sets `sequential_training.enabled: false`, and Phase 1's new file is
based on that default (step 1). This conflicts with the repo's own
"no hidden defaults; require explicit values" convention (AGENTS.md,
Configuration) and leaves the "(1–4)" acceptance bullet unfulfillable exactly
as written (a worker cannot label a 4th blocker-to-key mapping that was never
introduced).
Fix: add an explicit "**Blocker 4 — use non-sequential training for the final
profile.** `training.sequential_training.enabled: false` (inherited from the
default; set explicitly here per repo no-hidden-defaults convention)." step.

**M6 — Gate G0 has no protocol for a multi-day run that must temporarily give GPUs back to the LLM service.**
Plan location: "Gate G0" section (steps 1–7) and Phase 14 ("Gate G0 required.
The full-range final fit is a multi-day live GPU run.").
Evidence: Gate G0's steps are a single approve-stop-verify-run-restart cycle
per phase. Phase 14 explicitly projects a 2–4 day final fit; Phase 13 also
runs "the three final seed fits" as multi-day work. Nothing in Gate G0 or
Phase 14 addresses what happens if the LLM-serving service needs its GPUs
back mid-run (e.g., an operational/business need arises on day 2 of a 4-day
job), even though Phase 6's checkpoint/resume mechanism is exactly the tool
that would make a clean pause-service-resume-training cycle possible. There
is also no stated re-approval cadence (e.g., "re-confirm daily") for
multi-day exclusive occupation of shared production hardware — the plan's
own step 2 asks for a single upfront "for how long" estimate.
Fix: add a Gate G0 sub-step for phases >24h: "if the service must reclaim
GPUs mid-run, use the Phase 6 checkpoint to pause cleanly rather than kill
the process," and require daily (or otherwise periodic) re-confirmation of
continued approval for multi-day exclusive occupation, not a single upfront
grant.

### MINOR

**N1 — Phase 9's "Depends on" line is narrower than Appendix C's graph.**
Plan location: Phase 9, "Depends on: Phase 2 (cadence decision) for (b)."
vs. Appendix C: `Phase 1 (config) ─┬─>...└─> Phase 9 (lifecycle) <── (needs Phase 2)`,
which draws Phase 1 as a hard input to Phase 9 too.
Evidence: Phase 9's actual edits (eviction by `root_name`, the `end_date`
comment, the `kmeans` error message, a trailing-batch code comment) do not
functionally require Phase 1's new config file to exist first. Low impact,
but worth reconciling for consistency.

**N2 — Several phases list `config/validation_schema.yaml`/`utils/config_loader.py` in Write scope but not Context load.**
Plan location: Phase 4 ("Write scope: ... `utils/config_loader.py` /
`config/validation_schema.yaml`"), Phase 5, Phase 6, Phase 8 — none list
these files under "Context load" (Phase 3 and Phase 11 do this correctly).
Evidence: `config/validation_schema.yaml:3` sets `strict_unknown_keys: true`;
new keys must be registered through the dotted `required_keys`/`optional_keys`
mechanism implemented in `utils/config_loader.py:838-1065`
(`_build_allowed_key_tree`, `_validate_unknown_keys_recursive`,
`_validate_inline_dict_schema`, `_validate_config_schema`). A worker
following the plan's own Rule 2 ("do not read the whole repo... scope
reading to the phase's Context load") would not discover this mechanism for
these four phases, and in particular would not learn that a brand-new
top-level section (e.g. Phase 8's `reproducibility`) must also be added under
the schema's `sections:` map to be type-checked at all (`_validate_config_schema`
only validates sections it enumerates there; an unlisted top-level key is
silently unvalidated, not rejected).
Fix: add `utils/config_loader.py:838-1065` and one example section of
`config/validation_schema.yaml` to Context load for Phases 4, 5, 6, 8.

**N3 — Phase 8's and Phase 15b's `models/hyperparameter_tuning.py` citations are unscoped over a 2,718-line file.**
Plan location: Phase 8 "Context load" — *"`models/hyperparameter_tuning.py`
(Optuna study creation — where the sampler seed goes)"*; Phase 15 "Context
load" — *"`models/hyperparameter_tuning.py` (no `trial.report` / pruner
today)."*
Evidence: `optuna.create_study(` appears at four separate locations
(`:1910, :2184, :2234, :2559`), none cited. A worker must grep the whole
2,718-line file to find all four (all four need a seeded sampler for
Phase 8's "reproducible trial suggestions" acceptance criterion to hold).
Fix: cite the four line numbers explicitly.

**N4 — Phase 3 step 3 is likely already satisfied by step 2 (overstated scope, not a defect).**
Plan location: Phase 3, step 3: *"Apply the same stride to the frame-store
anchor emission in `_build_snapshot_chunks`."*
Evidence: `_build_snapshot_chunks` (`training/snapshot_dataset.py:2972-3400`)
constructs its builder via `_create_sample_builder(config)` and drives it
through `sample_builder.add_snapshot(snapshot)` at `:3335` — the identical
`StreamingSampleBuilder.add_snapshot` method already patched in step 2
(`:379-390`, anchor advance at `:384`). The single edit in step 2 likely
already covers both code paths. Not wrong, just double-counted; a worker
should verify with a quick test rather than assume a second edit site exists.

### NITPICK

**P1 — Some "run + analyze" acceptance criteria lack an explicit tolerance/threshold.**
Plan location: Phase 2 Acceptance — "throughput/VRAM are recorded and
compared to the benchmark expectation (about 241 samples/s at batch 80,
about 13 GiB peak)" has a baseline but no stated pass/fail band (±X%).
Phase 15 Acceptance — "XLA/compiled-LSTM benchmark reports a measured winner
(or confirms no gain)" has no minimum effect size to call a "winner" versus
run-to-run noise. Both are inherently exploratory phases, so some latitude is
appropriate, but an explicit tolerance (e.g. "±15% of 241 samples/s" / "≥10%
throughput or VRAM improvement to count as a winner") would make them
objectively checkable as the plan's own design principle requires.

**P2 — Phase 12's "evaluation/ (metric definitions)" context-load citation is a bare directory reference.**
Plan location: Phase 12, "Context load": *"`evaluation/` (metric
definitions)."* `evaluation/` contains `backtesting.py` (702 lines) and
`evaluator.py` (2,248 lines) with no pointer to either. Low risk because
Phase 12 is read-only/analysis work (no code changes to `evaluation/`), but
imprecise relative to every other phase's citation style.

---

## 3. Citation verification

| # | Plan location | Citation | Result | Actual line found |
|---|---|---|---|---|
| 1 | Phase 3 | `training/snapshot_dataset.py:379-388`, advance at `:384` | **OK** | `:379` `while self._latest_idx >= ...:` … `:384` `self._next_anchor_idx += 1` … `:388` buffer evict |
| 2 | Global invariants / Phase 3 | `training/snapshot_store.py:79-84` (config hash subset) | **OK** | `:79-84` = `compute_config_hash` exactly |
| 3 | Phase 9 | `training/snapshot_store.py:205-240`, prefix at `:212` | **OK** | `:205` `def maybe_evict_snapshots` … `:212` `prefix = f"{_sanitize_component(context.root_name)}_"` |
| 4 | Phase 5 | `preprocessing/train_test_split.py:16-45` | **OK** | `:16-45` = `compute_split_boundaries` exactly |
| 5 | Phase 5 / Phase 1 step 10 | `training/pipeline.py:251-284`, validation_split coupling `:262-266` | **OK** | `:251` `_resolve_snapshot_training_indices` … `:262-266` exact `raise ValueError` on mismatch |
| 6 | Phase 1 step 2 | balance guards `training/pipeline.py:343-344`, `:366-408` | **OK** | `:343-344` truncation to whole batches; `:366-408` both `min_samples_after_balance` and `min_fraction_after_balance` raises |
| 7 | Phase 7 | `main.py:209-221` (CLI only `--config`/`--schema`) | **OK** | exact `_parse_args`, only two `add_argument` calls |
| 8 | Phase 7, Phase 9 | `training/manifest_utils.py` (manifest fields) | **WRONG** | file is 39 lines of generic `load_manifest`/`write_manifest`/`sanitize_component`; real fields live in `training/snapshot_dataset.py` (chunks `:600-675`, `_build_snapshot_chunks` `:2972-3400`, `label_stats` `:1956-2020`) |
| 9 | Phase 9 | `config/training_config_default.yaml:75` (end_date "exclusive" comment) | **OK** | `:75` exactly: `end_date: "..." # Logical end ... (exclusive) ...` |
| 10 | Phase 1, Phase 9 | `training/sample_balancing.py:442-443` (kmeans unimplemented) | **OK** | `:442` `if selection_policy == _POLICY_KMEANS:` `:443` `raise ConfigError(... not implemented yet)` |
| 11 | Phase 13 | `models/hyperparameter_tuning.py:1709-1748` (trial objective) | **OK** | `:1709` `from training.pipeline import run_training_pipeline_result` … `:1748` the call site |
| 12 | Phase 13, assessment | `models/hyperparameter_tuning.py:1999-2008` (range vs categorical) | **OK** | exact `len(batch_size_space) >= 3` categorical branch |
| 13 | Phase 13 | `training/pipeline.py:576-592` (cross-window weighted average) | **OK** | `:576-592` = `_aggregate_hpo_window_metrics` exactly |
| 14 | Phase 13 | `main.py:418-463` (trial vs production mode) | **OK** | `:418` `if hpo_enabled:` … `:463` `_evaluate_snapshot_sequential(...)` |
| 15 | Phase 6 | `training/pipeline.py:1202-1243` ("callback creation, non-sequential fit") | **WRONG** | `:1202-1243` = entire `_resolve_sequential_windows` function (date-range chunking). Real non-sequential fit path: `:1250-1345` / `:1681-1730` |
| 16 | Phase 6 | `training/pipeline.py:1526-1542` (per-window resume save) | **OK** | exact `_save_sequential_resume_model`/`_save_sequential_resume_state` block |
| 17 | Phase 6 | `training/callbacks.py:13-95` | **OK** | exact `create_callbacks` (file is 98 lines total) |
| 18 | Phase 6 | `main.py:224-250` (post-train evaluation) | **OK** | exact `_evaluate_snapshot_sequential` |
| 19 | Phase 4 | `models/cnn_lstm_multiclass.py:399-469` (inputs/build/compile) | **OK** | `:400` `inputs = keras.Input(...)` … `:469` `model.compile(...)`, no `jit_compile` |
| 20 | Phase 4 | `training/snapshot_dataset.py:897-903` (aux broadcast) | **OK** | exact aux broadcast + concat block |
| 21 | Phase 4 | `training/snapshot_dataset.py:1648-1694` / `:2049-2094` | **OK** | batch concat/generator-with-weights; `_apply_normalization`/`_strip_mask_channels` exactly |
| 22 | Phase 11 | `evaluation/backtesting.py:253-269`, `:301-399` | **OK** | conflict-resolution block; `simulate_trades` with fixed `horizon_steps` exit and `*2` cost |
| 23 | Phase 11 | `training/snapshot_dataset.py:414-427`, `:3730-3750` | **OK** | max-excursion computation; `_compute_intensity_bins` exactly |
| 24 | Phase 8 | Optuna study creation in `models/hyperparameter_tuning.py` | **UNSCOPED (imprecise)** | `optuna.create_study(` at `:1910, :2184, :2234, :2559` — none of the 4 sites cited |
| 25 | Phase 14 | `data/greptime_client.py:554+` (multi_database) | **OK** | `:554` `multi_db_cfg = data_cfg["multi_database"]` inside `stream_order_book_chunks_by_time` |
| 26 | Phase 14 | `config/training_config_default.yaml:46-61` (multi_database block) | **OK** | `:46` `multi_database:` through the two connection entries |
| 27 | Phase 1 | `data/greptime_client.py:155` (end_date inclusive) | **OK** | exact `end_dt = end_dt.replace(hour=23, minute=59, second=59)` |
| 28 | Phase 1 | `training/snapshot_dataset.py:1526-1531` (trailing partial batch) | **OK** | exact `# Drop incomplete tail...` / `steps = ... // batch_size` block |
| 29 | Config facts | `config/e2e_fullscale_production_month1.yaml` (metric "loss", 2-GPU, batch 12, `gpu_memory_growth: false`) | **OK** | `:86` `metric: "loss"`, `:66`/`:92` `["gpu:0","gpu:1"]`, `:53` `batch_size: 12`, `:58` `gpu_memory_growth: false` |
| 30 | Config facts | `config/e2e_fullscale_production_2022to2024.yaml` (runtime commented out, undersampling, HPO ranges, dates) | **OK** | `:28-30` commented `runtime:`; `:51` `batch_size: [2, 4]`; `:53-69` undersampling block; `:8-9` `2022-01-01`/`2024-12-31` |

**Numeric claims independently recomputed (not just cited):**

| Claim | Method | Result |
|---|---|---|
| 241,434 params (full, with LT branch) | Built `build_cnn_lstm_model` from `e2e_fullscale_production_month1.yaml` merged config, `input_shape=(720,40,4,20)`, CPU-only (`CUDA_VISIBLE_DEVICES=""`) | **241,434 — exact match** |
| 239,098 params (no LT branch) | Same, `long_term_input_dim=0` | **239,098 — exact match** |
| 163,706 params, reduced `(256,20,4,10)+LT` | Same builder, `input_shape=(256,20,4,10)`, `long_term_input_dim=8` | **163,706 — exact match** |
| Undersampling 47,534→24 | Ran real `compute_undersample_counts(available_counts=[6,2719,7399,25653,11757], target_distribution=[0.0])` then `//12*12` | `[6,6,6,6,6]`=30 → **24 — exact match** |
| 67,907 samples / ~132 MiB | Read `snapshots/fullscale_trial_2022to2024_b3fcda89/manifest.json`, summed `chunks[].num_samples`; `du -sh` | **67,907 samples, 132M — exact match** |
| 603 unit tests currently pass | `CI=true python -m unittest discover -s tests -v` | **603 tests, OK, 13.2s** (clean baseline; supports every phase's "full suite passes" regression acceptance) |
| 241 samples/s, ~13 GiB @ batch 80 | Not independently reproducible without live GPU work (prohibited); internally consistent between assessment and review, no contrary evidence found | **Not re-run (out of scope for this pass)** |

---

## 4. Source coverage

**Appendix A (review corrections → phases):** All 4 imprecisions (1.1–1.4), all
13 gaps, and all 7 additional improvements are traceable to a specific phase
and step, verified row by row. No missing item, no fabricated mapping found.

**Assessment P0–P2 list → phases:** All 5 P0, 5 P1, and 3 P2 items map to
exactly one phase each (P0#1→3, P0#2→4, P0#3→5, P0#4→6, P0#5→11; P1#6→10,
P1#7→2+9(b), P1#8→7, P1#9→8, P1#10→9(a); P2#11/12/13→15 steps 1/2/3). No gaps.

**Assessment Stage 1–4 parameters → phases:** Stage 1→Phase 2, Stage 2→Phase 1,
Stage 3→Phase 13 step 1, Stage 4→Phase 13 steps 2–3. The "Full-Fidelity
Alternative" table (720×40×4×20 batch 6/8 path) is intentionally not given its
own phase since the plan adopts the reduced-shape path as primary — consistent
with the assessment's own recommendation, not a gap.

**Plan claims with no traceable source basis:** none found. Every quantitative
claim in the plan traces to either the assessment, the review, or a citation I
independently verified against the source tree.

**Items present in sources but weakly covered in the plan:**
- The assessment's "Immediate Configuration Blocker #4 — Sequential
  Evaluation" is functionally addressed (see M5 above) but never explicitly
  instantiated as a step, unlike blockers 1–3.
- Review gap 9 (duty-cycle weighting is a near no-op) is correctly recorded
  in Phase 1 step 8 and Phase 12 step 1, but no phase actually *removes* or
  *simplifies* the always-on duty-cycle weight — it remains a documented
  quirk rather than a resolved item. This matches the review's own framing
  (informational, not an actionable gap), so it is not a defect, just worth
  noting it will persist post-plan.

---

## 5. Per-phase context budget

Estimated by summing lines in each phase's cited Context load (unscoped
whole-file references counted at full file length) via `wc -l`. The 250k-token
budget is generous relative to this codebase's largest single files
(`training/snapshot_dataset.py` ≈ 3,958 lines, `models/hyperparameter_tuning.py`
≈ 2,718, `evaluation/evaluator.py` ≈ 2,248, `observability/server.py` ≈ 3,436,
`training/pipeline.py` ≈ 1,828), so most phases fit in raw token terms even
when a citation is imprecise. **TIGHT** below therefore mostly flags
*completeness/precision* risk (the stated Context load is wrong or
insufficient, forcing off-plan exploration), not literal token overflow,
except where noted.

| Phase | Context load size (approx.) | Verdict | Reason |
|---|---|---|---|
| 1 — Practical config | ~1,410 lines, all YAML + 2 tiny code refs | **FITS** | Config-only, no source edits |
| 2 — Preflight | ~4,100 lines (incl. unscoped `greptime_client.py` 986 + a 2,201-line example manifest.json + `diagnostics/` 582) | **FITS** | Large but mostly reference material for a run+report phase |
| 3 — Sample stride | ~2,340 lines (incl. unscoped `utils/config_loader.py` 1,118 + `validation_schema.yaml` 634, and an unscoped-but-likely-redundant 428-line function, see N4) | **FITS** | Comfortable; two whole-file refs are avoidable bulk, not a budget risk |
| 4 — Aux vector branch | ~3,120 lines as *stated* (dominated by an unscoped, unneeded 2,718-line `hyperparameter_tuning.py` ref); **missing** ~190 needed lines in `pipeline.py`/`distributed.py` (see M3) | **TIGHT** | Completeness gap, not size — stated context points at the wrong large file and omits small files it actually needs |
| 5 — Purged splits | ~1,060 lines | **FITS** | Well-scoped |
| 6 — Checkpoint/resume | ~270 lines cited, but the *correct* fit-path locations (~150 lines) are never cited (see M1) | **TIGHT** | Wrong citation forces the worker to rediscover the real edit site |
| 7 — Run planner | ~2,220 lines as stated, but includes two large **unscoped** files (`training/pipeline.py` 1,828 + `observability/server.py` 3,436 = 5,264 lines / likely 45–60k+ tokens) plus a mislabeled `manifest_utils.py` ref | **TIGHT→SPLIT risk** | The plan's own budget note ("main.py + manifest utils + a new module + tests. One session.") understates this; unlike Phases 10/15, Phase 7 has **no** explicit split hedge despite needing deep, non-adjacent understanding of `pipeline.py`'s split/balance/undersample internals *and* `observability/server.py`'s dashboard patterns. **Recommend adding an explicit hedge/split note**, mirroring Phase 10. |
| 8 — Reproducibility | ~3,840 lines as stated (dominated by unscoped `main.py` 493 + `hyperparameter_tuning.py` 2,718 + `mlflow_integration/` 627) | **TIGHT** | Four unlisted `optuna.create_study` call sites (N3) |
| 9 — Lifecycle | ~92 lines of precise citations, but the manifest-field citation is wrong (M2) | **FITS (tokens)** but citation-defective | Small in tokens; fix citation before use |
| 10 — Host pipeline | ~2,230 lines (incl. unscoped `training/pipeline.py` 1,828) | **TIGHT (plan already hedges)** | **Confirmed**: the plan's own "if it overruns, split the `tf.data` implementation and the benchmark into two sessions" is warranted and sufficient |
| 11 — Label/eval alignment | ~2,450 lines (dominated by unscoped `evaluation/evaluator.py`, 2,248 lines, cited with no line range) | **TIGHT** | Second-largest file in the repo cited with zero scoping; recommend citing the specific metric-computation functions relevant to task-aligned evaluation |
| 12 — Weighting/repr matrix | ~110 lines of precise citations + a bare `evaluation/` directory reference (P2) | **FITS** | Run+analyze phase; vague citation is low-risk since no `evaluation/` code changes |
| 13 — HPO + final fit | ~154 lines + Phase 1 config | **FITS** | Matches its own budget note; heavy lifting is the (non-agent) multi-day run |
| 14 — Scale-up | Frozen config + small ranges + optional whole `fine_tuning.py` (727) | **FITS** | Agent work is deriving 1 config + preflight + report |
| 15a — NVMe/XLA | ~21 lines (`cnn_lstm_multiclass.py:449-469`) + measurement | **FITS** | Confirmed: split is appropriate, 15a is lightweight |
| 15b — HPO pruning | Unscoped `hyperparameter_tuning.py` (2,718 lines) needed to locate the trial loop for `trial.report`/pruner wiring | **TIGHT** | Confirmed the split is warranted, but 15b itself still needs tighter line-range citations (same file as N3/Phase 8) |

**Refinement recommendation:** the plan explicitly hedges Phase 10 and splits
Phase 15; that judgment is confirmed correct. **Phase 7 deserves the same
treatment** — it currently reads as a "one session, main.py + a new module"
phase but actually requires deep, non-adjacent knowledge of two of the
repo's largest files. Consider splitting Phase 7 into "7a: planner core
(shape/bytes/date/window counts from the manifest + config, no dashboard
reuse)" and "7b: dashboard/run-state integration," or at minimum give it an
explicit budget-overrun hedge.

---

## 6. Open questions (left for a worker to guess at)

1. **Phase 4 + distributed training.** Is `aux_vector_branch: true` meant to
   be supported at all under `training.runtime.distributed.enabled: true`, or
   should it explicitly require `distributed.enabled: false`? The plan is
   silent (see M3); a worker must decide whether to extend
   `training/distributed.py`'s output-signature builders or add a guard.
2. **Phase 1 "Blocker 4."** Should the new profile *explicitly* set
   `training.sequential_training.enabled: false` (per repo no-hidden-defaults
   convention) or rely on the inherited default from
   `training_config_default.yaml:289`? The Acceptance criterion implies the
   former ("the key that fixes it") but Steps never say so (see M5).
3. **Phase 7 vs Phase 5 scheduling.** Given the Appendix C contradiction
   (M4), should Phase 7 actually start before Phase 5 lands (and simply
   report pre-purge counts until Phase 5 merges), or must it wait? The
   plan gives two different answers.
4. **Gate G0 re-approval cadence for multi-day runs.** Is a single upfront
   approval sufficient for a 2–4 day exclusive GPU occupation (Phase 13/14),
   or is periodic re-confirmation expected? Not specified (M6).
5. **Phase 9 sequencing.** Can Phase 9 run before Phase 1, given its own
   "Depends on" line only names Phase 2, while Appendix C's graph also draws
   Phase 1 as an input (N1)?
6. **Phase 13 search-space collision with Phase 4/3.** If Phase 4 (aux vector
   branch) has landed by the time Phase 13's HPO profile is written, does the
   CNN filter search space (`[16,24]`/`[32,48]`/`[64,96]`) still apply
   unchanged now that CNN input channels are 4 instead of 20? The plan does
   not say whether Phase 13's search space needs revisiting when Phase 3/4
   precede it (it only says they help "cut trial cost").
