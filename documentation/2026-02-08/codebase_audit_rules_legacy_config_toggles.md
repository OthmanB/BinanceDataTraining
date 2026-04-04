# Codebase Audit - Rules, Legacy, and Config Toggle Safety

Date: 2026-02-08

Scope:
- Repository-wide static audit focused on:
  1. AGENTS/global rules compliance
  2. Outdated/legacy/dead elements
  3. On/off safety for configuration sections
- Evidence is linked with file/line references.

Method summary:
- Searched and inspected code paths for snapshot/training/evaluation/HPO/diagnostics/runtime/config validation.
- Prioritized findings by severity.

## 1) Rule Compliance Findings

### High
- **Diagnostics flags are only partially honored in active snapshot flow**
  - Snapshot mode executes `run_snapshot_diagnostics` (`main.py:349`), but this function uses only a subset of `diagnostics.*` keys (`diagnostics/snapshot_diagnostics.py:54`, `diagnostics/snapshot_diagnostics.py:75`, `diagnostics/snapshot_diagnostics.py:114`, `diagnostics/snapshot_diagnostics.py:127`).
  - Legacy richer diagnostics exists in `diagnostics/data_diagnostics.py:22` but is not called anywhere (`run_data_diagnostics` has no call sites).
  - Risk: configuration appears fully supported but many toggles are effectively ignored.

- **Silent skip in snapshot diagnostics artifact logging**
  - `diagnostics/snapshot_diagnostics.py:189`-`diagnostics/snapshot_diagnostics.py:192` returns silently when MLflow import fails.
  - Rule impact: violates no-silent-skip expectation for operational behavior.

- **Schema says `sample_weighting.apply_to` is required, but snapshot path does not validate it**
  - Required by schema (`config/validation_schema.yaml:275`).
  - Snapshot training validates method/half-life only (`training/snapshot_dataset.py:718`-`training/snapshot_dataset.py:725`), no `apply_to` enforcement.
  - Risk: hidden config inconsistency.

### Medium
- **Broad exception handlers with `pass`/quiet fallback in runtime-critical paths**
  - Examples:
    - `training/pipeline.py:562`
    - `training/pipeline.py:918`
    - `evaluation/evaluator.py:888`
    - `evaluation/evaluator.py:1363`
    - `training/snapshot_dataset.py:1706`
    - `training/snapshot_dataset.py:1719`
  - Risk: failures get masked with insufficient operator signal.

- **Unknown config keys are not rejected**
  - Validation checks required/optional keys by type, but no strict extra-key rejection:
    - `utils/config_loader.py:111`-`utils/config_loader.py:149`
  - Risk: typos silently no-op.

- **`evaluation.metrics` appears schema-required but not used for runtime behavior**
  - Required in schema (`config/validation_schema.yaml:425`), no meaningful runtime selector use detected.
  - Risk: stale configuration surface.

### Low
- **`print` instead of logging in utility script**
  - `tools/mlflow_full_reset.py:19`, `tools/mlflow_full_reset.py:35`, `tools/mlflow_full_reset.py:53`.

- **Missing module docstrings (non-test modules)**
  - `diagnostics/data_diagnostics.py:1`
  - `preprocessing/snapshot_sequence_builder.py:1`
  - `preprocessing/time_utils.py:1`
  - `tools/mlflow_full_reset.py:1`
  - `data/__init__.py:1`
  - `utils/__init__.py:1`

- **Import-time side-effect (`matplotlib.use("Agg")`) in importable modules**
  - `diagnostics/data_diagnostics.py:11`
  - `diagnostics/snapshot_diagnostics.py:13`

## 2) Outdated / Legacy / Dead Elements

### High
- **Unreachable legacy in-memory training block after hard error**
  - Snapshot-only gate raises and then legacy code remains in same function:
    - raise path: `training/pipeline.py:1286`-`training/pipeline.py:1289`
    - dead/unreachable remainder starts after `training/pipeline.py:1289`
  - Recommendation: remove unreachable block or move to explicitly deprecated module.

### Medium
- **Unused legacy imports in `main.py`**
  - Imported but unused:
    - `main.py:26` (`load_order_book_data`)
    - `main.py:27` (`attach_temporal_features`)
    - `main.py:28` (`run_preprocessing_pipeline`)
    - `main.py:29` (`chronological_split_indices`)
    - `main.py:34` (`evaluate_model` from `evaluation`)

- **Legacy diagnostics path is currently dormant**
  - `run_data_diagnostics` is defined (`diagnostics/data_diagnostics.py:22`) but not invoked by active flow.
  - Recommendation: either integrate into snapshot diagnostics pipeline or deprecate/remove and shrink schema.

### Low
- **Potential stale constant**
  - `training/fine_tuning.py:27` (`DEFAULT_ALLOWED_OVERRIDES`) appears not consumed in active paths.

## 3) Config Toggle Safety Audit (Requested Elements)

### Snapshot
- **Status:** Safe and explicit.
- **Evidence:** non-snapshot path fails with clear error (`main.py:340`-`main.py:345`, `training/pipeline.py:1286`-`training/pipeline.py:1289`).

### Diagnostics / Visualization
- **Status:** Partially safe, partially misleading.
- **Behavior:** no crash for toggles, but many diagnostics keys are currently not used in snapshot diagnostics implementation.
- **Evidence:** snapshot diagnostics consumes subset only (`diagnostics/snapshot_diagnostics.py:54`, `:75`, `:114`, `:127`), while full diagnostics config surface remains in schema/config (`config/validation_schema.yaml:514` onward).
- **Risk:** silent functional under-delivery of configured diagnostics.

### Temporal degradation / Post-hoc calibration / Calibration analysis
- **Status:** Good fail-fast.
- **Evidence:** evaluator validates key bounds and raises clear errors:
  - calibration bins: `evaluation/evaluator.py:895`
  - temporal windows/overlap: `evaluation/evaluator.py:986`-`evaluation/evaluator.py:989`
  - post-hoc method/min_samples/bounds: `evaluation/evaluator.py:1119`-`evaluation/evaluator.py:1134`

### Hyperparameter optimization (Optuna)
- **Status:** Mostly safe with explicit validation; one silent-skip caveat.
- **Evidence:** parallel settings validated (`models/hyperparameter_tuning.py` parallel resolver paths), but unsupported framework/dependency may return without hard fail in some paths.

### Sequential training
- **Status:** Good fail-fast.
- **Evidence:** bounded window validation with explicit `ConfigError` for invalid `window_days`/dates (`training/pipeline.py:269`-`training/pipeline.py:299`).

### Runtime GPU settings
- **Status:** Good fail-fast.
- **Evidence:** `_apply_runtime_device` and `_validate_runtime_device_availability` enforce validity early (`main.py:282`-`main.py:288`).

### Fine-tuning
- **Status:** Good fail-fast when enabled.
- **Evidence:** required run/model registry inputs validated with explicit errors in pipeline fine-tuning branch (`training/pipeline.py:944`-`training/pipeline.py:971`).

### Sample weighting
- **Status:** Mostly safe; one consistency gap.
- **Evidence:** validates method and half-life (`training/snapshot_dataset.py:718`-`training/snapshot_dataset.py:725`).
- **Gap:** `apply_to` required by schema but not enforced in snapshot path.

### Long-term features
- **Status:** Good fail-fast.
- **Evidence:** checks for shape/input mismatch and model input compatibility:
  - training: `training/pipeline.py:825`-`training/pipeline.py:841`, `training/pipeline.py:998`-`training/pipeline.py:1017`
  - evaluation: `evaluation/evaluator.py:808`-`evaluation/evaluator.py:836`

## 4) Recommendations (Priority)

1. **Remove dead in-memory pipeline block in `training/pipeline.py`** and prune stale imports/usages in `main.py`.
2. **Align diagnostics surface with implementation**:
   - Either integrate remaining diagnostics checks/plots into snapshot diagnostics,
   - or trim schema/config keys to the currently supported subset with explicit deprecation notes.
3. **Eliminate silent skips**:
   - Add warnings where broad exceptions currently `pass`/return quietly,
   - especially in diagnostics and runtime observability hooks.
4. **Enforce `sample_weighting.apply_to` in snapshot path** (or remove from schema if intentionally unsupported).
5. **Add strict config mode** to reject unknown keys and prevent typo-driven no-op behavior.
6. **Rule hygiene cleanup**:
   - add missing module docstrings,
   - replace `print` in utilities with logging (or document explicit script exception),
   - avoid import-time backend side effects where practical.

---

Audit outcome summary:
- Core snapshot/HPO/evaluation toggles are mostly fail-fast and operationally safe.
- Main residual risk is **configuration-to-runtime mismatch** (diagnostics surface larger than actual snapshot implementation), plus a few **silent-skip** and **legacy-dead-code** remnants.
