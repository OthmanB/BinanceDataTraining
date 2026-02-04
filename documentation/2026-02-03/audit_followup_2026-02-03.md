# Audit Follow-up (2026-02-03)

This note records the follow-up work completed after the 2026-02-02 audit and
roadmap review.

## Summary
- Completed TD-009, TD-010, TD-012, TD-019 wiring and validation in the snapshot pipeline.
- Updated multi-asset alignment configuration to match the 2026-02-02 plan.
- Removed dead/unreachable legacy paths in the entrypoint and training pipeline.

## Notable Changes
### Alignment configuration (multi-asset)
- `data.asset_pairs.alignment` now uses:
  - `method: interpolate | bucket`
  - `missing_policy: forward_fill | skip | error`
  - `max_gap_seconds`
  - `bucket_tolerance_seconds`
  - `include_mask_channel`
- Prior keys (`missing_policy_small`, `missing_policy_large`, `large_gap_seconds`) are removed.
- Behavior: gaps beyond `max_gap_seconds` now follow `missing_policy`; zero-padding is no longer used.

### Long-term features (dual-input)
- Snapshot builds now persist target-asset series for long-term features.
- Long-term features are cached per snapshot dataset.
- Existing snapshots created before this change must be rebuilt to enable long-term inputs.

### Evaluation/backtesting
- Snapshot evaluation now supports backtesting with required `evaluation.backtesting.horizon_steps`.
- Calibrated probabilities are used for backtesting when post-hoc calibration is enabled.

### Fine-tuning
- Multi-input shape validation and compile-after-freeze are enforced for fine-tuning.

### Fail-fast configuration
- Runtime code now reads required keys directly (no implicit defaults).
- Schema updated to require model compilation metrics, long-term config, and MLflow artifact logging keys.

## Removed Dead Paths
- Unreachable HPO branch after the snapshot-only check in `main.py`.
- Legacy in-memory training pipeline block after the snapshot-only guard.

## Tests
Commands executed in `.venv`:
- `python -m unittest discover -s tests -v`
- `python -m pytest tests -q`

Results:
- Unittest: `Ran 312 tests` — OK
- Pytest: `456 passed` (warnings only from TensorFlow/MLflow)

## Migration Notes
- Update any custom configs to the new alignment keys.
- Ensure `model.output.activation`, `model.compilation.metrics`, and all required `mlflow` keys are present.
- Rebuild snapshot datasets if enabling long-term inputs on existing snapshots.
