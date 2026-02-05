# End-to-End Trial Matrix Plan (2026-02-03)

This document records the planned matrix of ~10 end-to-end trial runs for the
snapshot pipeline. The intent is to validate integration paths without running
full production-scale training.

## Environment and Execution
- Always run with the virtual environment:
  - `source .venv/bin/activate`
- Environment variables are sourced from `~/.bashrc`:
  - `MLFLOW_TRACKING_URI`, `DATABASE_URI`, `DATABASE_URI_HIST`, `DATABASE_URI_LIVE`

## Important constraint
- `run_mode.mode="trial"` skips training and evaluation in `main.py`.
- For true end-to-end checks, keep `run_mode.mode="production"` and use a
  reduced data window + short epochs to simulate trial runs.

## Base trial profile (applies to all runs unless overridden)
- `run_mode.mode: production`
- `training.epochs: 1`
- `training.batch_size: 8` (or 16)
- `training.debug_max_samples`: set >= total samples to satisfy production guard
- `mlflow.model_registry.register_model: false`
- `mlflow.artifact_logging.trained_model: false`
- `snapshot.name`: unique per run to avoid snapshot collisions
- Each run is defined as an override file with `base_config: training_config.yaml`
- `data.validation.fail_on_invalid: false`
- `data.validation.max_gap_seconds: 600`

## Matrix of trial runs
Each run uses a short time range with known data availability (initially a 1-week window).

1) Baseline smoke
   - Config: `config/e2e_trial_01_baseline.yaml`
   - No correlated assets
   - `model.long_term.enabled: false`
   - `evaluation.post_hoc_calibration.enabled: false`
   - `evaluation.temporal_degradation.enabled: false`
   - `evaluation.backtesting.enabled: false`

2) Full evaluation features
   - Config: `config/e2e_trial_02_full_eval.yaml`
   - Same as baseline, but:
     - `evaluation.post_hoc_calibration.enabled: true`
     - `evaluation.temporal_degradation.enabled: true`
     - `evaluation.backtesting.enabled: true`

3) Long-term dual input (short windows)
   - Config: `config/e2e_trial_03_long_term.yaml`
   - `model.long_term.enabled: true`
   - `model.long_term.windows_days: [1, 3, 7]`
   - `model.long_term.summary_method: mean`
   - Keep evaluation minimal (calibration off) to isolate long-term input path

4) Correlated asset + forward fill
   - Config: `config/e2e_trial_04_corr_forward_fill.yaml`
   - `data.asset_pairs.correlated_assets: [<asset>]`
   - `data.asset_pairs.alignment.missing_policy: forward_fill`
   - `data.asset_pairs.alignment.include_mask_channel: true`

5) Correlated asset + skip
   - Config: `config/e2e_trial_05_corr_skip.yaml`
   - Same as #4, but:
     - `missing_policy: skip`
     - `include_mask_channel: false`

6) Alignment method: bucket
   - Config: `config/e2e_trial_06_alignment_bucket.yaml`
   - `data.asset_pairs.alignment.method: bucket`
   - `bucket_tolerance_seconds: 2`
   - `missing_policy: forward_fill` (or `skip` if preferred)

7) Order book top_of_book
   - Config: `config/e2e_trial_07_top_of_book.yaml`
   - `data.order_book.representation: top_of_book`
   - `preprocessing.feature_engineering.enabled: false`

8) Multi-database time_split
   - Config: `config/e2e_trial_08_multi_db.yaml`
   - `data.multi_database.enabled: true`
   - `data.time_range` spans the historical -> recent boundary

9) Temporal features off
   - Config: `config/e2e_trial_09_temporal_off.yaml`
   - `data.temporal_features.local: []`
   - `data.temporal_features.global: []`
   - `model.input_representation.temporal_features.integration_mode: none`

10) Full eval + long-term
    - Config: `config/e2e_trial_10_full_eval_long_term.yaml`
    - Combine #2 + #3:
      - long-term inputs enabled
      - calibration + temporal degradation + backtesting enabled

## Fine-tuning passes
These are executed after the baseline run to obtain a run_id.

- Pass A (base run_id source)
  - Config: `config/e2e_trial_11_finetune_base.yaml`

- Pass B (fine-tune from run_id)
  - Config: `config/e2e_trial_12_finetune_from_runid.yaml`
  - Replace `base_model_run_id` with the run_id from Pass A.

## Resolved execution decisions
- Config management: `main.py` now supports `--config`.
- Data window: start with a 1-week window; adjust to a few days if runs are too long,
  or extend up to a month if too short.
- Correlated asset: `ADAUSDT`.

## Execution command
Example:

```bash
source .venv/bin/activate
python main.py --config config/e2e_trial_01_baseline.yaml
```
