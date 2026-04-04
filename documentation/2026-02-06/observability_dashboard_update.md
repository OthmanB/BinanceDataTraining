# Observability Dashboard and Config Editor Update (2026-02-06)

## Overview
This report summarizes the observability, configuration, and cleanup work completed
since the previous commit. The focus was on a fully local HTMX dashboard, robust
run-state reporting, and a richer training configuration editor.

## Observability Dashboard
- Added a stdlib `http.server`-based observability server with Basic Auth and a
  Prometheus `/metrics` endpoint.
- Dashboard panels now show run status, progress, duty-cycle metrics, expanded
  run stats, and live log tailing with auto-scroll.
- Run control is visible at all times; controls are disabled when run control is off.

## Training Configuration Editor
- Full-page configuration editor with file browser, load/save workflow, and
  inline hint bubbles.
- Simple/Extended modes:
  - **Simple** hides advanced sections (security, snapshot, mlflow, logging,
    diagnostics, advanced data/model settings) and uses
    `config/training_config_default.yaml` as the baseline for hidden fields.
  - **Extended** shows all required configuration fields.
- Form validation highlights missing required fields and blocks saves until
  requirements are met.
- Added select dropdowns for common enum fields (e.g., `run_mode.mode`).

## Runtime Device Selection
- Added `training.runtime.device` (`cpu`/`gpu`) and
  `training.runtime.gpu_visible_devices` to configuration and schema.
- Runtime device selection is applied in `main.py` before TensorFlow use.

## Run-State and Metrics
- Snapshot build, training, and evaluation stages update run-state JSON.
- Duty-cycle stats (min/median/p95) are computed during snapshot builds and
  surfaced in the UI and metrics.

## Cleanup
- Removed phase/skeleton terminology from comments, docstrings, and logs.
- Removed unused/orphan modules:
  - `training/metrics.py`
  - `evaluation/visualization.py`
  - `mlflow_integration/artifact_manager.py`
  - `data/external_sources.py`

## New/Updated Files
- Added: `config/observability.yaml`
- Added: `config/training_config_default.yaml`
- Updated: `observability/` module, `static/` assets, `AGENTS.md`, and core
  pipeline modules to reflect new observability and configuration behavior.
