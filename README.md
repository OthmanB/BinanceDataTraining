# Binance Data Training

Binance Data Training is a config-driven pipeline for training market microstructure models with snapshot datasets, Optuna-based HPO, MLflow tracking, and an observability dashboard.

## Highlights
- Snapshot-first training pipeline with sequential windows and resume support
- Parallel Optuna HPO with regime safeguards and GPU/CPU resource controls
- Evaluations: calibration analysis, post-hoc calibration, temporal degradation, backtesting
- Diagnostics and visualization logging (snapshot-native)
- Observability UI with system telemetry and run-state metrics

## Quick Start
1. Create the virtual environment and install dependencies:
   ```bash
   bash setup_venv.sh
   ```
2. Run a training job:
   ```bash
   .venv/bin/python3 main.py --config config/training_config.yaml
   ```
3. Run the observability server:
   ```bash
   .venv/bin/python3 -m observability.server --config config/observability.yaml
   ```

## Configuration
- Base configuration: `config/training_config.yaml`
- Example large-scale run: `config/e2e_fullscale_production_month1.yaml`
- Observability: `config/observability.yaml`

The pipeline validates configuration strictly against `config/validation_schema.yaml` and fails fast on unsupported keys or missing values.

## Tests
- Full suite:
  ```bash
  .venv/bin/python3 -m unittest discover -s tests -v
  ```
- CI-friendly:
  ```bash
  CI=true .venv/bin/python3 -m unittest discover -s tests -v
  ```

## Documentation

- [Architecture Overview](documentation/architecture.md) — pipeline flow, data flow, module dependencies, model architecture
- [Configuration Key Interactions](documentation/config_interactions.md) — which config keys depend on each other
- [Troubleshooting Guide](documentation/troubleshooting.md) — common errors and fixes
- [Codebase Audit](documentation/2026-02-08/codebase_full_audit.md) — full audit report

## Notes
- Python 3.9 to <3.13 is required (see `setup_venv.sh`).
- Snapshot datasets are the only supported training mode.

## License
GNU General Public License v3.0. See `LICENSE`.
