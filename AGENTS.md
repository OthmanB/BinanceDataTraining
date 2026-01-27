# AGENTS

This file guides agentic coding tools working in this repo.
Follow these rules before making changes.

## Quick Context
- Repo: BinanceDataTraining
- Primary language: Python
- Entry point: `main.py` (config-driven pipeline)
- Tests: `unittest` plus some `pytest`/`hypothesis`

## Setup
- Use `bash setup_venv.sh` to create `.venv` and install deps.
- Required Python: >=3.9 and <3.12 (per `setup_venv.sh`).
- Dependency install: `pip install --only-binary=pyarrow -r requirements.txt`.
- Sanity check: `python -m pip check`.

## Build
- No explicit build system detected (no Makefile, pyproject, or tox).
- If you need to run the pipeline, use `python main.py --config <path>`.

## Lint / Format
- No lint/format tooling configured in this repo.
- If you add tooling, document it here and keep PEP 8 style.

## Tests (Full Suite)
- Default: `python -m unittest discover -s tests -v`.
- CI-friendly run: `CI=true python -m unittest discover -s tests -v`.
- Hypothesis-based tests are skipped if hypothesis is not installed.

## Tests (Single Test)
- Single unittest module:
  `python -m unittest tests.test_config_loader`.
- Single unittest class:
  `python -m unittest tests.test_config_loader.TestConfigLoader`.
- Single unittest method:
  `python -m unittest tests.test_config_loader.TestConfigLoader.test_missing_required_section`.
- Pytest (optional; install pytest first):
  `python -m pytest tests/test_feature_engineering.py`.
- Pytest single test:
  `python -m pytest tests/test_feature_engineering.py -k test_name`.

## Cursor / Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
- Follow `.github/global_rules.md` (summarized below).

## Repo Rules (from `.github/global_rules.md`)
- Work methodically; avoid speculative edits.
- Build a quick codemap before editing new areas.
- Prefer YAML-first configuration; avoid implicit defaults.
- Validate configuration immediately on load; fail fast.
- Secrets/credentials must come from environment variables.
- Use the `logging` module; avoid `print`.
- Logging should be configurable, contextual, and colorized via `termcolor`.
- Keep commands safe and short; use timeouts when needed.
- Put new instructions/docs under `documentation/` or `instructions/` and date them.
- Do not edit files without approval when operating in agent mode.

## Code Style: Structure
- Keep a module docstring at the top of each file.
- Import order: stdlib, third-party, local (blank lines between groups).
- Use `__all__` to define module exports.
- Avoid side effects at import time.
- Guard entry points with `if __name__ == "__main__":`.

## Code Style: Types
- Add type hints for public functions and methods.
- Config objects are usually `Dict[str, Any]` or `Mapping[str, Any]`.
- Use `Optional[T]` for nullable values and document defaults explicitly.
- Add `from __future__ import annotations` if using forward references.

## Code Style: Naming
- Functions and variables: `snake_case`.
- Classes: `CamelCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: leading underscore (e.g., `_build_config`).
- Test classes: `Test*`; test methods start with `test_`.

## Code Style: Error Handling
- Use `ConfigError` for configuration problems.
- Use `ValueError` for invalid inputs or parameters.
- Validate config early; raise with clear, actionable messages.
- When catching broad exceptions, add `# noqa: BLE001` and log context.
- Do not silently swallow errors; log and re-raise when appropriate.

## Logging
- Use `logger = logging.getLogger(__name__)` per module.
- Prefer `logger.info` / `warning` / `error` over print.
- Include key context (config sections, asset IDs, sizes).
- Keep log messages concise and structured.

## Configuration
- Behavior is driven by YAML configuration.
- No hidden defaults; require explicit values.
- Environment placeholders `${VAR}` must resolve or raise `ConfigError`.
- Update config schema validation when adding new keys.
- Avoid leaking secrets to logs.

## Data / ML Conventions
- Validate array shapes and sizes early.
- Use numpy helpers for numeric comparisons.
- Allow small tolerances for floating point equality.
- Avoid non-deterministic behavior in tests.
- Keep feature engineering pure and config-driven.

## Tests Style
- Use `unittest.TestCase` for most unit tests.
- Pytest is used in some tests; keep assertions idiomatic.
- Hypothesis tests should be guarded by availability and CI scaling.
- Use `pytest.approx` or `np.testing` for numeric checks.
- Skip heavy property tests when `CI=true`.
- Keep tests deterministic and fast.

## External I/O
- Handle network/API failures with logging and clear error messages.
- Validate timeouts and URLs from config before use.
- Avoid real network calls in unit tests; use mocks.

## Repo Hygiene
- Keep changes localized and consistent with existing patterns.
- Preserve behavior unless requirements explicitly change.
- Add or update tests when changing logic.
