# Observability & Diagnostics Documentation

This folder contains comprehensive guides for understanding run status, identifying termination causes, and diagnosing pruning events in the BinanceDataTraining pipeline.

## Documents

### 1. **Quick Reference** (`diagnostics_quick_reference.md`)
**Start here for live troubleshooting.**

- API endpoints cheat sheet (what to query and when)
- Dashboard panels overview
- SQLite queries for historical analysis
- Log file grep patterns
- **Decision tree** for diagnosing issues
- Real-world example: memory-induced pruning

**Best for:** "Run just failed, what do I check first?"

---

### 2. **Full Field Map** (`observability_diagnostics_field_map.md`)
**Complete reference with source code locations.**

Covers:
- **Run State SQLite schema**: All 40+ fields, their meanings, and source code locations
  - Status, stage, progress
  - Error tracking (last_error, last_traceback)
  - HPO pruning metrics
  - Memory watchdog fields
  - Training metrics
  
- **HPO Pruning details**: When/why trials are pruned (lines 1865, 1869, 1872 in hyperparameter_tuning.py)

- **Dashboard rendering**: Which fields appear in which HTML panels
  - `/ui/dashboard` — status badge, error box, pipeline stepper
  - `/ui/stats` — progress counters, HPO trial summary
  - `/ui/system` — memory/CPU/GPU, watchdog triggers, worker RSS
  - `/ui/hpo` — per-trial table with status, duration, phase_rss_bytes

- **Log structure**: What to search for in run.log

- **Cross-reference table**: Every concept → source file + line numbers

**Best for:** "I need to understand where field X comes from" or "I want to add a new diagnostic"

---

## Quick Diagnostic Workflow

### Is the run still running?
```bash
curl http://observability-server:8008/api/run-state | jq '.status, .heartbeat_time'
```
- `status == "running"` and `heartbeat_time` recent (< 60 seconds ago) → Still alive
- `status == "failed"` → See below

### Did Optuna prune trials?
```bash
curl http://observability-server:8008/api/hpo | jq '.hpo_trials_pruned'
```
- > 0 → Yes, check `/api/system` for memory issues

### Was it a memory issue?
```bash
curl http://observability-server:8008/api/system | jq '.hpo_rss_watchdog_trigger_count, .hpo_wave_worker_rss_max_bytes'
```
- `hpo_rss_watchdog_trigger_count > 0` → **Memory limit exceeded**
- `mem_used_ratio > 0.9` → System RAM pressure

### Get the error message
```bash
curl http://observability-server:8008/api/run-state | jq '.last_error, .last_traceback'
```

### Detailed trial analysis
```bash
curl http://observability-server:8008/api/hpo | jq '.trials[] | select(.status=="pruned")'
```
- Look at `phase_rss_bytes` (peak memory during trial)
- Compare to limit in training config

---

## Source Code Organization

**Observability/Run State:**
- `observability/run_state.py` — SQLite persistence, all state mutations
- `observability/server.py` — REST API endpoints + HTML rendering
- `observability/training_progress.py` — Keras callback for epoch/batch tracking

**HPO & Pruning:**
- `models/hyperparameter_tuning.py` — Lines 1865, 1869, 1872 (where optuna.TrialPruned is raised)
- `models/hyperparameter_tuning.py:34` — Trial state counting

**Configuration:**
- `config/observability.yaml` — Dashboard settings (non-secrets)
- Environment variables: `OBSERVABILITY_USER`, `OBSERVABILITY_PASSWORD`, `RUN_STATE_PATH`, `RUN_LOG_PATH`

---

## Key Fields at a Glance

| Field | Type | Meaning | Where Set |
|-------|------|---------|-----------|
| `status` | str | `"running"`, `"completed"`, `"failed"`, `"idle"` | run_state.py:407, 418 |
| `stage` | str | Pipeline phase (snapshot_build, trial, training, evaluation) | run_state.py:247 |
| `progress` | float | 0.0-1.0 overall completion | run_state.py:427-452 |
| `last_error` | str | Human-readable error message | run_state.py:405 |
| `last_traceback` | str | Full traceback if error occurred | run_state.py:405 |
| `hpo_trials_pruned` | int | Count of Optuna-pruned trials | run_state.py:266-282 |
| `hpo_rss_watchdog_trigger_count` | int | Times memory limit was hit (→ pruning) | run_state.py:312-319 |
| `hpo_wave_worker_rss_current_bytes` | float | Hottest worker RSS now | run_state.py:284 |
| `hpo_wave_worker_rss_max_bytes` | float | Peak worker RSS ever | run_state.py:284-300 |
| `hpo_wave_worker_rss_top` | list | Top 3 workers by RSS | run_state.py:302-309 |
| `heartbeat_time` | float | Epoch seconds of last activity | run_state.py:454-458 |
| `stage_timestamps` | dict | When each stage was entered | run_state.py:233-239 |
| `training_epoch_metrics` | list | Per-epoch loss, val_loss, etc. | run_state.py:381-389 |
| `hpo_trial_results` | list | Full trial data (status, params, RSS) | run_state.py:399-403 |

---

## Common Diagnostics Scenarios

### Scenario 1: "Run stopped with memory errors and pruned most trials"
1. Check `/api/system` → `hpo_rss_watchdog_trigger_count` (should be > 0)
2. Check `/api/system` → `hpo_wave_worker_rss_max_bytes` (peak memory)
3. Check `/api/hpo` → `trials[]` → `phase_rss_bytes` (memory per trial)
4. Check `run.log` → `grep "Watchdog triggered"`
5. **Action:** Reduce per-worker memory limit or reduce worker count in config

### Scenario 2: "Run completed but loss values look wrong"
1. Check `/api/run-state` → `status == "completed"` (✓)
2. Check `/api/run-state` → `progress == 1.0` (✓)
3. Check `/api/run-state` → `training_epoch_metrics` (last epoch's loss values)
4. Check `run.log` → Last 50 lines for training summary

### Scenario 3: "Run hung after entering trial stage"
1. Check `/api/run-state` → `heartbeat_time` (is it stale?)
2. Check `/api/run-state` → `stage_timestamps` (how long in "trial" stage?)
3. Check `/api/system` → `mem_used_ratio` (running out of memory?)
4. Check `/api/logs` → Search for timeout/socket/connection errors
5. Check `run.log` → Last 100 lines for exception

### Scenario 4: "Need to understand why Trial #42 was pruned"
1. Check `/api/hpo` → `trials[42]` → `status` (should be "pruned")
2. Check `phase_rss_bytes` (was it memory-bound?)
3. Check `duration` and `params` (unusual hyperparameters?)
4. Check `run.log` → Search for "Trial 42" + "Pruned"

---

## API Endpoints Reference

All endpoints are JSON (use `jq` for pretty-printing):

```bash
# Get current run state (40+ fields)
curl http://localhost:8008/api/run-state | jq

# Get last 200 lines of run.log
curl http://localhost:8008/api/logs | jq '.lines'

# Get system metrics (CPU, RAM, GPU, watchdog)
curl http://localhost:8008/api/system | jq

# Get HPO trial data
curl http://localhost:8008/api/hpo | jq '.trials'

# Get history of all past runs
curl http://localhost:8008/api/history | jq '.runs'
```

---

## Environment Setup for Observability

**Required:**
```bash
export OBSERVABILITY_USER="admin"
export OBSERVABILITY_PASSWORD="password"
export RUN_STATE_PATH="sqlite:///path/to/db.sqlite"
export RUN_LOG_PATH="/path/to/run.log"
```

**Optional (defaults shown):**
```bash
export OBSERVABILITY_HOST="127.0.0.1"          # or "0.0.0.0" to expose
export OBSERVABILITY_PORT="8008"
export OBSERVABILITY_TAIL_MAX_LINES="200"      # Increase for detailed diagnostics
export RUN_STATE_CACHE_TTL_SECONDS="2.0"       # Cache freshness
```

---

## Updating This Documentation

When adding new observability fields or changing dashboard endpoints:

1. Update `observability_diagnostics_field_map.md` with:
   - New field name + type + meaning
   - Source file and line number
   - Which dashboard panel displays it
   
2. Update `diagnostics_quick_reference.md` with:
   - New endpoint/field if user-facing
   - Update decision tree if logic changes
   - Add new scenario if common use case

3. Add a note in this README.md pointing to the changes

---

## Contact & Troubleshooting

If the observability server itself fails:
- Check logs: `OBSERVABILITY_USER`, `OBSERVABILITY_PASSWORD` must be set
- Check SQLite path: `RUN_STATE_PATH` must be `sqlite:///...` format
- Check file permissions: `RUN_LOG_PATH` must be readable
- Check port: `OBSERVABILITY_PORT` not in use (try different port)

For questions about specific fields, refer to the **Full Field Map** and cross-reference with source code.
