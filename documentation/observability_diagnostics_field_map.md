# BinanceDataTraining: Run Status & Diagnostics Field Map

## Quick Reference: Where to Look for Run Termination/Pruning Evidence

### Dashboard JSON API Endpoints
| Endpoint | Returns | Key Fields for Diagnostics |
|----------|---------|---------------------------|
| `/api/run-state` | Complete run state | `status`, `stage`, `progress`, `last_error`, `last_traceback`, `hpo_trials_pruned`, `hpo_rss_watchdog_trigger_count` |
| `/api/logs` | Last N lines from run.log | Application-level errors, exceptions, pruning reasons |
| `/api/system` | System metrics | `mem_used_ratio`, `run_process_rss_bytes`, GPU memory/util |
| `/api/hpo` | HPO trial data | `hpo_trials_total`, `hpo_trials_completed`, `hpo_trials_pruned`, `hpo_trials_failed`, `trials[]` (with status/value) |
| `/api/history` | Historical run records | `status`, `final_loss`, `final_val_loss` from completed runs |

---

## Run State SQLite Schema (`RUN_STATE_PATH`)

**Source:** `observability/run_state.py:140-170`

### Table: `run_state` (upsert on id=1)
Stores current run snapshot as JSON payload with these fields:

#### Status & Stage Fields
- **`status`** (str): `"idle"`, `"running"`, `"completed"`, `"failed"`
  - Source: `RunStateWriter.set_error()` line 407, `RunStateWriter.complete()` line 418
- **`stage`** (str): `"idle"`, `"initializing"`, `"diagnostics"`, `"snapshot_build"`, `"trial"`, `"training"`, `"evaluation"`
  - Tracks pipeline phase; pruning can occur during `"trial"` stage
  - Source: `RunStateWriter.set_stage()` line 247

#### Progress Fields
- **`progress`** (float, 0.0-1.0): Overall completion ratio
- **`eta_seconds`** (Optional[float]): Estimated time to completion
  - Source: `RunStateWriter._update_eta_locked()` line 443
- **`snapshot_chunks_processed`**, **`snapshot_chunks_total`**: Snapshot build progress
- **`training_epochs_done`**, **`training_epochs_total`**: Training epoch progress
- **`training_batches_done`**, **`training_batches_total`**: Batch-level progress
- **`eval_batches_done`**, **`eval_batches_total`**: Evaluation progress

#### Error Tracking
- **`last_error`** (Optional[str]): Human-readable error message
  - Source: `RunStateWriter.set_error(message, ...)` line 405
- **`last_traceback`** (Optional[str]): Full traceback text
  - Source: `RunStateWriter.set_error(..., traceback_text)` line 405

#### HPO & Pruning Fields
- **`hpo_trials_total`**, **`hpo_trials_completed`**, **`hpo_trials_pruned`**, **`hpo_trials_failed`**
  - Source: `RunStateWriter.update_hpo_progress(completed=..., pruned=..., failed=...)` line 266
  - Pruning reason: Check `/api/hpo` → `trials[].status == "pruned"` for trial-level context
- **`hpo_trial_results`** (List[Dict]): Full trial data including `status`, `value`, `params`, `duration`, `phase_rss_bytes`
  - Source: `RunStateWriter.update_hpo_trial_results(trials)` line 399
  - **Key field for pruning**: `trials[].status` can be `"pruned"`, `"completed"`, `"failed"`, `"running"`

#### Resource Watchdog Fields (HPO Memory Safety)
- **`hpo_wave_worker_count`**: Number of active worker processes
- **`hpo_wave_worker_rss_current_bytes`**: Hottest worker RSS right now
- **`hpo_wave_worker_rss_max_bytes`**: Historical peak RSS across all workers
- **`hpo_wave_worker_rss_top`** (List[Dict]): Top 3 workers with `pid`, `rss_bytes`
  - Source: `RunStateWriter.update_hpo_wave_memory(rss_by_pid)` line 284
- **`hpo_rss_watchdog_trigger_count`**: Number of times memory limit was hit (→ pruning trigger)
- **`hpo_rss_watchdog_last_trigger_time`**, **`_pid`**, **`_rss_bytes`**, **`_limit_bytes`**
  - Source: `RunStateWriter.mark_hpo_rss_watchdog_trigger(pid, rss_bytes, limit_bytes)` line 312
  - **← Direct evidence of pruning cause: resource exhaustion**

#### Timing & Heartbeat
- **`start_time`**, **`updated_time`**, **`heartbeat_time`** (floats, epoch seconds)
  - `heartbeat_time` tracks last activity; staleness indicates hang/crash
- **`stage_timestamps`** (Dict[str, float]): When each stage was entered
  - Source: `RunStateWriter._record_stage_if_new_locked()` line 233
  - Useful for determining where run stalled

#### Training Metrics
- **`training_epoch_metrics`** (List[Dict]): Per-epoch data `{epoch, loss, val_loss, ...}`
  - Source: `RunStateWriter.update_epoch_metrics(epoch, metrics)` line 381
  - Last entry's loss values useful for understanding training state at termination

#### Duty Cycle (Data Quality Metric)
- **`duty_cycle_min`**, **`duty_cycle_median`**, **`duty_cycle_p95`**
- **`duty_cycle_history`** (List[Dict]): Historical `[{timestamp, median}, ...]`
  - Low duty cycle might trigger config warnings but doesn't directly cause pruning

### Table: `run_history`
Persists completed/failed runs with fields:
- `run_id`, `config_path`, `status` (`"completed"` or `"failed"`)
- `start_time`, `end_time`, `total_epochs`, `final_loss`, `final_val_loss`
- `payload`: Full run state as JSON

---

## Hyperparameter Optimization (HPO) Pruning

**Source:** `models/hyperparameter_tuning.py`

### Trial States & Pruning Triggers

**Function:** `_summarize_trial_states(study)` line 34
- Counts trials by state: `COMPLETE`, `PRUNED`, `FAIL`, `RUNNING`

**Pruning Raised At:** lines 1865, 1869, 1872
```python
raise optuna.TrialPruned(
    f"Resource-constrained trial pruned after {attempt} retries: {exc}"  # Memory/resource limit
)
# OR
raise optuna.TrialPruned(
    f"Trial pruned due to non-resource error: {type(exc).__name__}: {exc}"  # Other errors
)
```

**Pruning logged to dashboard via:**
- `writer.update_hpo_progress(completed=0, pruned=int(counts['pruned']), ...)` line 2287, 2385, 2569
- Each trial's `status` field set to `"pruned"` and stored in `hpo_trial_results`

**Finding pruning reason in logs:**
- Search `run.log` for `"TrialPruned"`, `"Resource-constrained"`, or the trial number
- Line 2640, 2648, 2656 in hyperparameter_tuning.py log summaries of pruned/failed counts

---

## Dashboard HTML Rendering

**Source:** `observability/server.py`

### Run Status Display (Line 2231-2250)
**Endpoint:** `/ui/dashboard` (main dashboard)
- **Status badge**: `run_status` from `state.get("status")`
- **Pipeline stepper**: Shows stages completed; red ✗ on failed stage
  - Source: `_render_pipeline_stepper()` line 1953
  - Takes `status`, `stage`, `stage_timestamps` as input
- **Error box (expandable)**:
  ```html
  <details>
    <summary>Error: {last_error}</summary>
    {last_traceback}
  </details>
  ```
  - Lines 2248-2254: Rendered when `last_error` is not None

### Run Stats (`/ui/stats`, Line 2353-2427)
- **Start time, Runtime, Last Update**: From `start_time`, `heartbeat_time`
- **Progress counters**:
  - Snapshots: `snapshot_chunks_processed / total`
  - Epochs: `training_epochs_done / total` (hidden if `stage == "trial"`)
  - Batches: `training_batches_done / total`
  - Eval: `eval_batches_done / total`
  - **HPO Trials**: `hpo_trials_completed / total (pruned=X, failed=Y)` ← **Key diagnostic line**

### System Metrics (`/ui/system`, Line 2429-2517)
- **CPU/RAM**: `load_1m`, `mem_used_ratio` (root cause of timeout/crash?)
- **Run process RSS**: `run_process_rss_bytes` (main process memory)
- **HPO Worker memory**: 
  - Current: `hpo_wave_worker_rss_current_bytes`
  - Max historical: `hpo_wave_worker_rss_max_bytes`
  - Top 3: `hpo_wave_worker_rss_top` (detailed per-worker)
  - **RSS Watchdog triggers**: `hpo_rss_watchdog_trigger_count` (≥1 means memory-induced pruning)
  - Last trigger details: `_last_trigger_time`, `_pid`, `_rss_bytes`, `_limit_bytes`
- **GPU stats**: `utilization_ratio`, `memory_used_bytes/total`, `power_draw_watts`

### HPO Trial Table (`/ui/hpo`, Line 2519-2580)
Displays per-trial data:
- Trial #, Status badge (color-coded: `completed`, `pruned`, `failed`, `running`, `abandoned`)
- Objective value (loss/metric)
- Duration (seconds)
- **Phase RSS** (memory used during trial, `phase_rss_bytes`)
- All hyperparameters

Trial status interpretation:
- `"pruned"`: Intentionally stopped (resource or performance-based)
- `"running"`: Still executing (or abandoned if run finished)
- `"failed"`: Raised exception (check trial logs)
- `"completed"`: Finished successfully

---

## Log File Structure (`RUN_LOG_PATH`)

**Source:** `observability/server.py:315-360` (_tail_log function)

Typical content:
- Application startup logs (config validation)
- Data pipeline logs (snapshot building, feature extraction)
- Training logs (epoch progress, loss values)
- **Optuna logs**: Trial creation, pruning decisions, memory watchdog triggers
- Error tracebacks (if `set_error()` called)

**Searching for pruning evidence:**
```bash
grep -i "prune\|trial.*pruned\|watchdog\|resource\|memory" run.log
grep -i "error\|exception\|traceback" run.log
tail -n 100 run.log  # Last 100 lines before termination
```

---

## Environment & Configuration

**Sources:** 
- `observability/server.py:1-17` (environment variables)
- `config/observability.yaml` (non-secrets only)

**Critical env vars for diagnostics:**
- `RUN_STATE_PATH`: SQLite URI (`sqlite:///path/to/db.sqlite`)
- `RUN_LOG_PATH`: Path to run.log
- `OBSERVABILITY_TAIL_MAX_LINES`: How many log lines are served (default 200)

---

## Diagnostic Workflow

### Step 1: Check Dashboard Summary (`/api/run-state`)
```python
{
  "status": "failed",           # or "completed", "running", "idle"
  "stage": "trial",             # or "training", etc.
  "hpo_trials_pruned": 45,      # Pruned trial count
  "hpo_trials_failed": 3,       # Failed trials
  "hpo_rss_watchdog_trigger_count": 5,  # Memory watchdog hits
  "last_error": "Optuna error...",
  "last_traceback": "...",
  "progress": 0.75
}
```

### Step 2: Check Resource Usage (`/api/system`)
```python
{
  "mem_used_ratio": 0.92,                    # 92% of system RAM
  "hpo_rss_watchdog_trigger_count": 5,       # Memory limit hit 5 times
  "hpo_wave_worker_rss_current_bytes": 8e9,  # 8 GB (per worker)
  "hpo_wave_worker_rss_max_bytes": 9e9,      # 9 GB peak
  "hpo_wave_worker_rss_top": [
    {"pid": 12345, "rss_bytes": 8e9},
    ...
  ]
}
```

### Step 3: Check HPO Trials (`/api/hpo`)
```python
{
  "hpo_trials_total": 100,
  "hpo_trials_completed": 52,
  "hpo_trials_pruned": 45,      # ← Most trials pruned
  "hpo_trials_failed": 3,
  "trials": [
    {
      "number": 0,
      "status": "pruned",        # OR "completed", "failed"
      "value": null,
      "duration": 123,           # seconds
      "phase_rss_bytes": 5e9,    # Peak RSS during trial
      "params": {...}
    }
  ]
}
```

### Step 4: Check Raw Logs (`/api/logs` or `tmp/dashboard/run.log`)
```
[2026-04-05 10:30:45] Trial 0 started
[2026-04-05 10:31:02] Trial 0: epoch 1/10 loss=0.5
[2026-04-05 10:31:45] RSS Watchdog triggered: PID 12345, RSS 8.5GB > limit 8GB
[2026-04-05 10:31:45] Optuna TrialPruned: Trial 0 pruned after 1 retries: Memory limit exceeded
[2026-04-05 10:32:01] Trial 1 started
...
```

---

## Common Diagnostic Signals

| Signal | Location | Interpretation |
|--------|----------|-----------------|
| `status == "failed"` | `/api/run-state` | Run crashed or explicitly stopped |
| `last_error` is set | `/api/run-state` | Error message + traceback available |
| `hpo_trials_pruned > 0` | `/api/run-state`, `/api/hpo` | Optuna pruning occurred |
| `hpo_rss_watchdog_trigger_count > 0` | `/api/system` | **Memory limit hit** (root cause of pruning) |
| `hpo_wave_worker_rss_current_bytes` near limit | `/api/system` | Workers at memory ceiling |
| `mem_used_ratio > 0.90` | `/api/system` | System running out of RAM |
| `progress < 1.0 AND status == "completed"` | `/api/run-state` | Unusual (check logs) |
| `heartbeat_time` stale (> 60s ago) | `/api/run-state` | Process hung or crashed |
| `stage_timestamps` gaps | `/api/run-state` | Long stall in particular stage |
| Trial `status == "pruned"` with high `phase_rss_bytes` | `/api/hpo` | Trial likely pruned due to memory |
| `stage == "trial"` + `training_epochs_done == 0` | `/api/run-state` | Training never started (pruned in HPO) |

---

## Source Code Cross-Reference

| Concept | File | Key Functions/Lines |
|---------|------|---------------------|
| Run state persistence | `observability/run_state.py` | `RunStateWriter.set_error()` (405), `complete()` (416), `_save_history_locked()` (172) |
| HPO progress tracking | `observability/run_state.py` | `update_hpo_progress()` (266), `update_hpo_wave_memory()` (284), `mark_hpo_rss_watchdog_trigger()` (312) |
| Pruning logic | `models/hyperparameter_tuning.py` | Lines 1865, 1869, 1872; `_summarize_trial_states()` (34) |
| Dashboard endpoints | `observability/server.py` | `/api/run-state` (2947), `/api/hpo` (2968), `/api/system` (2958), `/api/logs` (2952) |
| Error display | `observability/server.py` | Lines 2248-2254 (error box in HTML) |
| Stats display | `observability/server.py` | `/ui/stats` (2353), lines 2422-2424 (HPO trial summary) |
| System metrics | `observability/server.py` | `/ui/system` (2429), lines 2486-2500 (watchdog + worker display) |
| Training progress | `observability/training_progress.py` | `create_training_progress_callback()`, epoch metrics at line 60 |

