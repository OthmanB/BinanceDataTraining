# Run Termination & Pruning: Quick Reference

## Live Dashboard (during run)
Hit these endpoints on `http://observability-server:port`:

| What to Check | Endpoint | Key Field |
|---------------|----------|-----------|
| Run alive? | `/api/run-state` | `status`, `heartbeat_time` (now - heartbeat should be < 60s) |
| Why failed? | `/api/run-state` | `last_error`, `last_traceback` |
| HPO pruning? | `/api/hpo` | `hpo_trials_pruned` > 0? Check `trials[].status == "pruned"` |
| Memory issue? | `/api/system` | `hpo_rss_watchdog_trigger_count` > 0? `mem_used_ratio`? |
| Progress stalled? | `/api/run-state` | `progress`, `eta_seconds`, `updated_time` |
| Logs | `/api/logs` | Last 200 lines: search for "error", "Pruned", "watchdog" |

## Dashboard HTML Panels (rendered at `/ui/*`)
- `/ui/dashboard` — Overall status + error box (if any)
- `/ui/stats` — Progress bars: snapshots, epochs, batches, **HPO trials (pruned=X)**
- `/ui/system` — **RSS Watchdog triggers**, CPU, RAM, GPU
- `/ui/hpo` — Per-trial table with status, duration, phase_rss_bytes

## SQLite Database (`RUN_STATE_PATH`)
```python
import sqlite3
conn = sqlite3.connect('path/to/db.sqlite')
cur = conn.cursor()

# Get current run state (JSON payload)
cur.execute('SELECT payload FROM run_state WHERE id = 1')
payload = cur.fetchone()[0]  # → JSON string

# Get history of all runs
cur.execute('SELECT run_id, status, start_time, final_loss FROM run_history')
for row in cur.fetchall():
    print(row)
```

## Log File (`RUN_LOG_PATH` → `tmp/dashboard/run.log`)
```bash
# Search for memory pruning
grep -i "watchdog\|pruned\|memory limit" run.log

# Search for any error
grep -i "error\|exception\|traceback" run.log

# Last activity before termination
tail -n 100 run.log
```

## Source Code: Where Each Signal Originates

**Status/Error:**
- `status` field → `observability/run_state.py:407, 418` (set_error/complete)
- `last_error`, `last_traceback` → `observability/run_state.py:405-414`

**Pruning:**
- `hpo_trials_pruned` count → `observability/run_state.py:266-282` (update_hpo_progress)
- Pruning raised → `models/hyperparameter_tuning.py:1865, 1869, 1872` (optuna.TrialPruned)
- Trial results → `observability/run_state.py:399-403` (update_hpo_trial_results)

**Memory Watchdog (→ pruning trigger):**
- Watchdog trigger count → `observability/run_state.py:312-319` (mark_hpo_rss_watchdog_trigger)
- Worker RSS tracking → `observability/run_state.py:284-310` (update_hpo_wave_memory)
- Dashboard display → `observability/server.py:2486-2500` (system metrics page)

**Progress & Heartbeat:**
- `progress`, `eta_seconds` → `observability/run_state.py:427-452` (_update_progress_locked)
- `heartbeat_time` → `observability/run_state.py:454-458` (_touch_locked)
- Stage tracking → `observability/run_state.py:233-239` (_record_stage_if_new_locked)

**Dashboard Rendering:**
- Error box HTML → `observability/server.py:2248-2254`
- Stats panel → `observability/server.py:2353-2427` (especially line 2422-2424 for HPO summary)
- System panel → `observability/server.py:2429-2517` (watchdog display at 2486-2500)
- HPO trial table → `observability/server.py:2519-2580` (status badges & phase_rss_bytes)

---

## Diagnostic Decision Tree

```
1. Is run still alive?
   ├─ Check /api/run-state → status == "running"?
   └─ Check heartbeat_time < 60 seconds old?
   
2. If NOT running, did it complete or fail?
   ├─ status == "completed" → Check progress (should be 1.0)
   ├─ status == "failed" → Read last_error + last_traceback
   └─ status == "idle" → Never started
   
3. If failed, is it a pruning issue?
   ├─ hpo_trials_pruned > 0? → YES, Optuna pruned trials
   │  ├─ Check /api/hpo → trials[].status == "pruned"
   │  └─ Check trial phase_rss_bytes (memory-bound?)
   └─ hpo_trials_pruned == 0? → NO, different error
       └─ Read last_error, check /api/logs for stack trace
   
4. If pruning, what caused it?
   ├─ hpo_rss_watchdog_trigger_count > 0? → MEMORY (resource exhaustion)
   │  ├─ Check hpo_wave_worker_rss_max_bytes (peak RSS)
   │  └─ Check mem_used_ratio (system-wide memory pressure)
   └─ hpo_rss_watchdog_trigger_count == 0? → OTHER (config, data issue, etc.)
       └─ Search run.log for trial error message
   
5. If progress stalled?
   ├─ Check stage_timestamps → Which stage took longest?
   ├─ Check heartbeat_time → Last activity when?
   └─ Check /api/logs → Any "Traceback" or timeout messages?
```

---

## Example: Diagnosing a Memory-Induced Pruning Run

**Scenario:** Run status is "failed", HPO was 50% done with mostly pruned trials.

```bash
# 1. Query the dashboard
curl http://localhost:8008/api/run-state | jq '.hpo_trials_pruned, .hpo_rss_watchdog_trigger_count'
# Output: 45, 5  ← Yes, memory watchdog fired 5 times

# 2. Check peak memory
curl http://localhost:8008/api/system | jq '.hpo_wave_worker_rss_max_bytes'
# Output: 9000000000  ← 9 GB (likely hit a 8 GB per-worker limit)

# 3. Check individual trial RSS
curl http://localhost:8008/api/hpo | jq '.trials[] | select(.status=="pruned") | .phase_rss_bytes' | sort -rn | head -1
# Output: 8500000000  ← 8.5 GB during a pruned trial

# 4. Confirm in logs
tail -n 500 tmp/dashboard/run.log | grep -i "watchdog\|pruned"
# Output: [2026-04-05 10:31:45] RSS Watchdog triggered: PID 12345, RSS 8.5GB > limit 8GB
#         [2026-04-05 10:31:45] Optuna TrialPruned: Trial 0 pruned after 1 retries: Memory limit exceeded

# → DIAGNOSIS: Workers exceeded 8 GB RSS limit; Optuna pruned them
# → ACTION: Increase per-worker memory limit or reduce worker count in config
```

---

## Environment Variables (Observability Config)

```bash
# Required for server startup
export OBSERVABILITY_USER="admin"
export OBSERVABILITY_PASSWORD="secret"
export RUN_STATE_PATH="sqlite:////home/user/run_state.sqlite"
export RUN_LOG_PATH="/home/user/tmp/dashboard/run.log"

# Optional (override YAML config)
export OBSERVABILITY_HOST="0.0.0.0"
export OBSERVABILITY_PORT="8008"
export OBSERVABILITY_TAIL_MAX_LINES="500"  # More log lines for diagnostics
export RUN_STATE_CACHE_TTL_SECONDS="1.0"   # Faster updates
```

---

## When to Use Each Source

| Source | When to Use | Why |
|--------|------------|-----|
| `/api/run-state` | Quick overall status | Single JSON, no HTML parsing |
| `/api/hpo` | Diagnose trial-level pruning | Per-trial details (status, RSS, params) |
| `/api/system` | Check resource limits | CPU, RAM, GPU, watchdog triggers |
| `/api/logs` | Deep error investigation | Full stack traces, application logs |
| SQLite DB | Historical analysis | All past runs, raw payloads, schema queries |
| `run.log` file | Raw timeline | Grep-friendly, timestamps, Optuna logs |
| Dashboard HTML (`/ui/*`) | Visual overview | Charts, formatted tables, human-readable |

