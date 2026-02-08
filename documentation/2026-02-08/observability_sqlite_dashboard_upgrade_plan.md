## Observability Dashboard Upgrade Plan (SQLite-Only + System Telemetry)

Date: 2026-02-08

### Goals

- Enforce SQLite-only run-state backend for observability.
- Remove legacy file-based run-state code paths.
- Improve dashboard state freshness and source consistency.
- Add host/system telemetry (CPU, RAM, GPU) to UI and `/metrics`.
- Keep behavior robust on systems without NVIDIA tooling.

### Scope

1. SQLite-only run-state persistence.
2. Dashboard/run-control consistency hardening.
3. Resource telemetry and visualization additions.
4. Cleanup of obsolete code branches.
5. Validation tests and rollout notes.

### Implementation Steps

#### 1) Enforce SQLite-Only Run State

- File: `observability/run_state.py`
  - Remove JSON file write/read fallback branches.
  - Accept only `sqlite:///...` in `RunStateWriter`.
  - Fail fast on non-SQLite `RUN_STATE_PATH`.
  - Keep the single-row state table strategy (`id=1`) for atomic updates.
- File: `observability/server.py`
  - Validate `run_state_path` format as SQLite URI at startup.
  - Continue to support env override precedence, but SQLite-only.
- File: `config/observability.yaml`
  - Remove legacy wording for file path backend.
  - Document SQLite URI as the only supported run-state format.

#### 2) Fix Dashboard State Refresh Source Consistency

- Ensure run-control start path injects the same `RUN_STATE_PATH` and `RUN_LOG_PATH` used by server config.
- Add visible diagnostics in status panel:
  - active run-state source,
  - heartbeat age,
  - stale-state indicator if age exceeds threshold.
- Add defensive warning when server and controlled process paths diverge.

#### 3) Add System Telemetry (CPU/RAM/GPU)

- File: `observability/server.py`
  - Add helper collectors (with graceful fallback):
    - CPU: `os.getloadavg()` + normalized load.
    - RAM: `/proc/meminfo` parsing for total/available/used bytes.
    - GPU: `nvidia-smi` query parser for util, memory, power.
  - Add short TTL cache (2-5s) for expensive probes (especially GPU calls).
- Dashboard UI:
  - Add `/ui/system` panel showing host and per-GPU stats.
- Prometheus `/metrics`:
  - host memory gauges/ratios,
  - load average gauges,
  - per-GPU utilization/memory/power gauges with labels.

#### 4) Add Run Process Memory Visibility

- Extend run-state payload with process IDs where possible:
  - main process PID,
  - (optional, later) current worker PIDs from HPO supervisor.
- UI should display main PID RSS and optionally top worker RSS values.

#### 5) Remove Obsolete/Dead Paths

- Delete code branches tied to legacy run-state JSON files.
- Remove stale comments/docstrings referring to file-based run-state support.
- Keep API symbols stable where practical to minimize unnecessary churn.

### Testing and Validation

- Add focused tests (new module suggested):
  - SQLite writer/read semantics and startup validation.
  - Non-SQLite path rejection.
  - Telemetry parsing helpers (including malformed output cases).
- Manual verification endpoints:
  - `/ui/status`, `/ui/stats`, `/ui/system`, `/api/run`, `/metrics`.
- Confirm stale heartbeat behavior and data freshness in UI updates.

### Risks and Mitigations

- `nvidia-smi` polling overhead:
  - mitigate via TTL cache and low query frequency.
- No-NVIDIA hosts:
  - return `n/a` in UI and skip GPU metric emission safely.
- SQLite lock contention:
  - keep write payload small and single-row upsert pattern.

### Suggested Follow-ups

1. Add `run_state_stale` metric for alerting.
2. Add dashboard badge for stale state age threshold breach.
3. Expose HPO wave-level RSS max/current metrics from supervisor logic.
4. Add run-state provenance details (writer PID/start time) for debugging.
