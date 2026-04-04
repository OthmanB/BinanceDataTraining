"""Run state tracking for long-running pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WRITER: Optional["RunStateWriter"] = None

# TTL cache for load_run_state() to reduce sqlite reads
_RUN_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
_RUN_STATE_CACHE_TIME: Dict[str, float] = {}
_RUN_STATE_CACHE_LOCK = threading.Lock()


def _get_run_state_cache_ttl() -> float:
    """Get configurable TTL for run state cache (default 2.0 seconds)."""
    env_val = os.environ.get("RUN_STATE_CACHE_TTL_SECONDS")
    if env_val is not None:
        try:
            return max(0.0, float(env_val))
        except ValueError:
            logger.warning("Invalid RUN_STATE_CACHE_TTL_SECONDS=%r, using default 2.0", env_val)
    return 2.0


class _WriterRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._writer: Optional["RunStateWriter"] = None

    def get(self) -> Optional["RunStateWriter"]:
        with self._lock:
            return self._writer

    def clear(self) -> None:
        global _WRITER
        with self._lock:
            self._writer = None
            _WRITER = None

    def get_or_create(self, path: str) -> Optional["RunStateWriter"]:
        global _WRITER
        with self._lock:
            if self._writer is None:
                try:
                    self._writer = RunStateWriter(path)
                except ValueError as exc:
                    logger.warning("Run-state writer disabled due to invalid RUN_STATE_PATH %r: %s", path, exc)
                    return None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Run-state writer failed to initialize for %r: %s", path, exc)
                    return None
            _WRITER = self._writer
            return self._writer


_WRITER_REGISTRY = _WriterRegistry()

_STAGE_ORDER: Dict[str, int] = {
    "idle": 0,
    "initializing": 1,
    "diagnostics": 2,
    "snapshot_build": 3,
    "trial": 4,
    "training": 5,
    "evaluation": 6,
}


@dataclass
class RunState:
    run_id: Optional[str] = None
    status: str = "idle"
    stage: str = "idle"
    progress: float = 0.0
    eta_seconds: Optional[float] = None
    start_time: Optional[float] = None
    updated_time: Optional[float] = None
    heartbeat_time: Optional[float] = None
    snapshot_chunks_total: int = 0
    snapshot_chunks_processed: int = 0
    training_epochs_total: int = 0
    training_epochs_done: int = 0
    training_batches_total: int = 0
    training_batches_done: int = 0
    eval_batches_total: int = 0
    eval_batches_done: int = 0
    duty_cycle_min: Optional[float] = None
    duty_cycle_median: Optional[float] = None
    duty_cycle_p95: Optional[float] = None
    hpo_trials_total: int = 0
    hpo_trials_completed: int = 0
    hpo_trials_pruned: int = 0
    hpo_trials_failed: int = 0
    hpo_wave_worker_count: int = 0
    hpo_wave_worker_rss_current_bytes: Optional[float] = None
    hpo_wave_worker_rss_max_bytes: Optional[float] = None
    hpo_wave_worker_rss_top: Optional[List[Dict[str, float]]] = None
    hpo_rss_watchdog_trigger_count: int = 0
    hpo_rss_watchdog_last_trigger_time: Optional[float] = None
    hpo_rss_watchdog_last_trigger_pid: Optional[int] = None
    hpo_rss_watchdog_last_trigger_rss_bytes: Optional[float] = None
    hpo_rss_watchdog_last_trigger_limit_bytes: Optional[float] = None
    run_process_pid: Optional[int] = None
    run_state_path: Optional[str] = None
    run_log_path: Optional[str] = None
    last_error: Optional[str] = None
    last_traceback: Optional[str] = None
    training_epoch_metrics: Optional[List[Dict[str, Any]]] = None
    stage_timestamps: Optional[Dict[str, float]] = None
    hpo_trial_results: Optional[List[Dict[str, Any]]] = None
    duty_cycle_history: Optional[List[Dict[str, Any]]] = None


class RunStateWriter:
    """Write run state to sqlite with atomic updates."""

    def __init__(self, path: str) -> None:
        self._raw_path = str(path).strip()
        self._sqlite_path = _resolve_sqlite_path(self._raw_path)
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_schema()
        self._lock = threading.Lock()
        self._state = RunState()

    @property
    def path(self) -> Path:
        return self._sqlite_path

    def _init_sqlite_schema(self) -> None:
        conn = sqlite3.connect(str(self._sqlite_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload TEXT NOT NULL,
                    updated_time REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    config_path TEXT,
                    status TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    total_epochs INTEGER,
                    final_loss REAL,
                    final_val_loss REAL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _save_history_locked(self) -> None:
        """Persist current run state as a history record (must hold _lock)."""
        state_dict = asdict(self._state)
        payload = json.dumps(state_dict, default=str)
        epoch_metrics = self._state.training_epoch_metrics or []
        final_loss: Optional[float] = None
        final_val_loss: Optional[float] = None
        if epoch_metrics:
            last = epoch_metrics[-1]
            final_loss = last.get("loss")
            final_val_loss = last.get("val_loss")
        conn = sqlite3.connect(str(self._sqlite_path))
        try:
            conn.execute(
                """
                INSERT INTO run_history
                    (run_id, config_path, status, start_time, end_time,
                     total_epochs, final_loss, final_val_loss, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._state.run_id,
                    None,
                    self._state.status,
                    self._state.start_time,
                    time.time(),
                    self._state.training_epochs_done,
                    final_loss,
                    final_val_loss,
                    payload,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def start(self, run_id: Optional[str] = None) -> None:
        with self._lock:
            now = time.time()
            self._state.run_process_pid = int(os.getpid())
            self._state.run_state_path = self._raw_path
            self._state.run_log_path = os.environ.get("RUN_LOG_PATH")
            self._state.start_time = now
            self._state.updated_time = now
            self._state.heartbeat_time = now
            self._state.status = "running"
            self._state.stage = "initializing"
            self._state.stage_timestamps = {"initializing": now}
            self._state.training_epoch_metrics = None
            self._state.progress = 0.0
            if run_id:
                self._state.run_id = run_id
            self._state.last_error = None
            self._state.last_traceback = None
            self._write_locked()

    def set_run_id(self, run_id: str) -> None:
        with self._lock:
            self._state.run_id = run_id
            self._touch_locked()

    def _record_stage_if_new_locked(self, stage: str) -> None:
        """Record a timestamp if transitioning to a new stage (must hold _lock)."""
        if self._state.stage != stage:
            if self._state.stage_timestamps is None:
                self._state.stage_timestamps = {}
            if stage not in self._state.stage_timestamps:
                self._state.stage_timestamps[stage] = time.time()

    def _is_stage_forward_locked(self, target: str) -> bool:
        """Return True if *target* is at or ahead of the current stage (must hold _lock)."""
        cur_order = _STAGE_ORDER.get(self._state.stage, -1)
        tgt_order = _STAGE_ORDER.get(target, -1)
        return tgt_order >= cur_order

    def set_stage(self, stage: str) -> None:
        with self._lock:
            if not self._is_stage_forward_locked(stage):
                return
            self._record_stage_if_new_locked(stage)
            self._state.stage = stage
            if self._state.status != "failed":
                self._state.status = "running"
            self._touch_locked()

    def heartbeat(self, stage: Optional[str] = None) -> None:
        with self._lock:
            if stage is not None and self._is_stage_forward_locked(stage):
                self._record_stage_if_new_locked(stage)
                self._state.stage = stage
            if self._state.status != "failed":
                self._state.status = "running"
            self._touch_locked()

    def update_hpo_progress(
        self,
        *,
        completed: int,
        total: int,
        pruned: int = 0,
        failed: int = 0,
    ) -> None:
        with self._lock:
            if self._is_stage_forward_locked("trial"):
                self._record_stage_if_new_locked("trial")
                self._state.stage = "trial"
            self._state.hpo_trials_total = max(0, int(total))
            self._state.hpo_trials_completed = max(0, int(completed))
            self._state.hpo_trials_pruned = max(0, int(pruned))
            self._state.hpo_trials_failed = max(0, int(failed))
            self._update_progress_locked(processed=self._state.hpo_trials_completed, total=self._state.hpo_trials_total)

    def update_hpo_wave_memory(self, rss_by_pid: Dict[int, int]) -> None:
        with self._lock:
            valid = {int(pid): int(rss) for pid, rss in rss_by_pid.items() if int(pid) > 0 and int(rss) > 0}
            self._state.hpo_wave_worker_count = len(valid)
            if not valid:
                self._state.hpo_wave_worker_rss_current_bytes = None
                self._state.hpo_wave_worker_rss_top = None
                self._touch_locked()
                return

            hottest_pid, hottest_rss = max(valid.items(), key=lambda item: item[1])
            self._state.hpo_wave_worker_rss_current_bytes = float(hottest_rss)
            historical = self._state.hpo_wave_worker_rss_max_bytes
            if historical is None:
                self._state.hpo_wave_worker_rss_max_bytes = float(hottest_rss)
            else:
                self._state.hpo_wave_worker_rss_max_bytes = max(float(historical), float(hottest_rss))

            top_workers = sorted(valid.items(), key=lambda item: item[1], reverse=True)[:3]
            self._state.hpo_wave_worker_rss_top = [
                {
                    "pid": float(pid),
                    "rss_bytes": float(rss),
                }
                for pid, rss in top_workers
            ]
            self._touch_locked()

    def mark_hpo_rss_watchdog_trigger(self, *, pid: int, rss_bytes: int, limit_bytes: int) -> None:
        with self._lock:
            self._state.hpo_rss_watchdog_trigger_count = int(self._state.hpo_rss_watchdog_trigger_count) + 1
            self._state.hpo_rss_watchdog_last_trigger_time = float(time.time())
            self._state.hpo_rss_watchdog_last_trigger_pid = int(pid)
            self._state.hpo_rss_watchdog_last_trigger_rss_bytes = float(rss_bytes)
            self._state.hpo_rss_watchdog_last_trigger_limit_bytes = float(limit_bytes)
            self._touch_locked()

    def update_snapshot_progress(self, processed: int, total: int) -> None:
        with self._lock:
            if self._is_stage_forward_locked("snapshot_build"):
                self._record_stage_if_new_locked("snapshot_build")
                self._state.stage = "snapshot_build"
                self._state.snapshot_chunks_processed = processed
                self._state.snapshot_chunks_total = total
                self._update_progress_locked(processed, total)
            else:
                self._touch_locked()

    def update_training_progress(
        self,
        epochs_done: int,
        epochs_total: int,
        batches_done: int,
        batches_total: int,
    ) -> None:
        with self._lock:
            if self._is_stage_forward_locked("training"):
                self._record_stage_if_new_locked("training")
                self._state.stage = "training"
            self._state.training_epochs_done = epochs_done
            self._state.training_epochs_total = epochs_total
            self._state.training_batches_done = batches_done
            self._state.training_batches_total = batches_total

            if epochs_total > 0 and batches_total > 0:
                epoch_fraction = min(batches_done / float(batches_total), 1.0)
                progress = (epochs_done + epoch_fraction) / float(epochs_total)
                self._update_progress_locked(progress=progress)
            else:
                self._touch_locked()

    def update_eval_progress(self, processed: int, total: int) -> None:
        with self._lock:
            if self._is_stage_forward_locked("evaluation"):
                self._record_stage_if_new_locked("evaluation")
                self._state.stage = "evaluation"
            self._state.eval_batches_done = processed
            self._state.eval_batches_total = total
            self._update_progress_locked(processed, total)

    _DUTY_CYCLE_HISTORY_CAP = 10000

    def update_duty_cycle_stats(self, minimum: float, median: float, p95: float) -> None:
        with self._lock:
            self._state.duty_cycle_min = float(minimum)
            self._state.duty_cycle_median = float(median)
            self._state.duty_cycle_p95 = float(p95)
            if self._state.duty_cycle_history is None:
                self._state.duty_cycle_history = []
            self._state.duty_cycle_history.append({
                "timestamp": time.time(),
                "median": float(median),
            })
            if len(self._state.duty_cycle_history) > self._DUTY_CYCLE_HISTORY_CAP:
                self._state.duty_cycle_history = self._state.duty_cycle_history[-self._DUTY_CYCLE_HISTORY_CAP:]
            self._touch_locked()

    def update_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Append per-epoch training metrics (loss, val_loss, etc.)."""
        with self._lock:
            if self._state.training_epoch_metrics is None:
                self._state.training_epoch_metrics = []
            entry: Dict[str, Any] = {"epoch": epoch}
            entry.update(metrics)
            self._state.training_epoch_metrics.append(entry)
            self._touch_locked()

    def record_stage_transition(self, stage: str) -> None:
        """Record a timestamp when a pipeline stage begins."""
        with self._lock:
            if self._state.stage_timestamps is None:
                self._state.stage_timestamps = {}
            self._state.stage_timestamps[stage] = time.time()
            self._touch_locked()

    def update_hpo_trial_results(self, trials: List[Dict[str, Any]]) -> None:
        """Replace the HPO trial results list with the latest snapshot."""
        with self._lock:
            self._state.hpo_trial_results = list(trials)
            self._touch_locked()

    def set_error(self, message: str, traceback_text: Optional[str] = None) -> None:
        with self._lock:
            self._state.status = "failed"
            self._state.last_error = message
            self._state.last_traceback = traceback_text
            self._touch_locked()
            try:
                self._save_history_locked()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to save run history on error")

    def complete(self) -> None:
        with self._lock:
            self._state.status = "completed"
            self._state.progress = 1.0
            self._state.eta_seconds = 0.0
            self._touch_locked()
            try:
                self._save_history_locked()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to save run history on completion")

    def _update_progress_locked(
        self,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        progress: Optional[float] = None,
    ) -> None:
        if progress is None:
            if processed is None or total is None or total <= 0:
                self._touch_locked()
                return
            progress = min(max(processed / float(total), 0.0), 1.0)

        self._state.progress = float(progress)
        self._update_eta_locked()
        self._touch_locked()

    def _update_eta_locked(self) -> None:
        if self._state.start_time is None:
            self._state.eta_seconds = None
            return
        if self._state.progress <= 0.0 or self._state.progress >= 1.0:
            self._state.eta_seconds = None
            return
        elapsed = time.time() - self._state.start_time
        remaining = elapsed * (1.0 / self._state.progress - 1.0)
        self._state.eta_seconds = max(0.0, remaining)

    def _touch_locked(self) -> None:
        now = time.time()
        self._state.updated_time = now
        self._state.heartbeat_time = now
        self._write_locked()

    def _write_locked(self) -> None:
        payload = asdict(self._state)
        conn = sqlite3.connect(str(self._sqlite_path))
        try:
            conn.execute(
                """
                INSERT INTO run_state (id, payload, updated_time)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_time = excluded.updated_time
                """,
                (json.dumps(payload, sort_keys=True), float(time.time())),
            )
            conn.commit()
        finally:
            conn.close()


def _resolve_sqlite_path(path: str) -> Path:
    normalized = str(path).strip()
    if not normalized.startswith("sqlite:///"):
        raise ValueError(
            "RUN_STATE_PATH must use sqlite URI format 'sqlite:///...'; "
            f"got {path!r}"
        )
    sqlite_file = normalized[len("sqlite:///") :]
    if not sqlite_file:
        raise ValueError("RUN_STATE_PATH sqlite URI must include a database file path")
    return Path(sqlite_file).expanduser().resolve()


def resolve_sqlite_path(path: str) -> Path:
    return _resolve_sqlite_path(path)


def get_run_state_writer() -> Optional[RunStateWriter]:
    """Return the shared run state writer if configured via RUN_STATE_PATH."""
    global _WRITER
    path = os.environ.get("RUN_STATE_PATH") or os.environ.get("OBSERVABILITY_RUN_STATE_PATH")
    if not path:
        return None
    if _WRITER is None and _WRITER_REGISTRY.get() is not None:
        _WRITER_REGISTRY.clear()
    if _WRITER is not None:
        return _WRITER
    return _WRITER_REGISTRY.get_or_create(path)


def clear_run_state_writer() -> None:
    _WRITER_REGISTRY.clear()
    with _RUN_STATE_CACHE_LOCK:
        _RUN_STATE_CACHE.clear()
        _RUN_STATE_CACHE_TIME.clear()


def load_run_state(path: str) -> Optional[Dict[str, Any]]:
    """Load run state JSON payload from sqlite if available."""
    now = time.time()
    normalized_path = str(path).strip()
    cache_ttl = _get_run_state_cache_ttl()
    
    with _RUN_STATE_CACHE_LOCK:
        cached_time = _RUN_STATE_CACHE_TIME.get(normalized_path, 0.0)
        if normalized_path in _RUN_STATE_CACHE and (now - cached_time) < cache_ttl:
            return dict(_RUN_STATE_CACHE[normalized_path])
    
    try:
        path_obj = _resolve_sqlite_path(path)
    except ValueError as exc:
        logger.warning("Invalid run state path: %s", exc)
        return None
    if not path_obj.exists():
        return None
    try:
        conn = sqlite3.connect(str(path_obj))
        try:
            row = conn.execute("SELECT payload FROM run_state WHERE id = 1").fetchone()
            if row is None:
                return None
            payload = row[0]
            if not isinstance(payload, str):
                return None
            parsed = json.loads(payload)
            
            with _RUN_STATE_CACHE_LOCK:
                _RUN_STATE_CACHE[normalized_path] = dict(parsed)
                _RUN_STATE_CACHE_TIME[normalized_path] = now
            
            return parsed
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read run state from sqlite: %s", exc)
        return None


def load_run_history(path: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Load recent run history records from sqlite."""
    try:
        path_obj = _resolve_sqlite_path(path)
    except ValueError as exc:
        logger.warning("Invalid run state path for history: %s", exc)
        return []
    if not path_obj.exists():
        return []
    try:
        conn = sqlite3.connect(str(path_obj))
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='run_history'"
            )
            if cursor.fetchone() is None:
                return []
            rows = conn.execute(
                """
                SELECT id, run_id, config_path, status, start_time, end_time,
                       total_epochs, final_loss, final_val_loss
                FROM run_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            columns = [
                "id", "run_id", "config_path", "status", "start_time", "end_time",
                "total_epochs", "final_loss", "final_val_loss",
            ]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read run history from sqlite: %s", exc)
        return []


__all__ = [
    "RunState",
    "RunStateWriter",
    "clear_run_state_writer",
    "get_run_state_writer",
    "load_run_state",
    "load_run_history",
    "resolve_sqlite_path",
]
