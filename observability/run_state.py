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

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._state.stage = stage
            if self._state.status != "failed":
                self._state.status = "running"
            self._touch_locked()

    def heartbeat(self, stage: Optional[str] = None) -> None:
        with self._lock:
            if stage is not None:
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
            self._state.stage = "snapshot_build"
            self._state.snapshot_chunks_processed = processed
            self._state.snapshot_chunks_total = total
            self._update_progress_locked(processed, total)

    def update_training_progress(
        self,
        epochs_done: int,
        epochs_total: int,
        batches_done: int,
        batches_total: int,
    ) -> None:
        with self._lock:
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
            self._state.stage = "evaluation"
            self._state.eval_batches_done = processed
            self._state.eval_batches_total = total
            self._update_progress_locked(processed, total)

    def update_duty_cycle_stats(self, minimum: float, median: float, p95: float) -> None:
        with self._lock:
            self._state.duty_cycle_min = float(minimum)
            self._state.duty_cycle_median = float(median)
            self._state.duty_cycle_p95 = float(p95)
            self._touch_locked()

    def set_error(self, message: str, traceback_text: Optional[str] = None) -> None:
        with self._lock:
            self._state.status = "failed"
            self._state.last_error = message
            self._state.last_traceback = traceback_text
            self._touch_locked()

    def complete(self) -> None:
        with self._lock:
            self._state.status = "completed"
            self._state.progress = 1.0
            self._state.eta_seconds = 0.0
            self._touch_locked()

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


def get_run_state_writer() -> Optional[RunStateWriter]:
    """Return the shared run state writer if configured via RUN_STATE_PATH."""
    global _WRITER
    path = os.environ.get("RUN_STATE_PATH")
    if not path:
        return None
    if _WRITER is None:
        _WRITER = RunStateWriter(path)
    return _WRITER


def load_run_state(path: str) -> Optional[Dict[str, Any]]:
    """Load run state JSON payload from sqlite if available."""
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
            return json.loads(payload)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read run state from sqlite: %s", exc)
        return None


__all__ = [
    "RunState",
    "RunStateWriter",
    "get_run_state_writer",
    "load_run_state",
    "_resolve_sqlite_path",
]
