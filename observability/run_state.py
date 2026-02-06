"""Run state tracking for long-running pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

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
    last_error: Optional[str] = None
    last_traceback: Optional[str] = None


class RunStateWriter:
    """Write run state to disk with atomic updates."""

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = RunState()

    @property
    def path(self) -> Path:
        return self._path

    def start(self, run_id: Optional[str] = None) -> None:
        with self._lock:
            now = time.time()
            self._state.start_time = now
            self._state.updated_time = now
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
        self._state.updated_time = time.time()
        self._write_locked()

    def _write_locked(self) -> None:
        payload = asdict(self._state)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self._path)


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
    """Load run state JSON from disk if available."""
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        return json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode run state JSON: %s", exc)
        return None


__all__ = [
    "RunState",
    "RunStateWriter",
    "get_run_state_writer",
    "load_run_state",
]
