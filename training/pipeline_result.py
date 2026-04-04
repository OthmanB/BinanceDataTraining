"""Structured result types for training pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PipelineResult:
    model: Optional[Any]
    epochs_ran: int = 0
    hpo_metric_value: Optional[float] = None
    hpo_metric_weight: float = 0.0


__all__ = ["PipelineResult"]
