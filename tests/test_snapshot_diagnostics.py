"""Tests for snapshot-native diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from diagnostics.snapshot_diagnostics import run_snapshot_diagnostics


class _FakeSnapshotDataset:
    def __init__(self, snapshot_dir: str) -> None:
        self.total_samples = 4
        self.snapshot_dir = snapshot_dir


class TestSnapshotDiagnostics(unittest.TestCase):
    def test_run_snapshot_diagnostics_with_snapshot_inputs(self) -> None:
        config = {
            "diagnostics": {
                "enabled": True,
                "sampling": {"method": "uniform", "num_samples": 2, "random_seed": 42},
                "gap_checks": {"large_gap_multiplier": 2.0, "very_large_gap_multiplier": 6.0},
                "visualization": {"enabled": False, "time_series": False, "histograms": False, "histogram_bins": 10},
            },
            "preprocessing": {"train_test_split": {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15}},
            "data": {"time_range": {"cadence_seconds": 10}},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = _FakeSnapshotDataset(tmp_dir)

            def _iter_batches(_dataset: _FakeSnapshotDataset, _start: int, _end: int):
                yield (
                    np.zeros((2, 1, 1, 1, 1), dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int64),
                    np.asarray([1, 0], dtype=np.int64),
                    np.asarray([0, 10], dtype=np.int64),
                    np.asarray([1.0, 0.5], dtype=np.float32),
                )

            def _fake_load_snapshot_series(_dataset: _FakeSnapshotDataset):
                return (
                    np.asarray([0, 10, 20, 30], dtype=np.int64),
                    np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
                    np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                )

            def _fake_load_anchor_timestamps(_dataset: _FakeSnapshotDataset):
                return np.asarray([0, 10, 20, 30], dtype=np.int64)

            with mock.patch("diagnostics.snapshot_diagnostics.prepare_snapshot_dataset", return_value=dataset), mock.patch(
                "diagnostics.snapshot_diagnostics.iter_snapshot_batches",
                side_effect=_iter_batches,
            ), mock.patch(
                "diagnostics.snapshot_diagnostics.load_anchor_timestamps",
                side_effect=_fake_load_anchor_timestamps,
            ), mock.patch(
                "diagnostics.snapshot_diagnostics.load_snapshot_series",
                side_effect=_fake_load_snapshot_series,
            ):
                run_snapshot_diagnostics(config)


if __name__ == "__main__":
    unittest.main()
