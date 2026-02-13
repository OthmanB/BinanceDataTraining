"""Tests for snapshot-native diagnostics."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from diagnostics.snapshot_diagnostics import (
    DIAGNOSTICS_MODE_PER_SNAPSHOT,
    DIAGNOSTICS_MODE_STANDALONE,
    _extract_top_of_book,
    _resolve_diagnostics_artifact_path,
    resolve_diagnostics_execution_mode,
    run_snapshot_diagnostics,
)


class _FakeSnapshotDataset:
    def __init__(self, snapshot_dir: str) -> None:
        self.total_samples = 4
        self.snapshot_dir = snapshot_dir


class TestSnapshotDiagnostics(unittest.TestCase):
    def test_extract_top_of_book_uses_field_axis_for_hybrid_shape(self) -> None:
        # sample shape: (T, levels, fields, channels)
        # fields order: [bid_price, bid_qty, ask_price, ask_qty]
        sample = np.zeros((2, 40, 4, 3), dtype=np.float32)
        sample[-1, 0, 0, 0] = 100.0
        sample[-1, 0, 1, 0] = 5.0
        sample[-1, 0, 2, 0] = 101.0
        sample[-1, 0, 3, 0] = 6.0
        # Different values on level 1 ensure we are not reading wrong indices
        sample[-1, 1, 0, 0] = 90.0
        sample[-1, 1, 1, 0] = 7.0
        sample[-1, 1, 2, 0] = 91.0
        sample[-1, 1, 3, 0] = 8.0

        extracted = _extract_top_of_book(sample)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        bid_price, bid_qty, ask_price, ask_qty = extracted
        self.assertEqual(bid_price, 100.0)
        self.assertEqual(bid_qty, 5.0)
        self.assertEqual(ask_price, 101.0)
        self.assertEqual(ask_qty, 6.0)
        self.assertGreater(ask_price - bid_price, 0.0)

    def test_extract_top_of_book_supports_2x2_layout(self) -> None:
        sample = np.zeros((3, 2, 2, 1), dtype=np.float32)
        sample[-1, 0, 0, 0] = 100.0
        sample[-1, 0, 1, 0] = 4.0
        sample[-1, 1, 0, 0] = 101.0
        sample[-1, 1, 1, 0] = 5.0

        extracted = _extract_top_of_book(sample)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted, (100.0, 4.0, 101.0, 5.0))

    def test_resolve_diagnostics_artifact_path_scopes(self) -> None:
        self.assertEqual(_resolve_diagnostics_artifact_path(None), "snapshot_diagnostics")
        self.assertEqual(_resolve_diagnostics_artifact_path(""), "snapshot_diagnostics")
        self.assertEqual(
            _resolve_diagnostics_artifact_path("window 1/2"),
            "snapshot_diagnostics/window_1_2",
        )
        self.assertEqual(
            _resolve_diagnostics_artifact_path("Window-2"),
            "snapshot_diagnostics/window_2",
        )

    def test_resolve_diagnostics_execution_mode_defaults_to_standalone(self) -> None:
        self.assertEqual(resolve_diagnostics_execution_mode({}), DIAGNOSTICS_MODE_STANDALONE)

    def test_resolve_diagnostics_execution_mode_accepts_per_snapshot(self) -> None:
        config = {"diagnostics": {"execution_mode": "per_snapshot"}}
        self.assertEqual(resolve_diagnostics_execution_mode(config), DIAGNOSTICS_MODE_PER_SNAPSHOT)

    def test_resolve_diagnostics_execution_mode_rejects_invalid_value(self) -> None:
        config = {"diagnostics": {"execution_mode": "invalid"}}
        with self.assertRaises(ValueError):
            resolve_diagnostics_execution_mode(config)

    def test_run_snapshot_diagnostics_with_snapshot_inputs(self) -> None:
        config = {
            "diagnostics": {
                "enabled": True,
                "execution_mode": "standalone",
                "sampling": {"method": "uniform", "num_samples": 2, "random_seed": 42},
                "gap_checks": {"large_gap_multiplier": 2.0, "very_large_gap_multiplier": 6.0},
                "visualization": {"enabled": False, "time_series": False, "histograms": False, "histogram_bins": 10},
            },
            "preprocessing": {"train_test_split": {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15}},
            "data": {"time_range": {"cadence_seconds": 10}},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            import os

            from training.snapshot_dataset import SnapshotChunk, SnapshotDataset

            x = np.zeros((4, 1, 1, 1, 1), dtype=np.float32)
            y_up = np.asarray([0, 1, 0, 1], dtype=np.int64)
            y_down = np.asarray([1, 0, 1, 0], dtype=np.int64)
            anchor_ts = np.asarray([0, 10, 20, 30], dtype=np.int64)
            duty_cycle = np.asarray([1.0, 0.5, 1.0, 0.5], dtype=np.float32)

            chunk_path = os.path.join(tmp_dir, "chunk.npz")
            np.savez_compressed(
                chunk_path,
                x=x,
                y_up=y_up,
                y_down=y_down,
                anchor_ts=anchor_ts,
                duty_cycle=duty_cycle,
            )

            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=4,
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=4,
                config_hash="hash",
            )

            def _fake_load_snapshot_series(_dataset: _FakeSnapshotDataset):
                return (
                    np.asarray([0, 10, 20, 30], dtype=np.int64),
                    np.asarray([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
                    np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                )

            def _fake_load_anchor_timestamps(_dataset: _FakeSnapshotDataset):
                return anchor_ts

            with mock.patch("diagnostics.snapshot_diagnostics.prepare_snapshot_dataset", return_value=dataset), mock.patch(
                "diagnostics.snapshot_diagnostics.load_anchor_timestamps",
                side_effect=_fake_load_anchor_timestamps,
            ), mock.patch(
                "diagnostics.snapshot_diagnostics.load_snapshot_series",
                side_effect=_fake_load_snapshot_series,
            ):
                run_snapshot_diagnostics(config)


if __name__ == "__main__":
    unittest.main()
