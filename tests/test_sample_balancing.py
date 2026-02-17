"""Unit tests for coarse undersampling utilities."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from training.sample_balancing import (
    compute_available_class_counts,
    compute_class_counts_for_indices,
    compute_undersample_counts,
    select_undersampled_indices,
)
from training.snapshot_dataset import SnapshotChunk, SnapshotDataset
from utils.config_loader import ConfigError


class TestSampleBalancing(unittest.TestCase):
    def _make_npz_chunk(self, base_dir: str, name: str, y_up: np.ndarray, y_down: np.ndarray) -> str:
        path = os.path.join(base_dir, name)
        n = int(y_up.shape[0])
        np.savez_compressed(
            path,
            x=np.zeros((n, 1, 1, 1, 1), dtype=np.float32),
            y_up=y_up.astype("int64"),
            y_down=y_down.astype("int64"),
            anchor_ts=np.arange(n, dtype="int64"),
            duty_cycle=np.ones((n,), dtype=np.float32),
        )
        return path

    def test_compute_undersample_counts_fails_fast_on_missing_target_class(self) -> None:
        with self.assertRaises(ConfigError):
            compute_undersample_counts(
                available_counts=[10, 0, 3, 1],
                target_distribution=[1.0, 1.0, 0.0, 0.0],
            )

    def test_uniform_time_selection_is_deterministic_and_hits_targets(self) -> None:
        # Labels 0,0,1,1,2,2,3,3,3,3
        y_up = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3], dtype="int64")
        y_down = np.zeros_like(y_up)

        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = self._make_npz_chunk(tmp, "chunk.npz", y_up=y_up, y_down=y_down)
            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=int(y_up.shape[0]),
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=int(y_up.shape[0]),
                config_hash="hash",
            )

            counts = compute_available_class_counts(
                dataset=dataset,
                start_index=0,
                end_index=10,
                num_classes=4,
                labeling_criteria="max_intensity",
            )
            self.assertEqual(counts, [2, 2, 2, 4])

            keep = compute_undersample_counts(
                available_counts=counts,
                target_distribution=[1.0, 1.0, 1.0, 1.0],
            )
            self.assertEqual(keep, [2, 2, 2, 2])

            selected = select_undersampled_indices(
                dataset=dataset,
                start_index=0,
                end_index=10,
                num_classes=4,
                labeling_criteria="max_intensity",
                keep_counts=keep,
                selection_policy="uniform_time",
                random_seed=0,
            )

            self.assertEqual(selected.tolist(), [0, 1, 2, 3, 4, 5, 7, 9])

    def test_random_selection_returns_correct_size(self) -> None:
        y_up = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3], dtype="int64")
        y_down = np.zeros_like(y_up)

        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = self._make_npz_chunk(tmp, "chunk.npz", y_up=y_up, y_down=y_down)
            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=int(y_up.shape[0]),
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=int(y_up.shape[0]),
                config_hash="hash",
            )

            counts = [2, 2, 2, 4]
            keep = [1, 1, 1, 1]
            selected = select_undersampled_indices(
                dataset=dataset,
                start_index=0,
                end_index=10,
                num_classes=4,
                labeling_criteria="max_intensity",
                keep_counts=keep,
                selection_policy="random",
                random_seed=123,
            )

        self.assertEqual(int(selected.shape[0]), 4)
        self.assertTrue(bool(np.all(selected[:-1] <= selected[1:])))

        # Validate per-class counts from selected indices.
        y_bal = y_up[selected]
        binc = np.bincount(y_bal, minlength=4)
        self.assertEqual(binc.tolist(), keep)

    def test_compute_class_counts_for_indices_matches_available_on_full_range(self) -> None:
        y_up = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3], dtype="int64")
        y_down = np.zeros_like(y_up)

        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = self._make_npz_chunk(tmp, "chunk.npz", y_up=y_up, y_down=y_down)
            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=int(y_up.shape[0]),
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=int(y_up.shape[0]),
                config_hash="hash",
            )

            all_indices = np.arange(0, int(y_up.shape[0]), dtype="int64")
            by_range = compute_available_class_counts(
                dataset=dataset,
                start_index=0,
                end_index=int(y_up.shape[0]),
                num_classes=4,
                labeling_criteria="max_intensity",
            )
            by_indices = compute_class_counts_for_indices(
                dataset=dataset,
                indices=all_indices,
                num_classes=4,
                labeling_criteria="max_intensity",
            )

            self.assertEqual(by_indices, by_range)

    def test_compute_undersample_counts_auto_skips_missing_classes(self) -> None:
        counts = [10, 0, 4, 0]
        keep = compute_undersample_counts(
            available_counts=counts,
            target_distribution=[0.0],
        )
        self.assertEqual(keep, [4, 0, 4, 0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
