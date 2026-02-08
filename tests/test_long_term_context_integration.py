"""Integration tests for long-term context data loading and caching."""

import os
import tempfile
import unittest
from typing import Iterator, Tuple

import numpy as np

from training.long_term_context import (
    compute_long_term_features_for_dataset,
    wrap_generator_with_long_term,
)
from training.snapshot_dataset import SnapshotChunk, SnapshotDataset
from utils.config_loader import ConfigError


def _make_long_term_config(enabled: bool = True) -> dict:
    return {
        "model": {
            "long_term": {
                "enabled": enabled,
                "windows_days": [1],
                "resolution_days": 1,
                "features": ["mean_return"],
                "summary_method": "mean",
                "ewma_halflife_days": 7.0,
                "input_dim": None,
                "dense": {"layers": [32], "dropout_rates": [0.2]},
            },
        },
        "data": {
            "time_range": {"cadence_seconds": 60},
        },
    }


class TestLongTermContextIntegration(unittest.TestCase):
    """Tests for long-term feature computation from snapshot series."""

    def test_compute_long_term_features_from_snapshot_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunks_dir = os.path.join(tmp_dir, "chunks")
            series_dir = os.path.join(tmp_dir, "series")
            os.makedirs(chunks_dir, exist_ok=True)
            os.makedirs(series_dir, exist_ok=True)

            anchor_ts = np.array([86400, 2 * 86400], dtype="int64")
            chunk_file = os.path.join(chunks_dir, "chunk_0.npz")
            np.savez_compressed(chunk_file, anchor_ts=anchor_ts)

            series_timestamps = np.array([86400, 86460, 86520, 172800], dtype="int64")
            mid_prices = np.array([100.0, 101.0, 102.0, 103.0], dtype="float64")
            volumes = np.array([10.0, 11.0, 12.0, 13.0], dtype="float64")
            series_file_rel = os.path.join("series", "series_0.npz")
            series_file = os.path.join(tmp_dir, series_file_rel)
            np.savez_compressed(
                series_file,
                timestamps=series_timestamps,
                mid_prices=mid_prices,
                volumes=volumes,
            )

            manifest = {
                "config_hash": "testhash",
                "chunks": [
                    {
                        "start": "2020-01-01 00:00:00",
                        "end": "2020-01-01 01:00:00",
                        "file": os.path.join("chunks", "chunk_0.npz"),
                        "num_samples": int(anchor_ts.shape[0]),
                    }
                ],
                "series": {
                    "chunks": [
                        {
                            "start": "2020-01-01 00:00:00",
                            "end": "2020-01-01 01:00:00",
                            "file": series_file_rel,
                            "num_snapshots": int(series_timestamps.shape[0]),
                        }
                    ]
                },
            }

            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest=manifest,
                chunks=[
                    SnapshotChunk(
                        start="2020-01-01 00:00:00",
                        end="2020-01-01 01:00:00",
                        file_path=chunk_file,
                        num_samples=int(anchor_ts.shape[0]),
                        start_index=0,
                    )
                ],
                total_samples=int(anchor_ts.shape[0]),
                config_hash="testhash",
            )

            config = _make_long_term_config(enabled=True)
            features = compute_long_term_features_for_dataset(
                config,
                dataset,
                cadence_seconds=60,
            )

            self.assertEqual(features.shape[0], anchor_ts.shape[0])
            self.assertEqual(features.shape[1], 1)
            self.assertTrue(np.isfinite(features).all())

            cache_path = os.path.join(tmp_dir, "long_term_features.npz")
            self.assertTrue(os.path.exists(cache_path))

    def test_missing_series_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_file = os.path.join(tmp_dir, "chunks", "chunk_0.npz")
            os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
            anchor_ts = np.array([86400], dtype="int64")
            np.savez_compressed(chunk_file, anchor_ts=anchor_ts)

            manifest = {
                "config_hash": "testhash",
                "chunks": [
                    {
                        "start": "2020-01-01 00:00:00",
                        "end": "2020-01-01 01:00:00",
                        "file": os.path.join("chunks", "chunk_0.npz"),
                        "num_samples": int(anchor_ts.shape[0]),
                    }
                ],
            }

            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest=manifest,
                chunks=[
                    SnapshotChunk(
                        start="2020-01-01 00:00:00",
                        end="2020-01-01 01:00:00",
                        file_path=chunk_file,
                        num_samples=int(anchor_ts.shape[0]),
                        start_index=0,
                    )
                ],
                total_samples=int(anchor_ts.shape[0]),
                config_hash="testhash",
            )

            config = _make_long_term_config(enabled=True)
            with self.assertRaises(ConfigError):
                compute_long_term_features_for_dataset(config, dataset, cadence_seconds=60)


class TestWrapGeneratorWithLongTerm(unittest.TestCase):
    """Tests for wrap_generator_with_long_term function."""

    def _make_generator(
        self, batches: list, include_sample_weights: bool = False
    ) -> Iterator[Tuple]:
        """Create a mock generator yielding batches."""
        for batch in batches:
            if include_sample_weights:
                x, y, sw = batch
                yield (x, y, sw)
            else:
                x, y = batch
                yield (x, y)

    def test_wraps_xy_tuple_to_dual_input(self) -> None:
        """Basic (x, y) -> ([x, lt], y) wrapping."""
        batch_size = 4
        x = np.random.randn(batch_size, 10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size, 2)).astype(np.float32)

        long_term_features = np.random.randn(batch_size, 3).astype(np.float32)

        base_gen = self._make_generator([(x, y)])
        wrapped_gen = wrap_generator_with_long_term(base_gen, long_term_features, start_index=0)

        result = next(wrapped_gen)
        self.assertEqual(len(result), 2)  # ([x, lt], y)

        x_dual, y_out = result
        self.assertIsInstance(x_dual, tuple)
        self.assertEqual(len(x_dual), 2)
        np.testing.assert_array_equal(x_dual[0], x)
        np.testing.assert_array_equal(x_dual[1], long_term_features)
        np.testing.assert_array_equal(y_out, y)

    def test_wraps_xy_sw_tuple_to_dual_input(self) -> None:
        """With sample weights: (x, y, sw) -> ([x, lt], y, sw)."""
        batch_size = 4
        x = np.random.randn(batch_size, 10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size, 2)).astype(np.float32)
        sw = np.random.rand(batch_size).astype(np.float32)

        long_term_features = np.random.randn(batch_size, 3).astype(np.float32)

        base_gen = self._make_generator([(x, y, sw)], include_sample_weights=True)
        wrapped_gen = wrap_generator_with_long_term(base_gen, long_term_features, start_index=0)

        result = next(wrapped_gen)
        self.assertEqual(len(result), 3)  # ([x, lt], y, sw)

        x_dual, y_out, sw_out = result
        self.assertIsInstance(x_dual, tuple)
        self.assertEqual(len(x_dual), 2)
        np.testing.assert_array_equal(x_dual[0], x)
        np.testing.assert_array_equal(x_dual[1], long_term_features)
        np.testing.assert_array_equal(y_out, y)
        np.testing.assert_array_equal(sw_out, sw)

    def test_slices_long_term_features_correctly(self) -> None:
        """Long-term features are sliced by start_index."""
        batch_size = 2
        total_samples = 10
        start_index = 4

        x = np.random.randn(batch_size, 10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size, 2)).astype(np.float32)

        # Create long-term features with distinct values to verify slicing
        long_term_features = np.arange(total_samples * 3).reshape(total_samples, 3).astype(np.float32)

        base_gen = self._make_generator([(x, y)])
        wrapped_gen = wrap_generator_with_long_term(
            base_gen, long_term_features, start_index=start_index
        )

        result = next(wrapped_gen)
        x_dual, _ = result

        # Should get features from indices 4:6
        expected_lt = long_term_features[start_index : start_index + batch_size]
        np.testing.assert_array_equal(x_dual[1], expected_lt)

    def test_multiple_batches_accumulate_index(self) -> None:
        """Index advances correctly across multiple batches."""
        batch_size = 3
        num_batches = 4
        total_samples = batch_size * num_batches

        # Create multiple batches
        batches = []
        for i in range(num_batches):
            x = np.full((batch_size, 5, 2), i, dtype=np.float32)
            y = np.full((batch_size, 2), i, dtype=np.float32)
            batches.append((x, y))

        # Long-term features with distinct values per sample
        long_term_features = np.arange(total_samples * 2).reshape(total_samples, 2).astype(np.float32)

        base_gen = self._make_generator(batches)
        wrapped_gen = wrap_generator_with_long_term(base_gen, long_term_features, start_index=0)

        for batch_idx, result in enumerate(wrapped_gen):
            x_dual, _ = result
            start = batch_idx * batch_size
            end = start + batch_size
            expected_lt = long_term_features[start:end]
            np.testing.assert_array_equal(
                x_dual[1],
                expected_lt,
                err_msg=f"Batch {batch_idx}: long-term features mismatch",
            )

    def test_batch_size_mismatch_raises(self) -> None:
        """Raises ValueError if lt_batch size doesn't match x batch."""
        batch_size = 4
        x = np.random.randn(batch_size, 10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=(batch_size, 2)).astype(np.float32)

        # Long-term features with fewer samples than needed
        long_term_features = np.random.randn(2, 3).astype(np.float32)

        base_gen = self._make_generator([(x, y)])
        wrapped_gen = wrap_generator_with_long_term(base_gen, long_term_features, start_index=0)

        with self.assertRaises(ValueError) as ctx:
            next(wrapped_gen)

        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_start_index_offset_with_multiple_batches(self) -> None:
        """Verify correct indexing when start_index is non-zero."""
        batch_size = 2
        num_batches = 2
        start_index = 6
        total_samples = start_index + (batch_size * num_batches)

        batches = []
        for i in range(num_batches):
            x = np.full((batch_size, 3, 2), i, dtype=np.float32)
            y = np.full((batch_size, 1), i, dtype=np.float32)
            batches.append((x, y))

        long_term_features = np.arange(total_samples).reshape(total_samples, 1).astype(np.float32)

        base_gen = self._make_generator(batches)
        wrapped_gen = wrap_generator_with_long_term(
            base_gen, long_term_features, start_index=start_index
        )

        results = list(wrapped_gen)
        self.assertEqual(len(results), num_batches)

    def test_partial_batch_supported(self) -> None:
        """Partial final batches should be supported without mismatches."""
        batches = []
        batches.append((np.random.randn(24, 3, 2).astype(np.float32), np.zeros((24, 1), dtype=np.float32)))
        batches.append((np.random.randn(8, 3, 2).astype(np.float32), np.zeros((8, 1), dtype=np.float32)))

        long_term_features = np.random.randn(32, 4).astype(np.float32)
        base_gen = self._make_generator(batches)
        wrapped_gen = wrap_generator_with_long_term(base_gen, long_term_features, start_index=0)

        results = list(wrapped_gen)
        self.assertEqual(len(results), 2)
        first = results[0][0][1]
        second = results[1][0][1]
        self.assertEqual(first.shape[0], 24)
        self.assertEqual(second.shape[0], 8)

        np.testing.assert_array_equal(results[0][0][1], long_term_features[:24])
        np.testing.assert_array_equal(results[1][0][1], long_term_features[24:32])

    def test_epoch_boundary_reset(self) -> None:
        """Wrapper resets current_idx at epoch boundary (while-True base generator)."""
        batch_size = 4
        total_samples = 8
        num_epochs = 3

        # Build a base generator that loops infinitely (like build_training_generator)
        x_batch_1 = np.arange(batch_size * 2).reshape(batch_size, 2).astype(np.float32)
        y_batch_1 = np.zeros((batch_size, 1), dtype=np.float32)
        x_batch_2 = np.arange(batch_size * 2, total_samples * 2).reshape(batch_size, 2).astype(np.float32)
        y_batch_2 = np.ones((batch_size, 1), dtype=np.float32)

        def infinite_gen() -> Iterator[Tuple]:
            while True:
                yield (x_batch_1, y_batch_1)
                yield (x_batch_2, y_batch_2)

        long_term_features = np.arange(total_samples * 3).reshape(total_samples, 3).astype(np.float32)

        wrapped_gen = wrap_generator_with_long_term(
            infinite_gen(),
            long_term_features,
            start_index=0,
            end_index=total_samples,
        )

        steps_per_epoch = 2
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                result = next(wrapped_gen)
                x_dual, y_out = result
                lt_batch = x_dual[1]
                expected_start = step * batch_size
                expected_end = expected_start + batch_size
                np.testing.assert_array_equal(
                    lt_batch,
                    long_term_features[expected_start:expected_end],
                    err_msg=f"Mismatch at epoch={epoch}, step={step}",
                )

    def test_epoch_boundary_reset_with_val_offset(self) -> None:
        """Wrapper resets correctly for validation generators with non-zero start_index."""
        batch_size = 3
        val_start = 10
        val_end = 16
        num_epochs = 2

        x_batch_1 = np.ones((batch_size, 2), dtype=np.float32)
        y_batch_1 = np.zeros((batch_size, 1), dtype=np.float32)
        x_batch_2 = np.ones((batch_size, 2), dtype=np.float32) * 2
        y_batch_2 = np.ones((batch_size, 1), dtype=np.float32)

        def infinite_gen() -> Iterator[Tuple]:
            while True:
                yield (x_batch_1, y_batch_1)
                yield (x_batch_2, y_batch_2)

        total_features = 20
        long_term_features = np.arange(total_features * 2).reshape(total_features, 2).astype(np.float32)

        wrapped_gen = wrap_generator_with_long_term(
            infinite_gen(),
            long_term_features,
            start_index=val_start,
            end_index=val_end,
        )

        steps_per_epoch = 2
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                result = next(wrapped_gen)
                lt_batch = result[0][1]
                expected_start = val_start + step * batch_size
                expected_end = expected_start + batch_size
                np.testing.assert_array_equal(
                    lt_batch,
                    long_term_features[expected_start:expected_end],
                    err_msg=f"Val mismatch at epoch={epoch}, step={step}",
                )


if __name__ == "__main__":
    unittest.main()
