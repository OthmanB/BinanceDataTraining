import os
import tempfile
import unittest
from typing import Tuple, cast

import numpy as np
from hypothesis import given, strategies as st

from training.snapshot_dataset import (
    GapHandler,
    SnapshotChunk,
    SnapshotDataset,
    SnapshotRecord,
    _compute_intensity_bins,
    build_training_generator,
    compute_normalization_stats,
    iter_snapshot_batches,
)


class TestSnapshotDataset(unittest.TestCase):
    def test_gap_handler_forward_fill(self) -> None:
        handler = GapHandler(
            cadence_seconds=10,
            max_gap_seconds=60,
            handle_gaps="forward_fill",
            check_missing_data=True,
            fail_on_invalid=True,
        )

        t0 = np.datetime64("2024-01-01T00:00:00")
        t1 = np.datetime64("2024-01-01T00:00:30")

        rec0 = SnapshotRecord(
            timestamp=t0,
            snapshot_features=[1.0, 1.0, 1.0, 1.0],
            depth=None,
            mid_price=100.0,
            hybrid_snapshot=None,
            volume_proxy=10.0,
        )
        rec1 = SnapshotRecord(
            timestamp=t1,
            snapshot_features=[2.0, 2.0, 2.0, 2.0],
            depth=None,
            mid_price=130.0,
            hybrid_snapshot=None,
            volume_proxy=12.0,
        )

        output = list(handler.iter_gap_handled([rec0, rec1]))
        self.assertEqual(len(output), 4)
        timestamps = [rec.timestamp for rec in output]
        expected = [
            t0,
            np.datetime64("2024-01-01T00:00:10"),
            np.datetime64("2024-01-01T00:00:20"),
            t1,
        ]
        self.assertEqual(timestamps, expected)
        self.assertEqual(output[1].mid_price, 100.0)
        self.assertEqual(output[2].mid_price, 100.0)

    def test_gap_handler_interpolate(self) -> None:
        handler = GapHandler(
            cadence_seconds=10,
            max_gap_seconds=60,
            handle_gaps="interpolate",
            check_missing_data=True,
            fail_on_invalid=True,
        )

        t0 = np.datetime64("2024-01-01T00:00:00")
        t1 = np.datetime64("2024-01-01T00:00:30")

        rec0 = SnapshotRecord(
            timestamp=t0,
            snapshot_features=[1.0, 1.0, 1.0, 1.0],
            depth=None,
            mid_price=100.0,
            hybrid_snapshot=None,
            volume_proxy=10.0,
        )
        rec1 = SnapshotRecord(
            timestamp=t1,
            snapshot_features=[2.0, 2.0, 2.0, 2.0],
            depth=None,
            mid_price=130.0,
            hybrid_snapshot=None,
            volume_proxy=12.0,
        )

        output = list(handler.iter_gap_handled([rec0, rec1]))
        self.assertEqual(len(output), 4)
        self.assertAlmostEqual(output[1].mid_price, 110.0, places=4)
        self.assertAlmostEqual(output[2].mid_price, 120.0, places=4)

    def test_compute_normalization_stats_min_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x = np.asarray(
                [
                    [[[[1.0]]]],
                    [[[[2.0]]]],
                    [[[[3.0]]]],
                    [[[[4.0]]]],
                ],
                dtype="float32",
            )
            y_up = np.zeros((4,), dtype="int64")
            y_down = np.zeros((4,), dtype="int64")
            anchor_ts = np.arange(4, dtype="int64")

            chunk_path = os.path.join(tmp_dir, "chunk.npz")
            np.savez_compressed(chunk_path, x=x, y_up=y_up, y_down=y_down, anchor_ts=anchor_ts)

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

            stats = compute_normalization_stats(dataset, 0, 4, method="min_max")
            x_flat = x.reshape(x.shape[0], -1)
            expected_min = np.min(x_flat, axis=0)
            expected_max = np.max(x_flat, axis=0)

            assert stats.min is not None
            assert stats.max is not None
            np.testing.assert_allclose(stats.min, expected_min)
            np.testing.assert_allclose(stats.max, expected_max)

    def test_iter_snapshot_batches_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x1 = np.asarray(
                [
                    [[[[0.0]]]],
                    [[[[1.0]]]],
                    [[[[2.0]]]],
                ],
                dtype="float32",
            )
            x2 = np.asarray(
                [
                    [[[[3.0]]]],
                    [[[[4.0]]]],
                    [[[[5.0]]]],
                ],
                dtype="float32",
            )

            def _save_chunk(path: str, x: np.ndarray) -> None:
                n = x.shape[0]
                np.savez_compressed(
                    path,
                    x=x,
                    y_up=np.zeros((n,), dtype="int64"),
                    y_down=np.zeros((n,), dtype="int64"),
                    anchor_ts=np.arange(n, dtype="int64"),
                )

            chunk1_path = os.path.join(tmp_dir, "chunk1.npz")
            chunk2_path = os.path.join(tmp_dir, "chunk2.npz")
            _save_chunk(chunk1_path, x1)
            _save_chunk(chunk2_path, x2)

            chunk1 = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk1_path,
                num_samples=3,
                start_index=0,
            )
            chunk2 = SnapshotChunk(
                start="2024-01-01 01:00:01",
                end="2024-01-01 02:00:00",
                file_path=chunk2_path,
                num_samples=3,
                start_index=3,
            )

            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk1, chunk2],
                total_samples=6,
                config_hash="hash",
            )

            collected = []
            for x_batch, _, _, _ in iter_snapshot_batches(dataset, 2, 5):
                collected.append(x_batch)

            result = np.concatenate(collected, axis=0)
            expected = np.concatenate([x1[2:3], x2[:2]], axis=0)
            np.testing.assert_allclose(result, expected)

    def test_build_training_generator_with_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x = np.asarray(
                [
                    [[[[1.0]]]],
                    [[[[2.0]]]],
                    [[[[3.0]]]],
                ],
                dtype="float32",
            )
            y_up = np.asarray([0, 1, 1], dtype="int64")
            y_down = np.asarray([0, 1, 1], dtype="int64")
            anchor_ts = np.asarray([0, 86400, 2 * 86400], dtype="int64")

            chunk_path = os.path.join(tmp_dir, "chunk.npz")
            np.savez_compressed(chunk_path, x=x, y_up=y_up, y_down=y_down, anchor_ts=anchor_ts)

            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=chunk_path,
                num_samples=3,
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=3,
                config_hash="hash",
            )

            sample_weight_cfg = {
                "enabled": True,
                "method": "exponential_decay",
                "half_life_days": 1,
            }

            gen, _ = build_training_generator(
                dataset=dataset,
                start_index=0,
                end_index=3,
                batch_size=2,
                num_classes=2,
                normalization=None,
                sample_weight_cfg=sample_weight_cfg,
            )

            x_batch, y_batch, sample_weight = next(iter(gen))
            self.assertEqual(x_batch.shape[0], 2)
            self.assertEqual(len(y_batch), 2)
            self.assertIsNotNone(sample_weight)
            assert sample_weight is not None
            self.assertEqual(len(sample_weight), 2)

            sample_weight_pair = cast(Tuple[np.ndarray, np.ndarray], sample_weight)
            w_up = np.asarray(sample_weight_pair[0], dtype="float32")
            w_down = np.asarray(sample_weight_pair[1], dtype="float32")
            self.assertTrue(np.allclose(w_up, w_down))

            expected_weights = np.array([0.25, 0.5], dtype="float32")
            self.assertTrue(np.allclose(w_up, expected_weights, rtol=1e-6, atol=1e-6))

    @given(
        boundaries=st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        ).map(lambda xs: sorted(set(round(x, 6) for x in xs)) or [0.1]),
        max_up=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        max_down=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_intensity_bins_within_range(self, boundaries, max_up, max_down) -> None:
        y_up, y_down = _compute_intensity_bins(boundaries, max_up, max_down)
        self.assertGreaterEqual(y_up, 0)
        self.assertGreaterEqual(y_down, 0)
        self.assertLessEqual(y_up, len(boundaries))
        self.assertLessEqual(y_down, len(boundaries))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
