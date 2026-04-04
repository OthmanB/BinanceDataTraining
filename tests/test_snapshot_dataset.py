import os
import tempfile
import unittest
from typing import Tuple, cast

import numpy as np
from hypothesis import given, strategies as st

from training.snapshot_dataset import (
    CHUNK_STORAGE_NPY_SHARDS_V1,
    GapHandler,
    SnapshotChunk,
    SnapshotDataset,
    SnapshotRecord,
    _compute_intensity_bins,
    _compute_current_day,
    build_training_generator,
    build_training_generator_for_indices,
    compute_normalization_stats,
    iter_snapshot_batches,
    iter_snapshot_minibatches,
)


class TestSnapshotDataset(unittest.TestCase):
    def test_gap_handler_forward_fill(self) -> None:
        handler = GapHandler(
            cadence_seconds=10,
            validation_max_gap_seconds=60,
            alignment_max_gap_seconds=600,
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
            confidence=1.0,
            gap_reset=False,
            observed=True,
        )
        rec1 = SnapshotRecord(
            timestamp=t1,
            snapshot_features=[2.0, 2.0, 2.0, 2.0],
            depth=None,
            mid_price=130.0,
            hybrid_snapshot=None,
            volume_proxy=12.0,
            confidence=1.0,
            gap_reset=False,
            observed=True,
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
            validation_max_gap_seconds=60,
            alignment_max_gap_seconds=600,
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
            confidence=1.0,
            gap_reset=False,
            observed=True,
        )
        rec1 = SnapshotRecord(
            timestamp=t1,
            snapshot_features=[2.0, 2.0, 2.0, 2.0],
            depth=None,
            mid_price=130.0,
            hybrid_snapshot=None,
            volume_proxy=12.0,
            confidence=1.0,
            gap_reset=False,
            observed=True,
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
            duty_cycle = np.ones((4,), dtype="float32")

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

            stats = compute_normalization_stats(dataset, 0, 4, batch_size=2, method="min_max")
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
                    duty_cycle=np.ones((n,), dtype="float32"),
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
            for x_batch, _, _, _, _ in iter_snapshot_batches(dataset, 2, 5):
                collected.append(x_batch)

            result = np.concatenate(collected, axis=0)
            expected = np.concatenate([x1[2:3], x2[:2]], axis=0)
            np.testing.assert_allclose(result, expected)

    def test_compute_current_day_uses_anchor_timestamps_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            anchor_ts = np.asarray([0, 86400, 2 * 86400], dtype="int64")
            anchor_path = os.path.join(tmp_dir, "anchor_ts.npy")
            np.save(anchor_path, anchor_ts, allow_pickle=False)

            chunk = SnapshotChunk(
                start="2024-01-01 00:00:00",
                end="2024-01-01 01:00:00",
                file_path=os.path.join(tmp_dir, "missing_x.npy"),
                num_samples=3,
                start_index=0,
                storage_format=CHUNK_STORAGE_NPY_SHARDS_V1,
                array_paths={"anchor_ts": anchor_path},
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=3,
                config_hash="hash",
            )

            current_day = _compute_current_day(dataset, 0, 3)
            self.assertEqual(current_day, 2)

    def test_iter_snapshot_minibatches_drops_incomplete_tail_per_chunk(self) -> None:
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

            def _save_chunk(path: str, x: np.ndarray, base_ts: int) -> None:
                n = x.shape[0]
                np.savez_compressed(
                    path,
                    x=x,
                    y_up=np.arange(n, dtype="int64"),
                    y_down=np.arange(n, dtype="int64"),
                    anchor_ts=np.arange(base_ts, base_ts + n, dtype="int64"),
                    duty_cycle=np.ones((n,), dtype="float32"),
                )

            chunk1_path = os.path.join(tmp_dir, "chunk1.npz")
            chunk2_path = os.path.join(tmp_dir, "chunk2.npz")
            _save_chunk(chunk1_path, x1, 100)
            _save_chunk(chunk2_path, x2, 200)

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

            outputs = list(iter_snapshot_minibatches(dataset, 0, 6, batch_size=2))
            self.assertEqual(len(outputs), 2)

            x_batch0, y_up0, y_down0, anchor0, duty0 = outputs[0]
            x_batch1, y_up1, y_down1, anchor1, duty1 = outputs[1]

            np.testing.assert_allclose(x_batch0, x1[:2])
            np.testing.assert_array_equal(y_up0, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(y_down0, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(anchor0, np.array([100, 101], dtype="int64"))
            np.testing.assert_allclose(duty0, np.ones((2,), dtype="float32"))

            np.testing.assert_allclose(x_batch1, x2[:2])
            np.testing.assert_array_equal(y_up1, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(y_down1, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(anchor1, np.array([200, 201], dtype="int64"))
            np.testing.assert_allclose(duty1, np.ones((2,), dtype="float32"))

    def test_iter_snapshot_minibatches_includes_tail_when_drop_remainder_false(self) -> None:
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

            def _save_chunk(path: str, x: np.ndarray, base_ts: int) -> None:
                n = x.shape[0]
                np.savez_compressed(
                    path,
                    x=x,
                    y_up=np.arange(n, dtype="int64"),
                    y_down=np.arange(n, dtype="int64"),
                    anchor_ts=np.arange(base_ts, base_ts + n, dtype="int64"),
                    duty_cycle=np.ones((n,), dtype="float32"),
                )

            chunk1_path = os.path.join(tmp_dir, "chunk1.npz")
            chunk2_path = os.path.join(tmp_dir, "chunk2.npz")
            _save_chunk(chunk1_path, x1, 100)
            _save_chunk(chunk2_path, x2, 200)

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

            outputs = list(iter_snapshot_minibatches(dataset, 0, 6, batch_size=2, drop_remainder=False))
            self.assertEqual(len(outputs), 4)

            x_batch0, y_up0, y_down0, anchor0, duty0 = outputs[0]
            x_batch1, y_up1, y_down1, anchor1, duty1 = outputs[1]
            x_batch2, y_up2, y_down2, anchor2, duty2 = outputs[2]
            x_batch3, y_up3, y_down3, anchor3, duty3 = outputs[3]

            np.testing.assert_allclose(x_batch0, x1[:2])
            np.testing.assert_array_equal(y_up0, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(y_down0, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(anchor0, np.array([100, 101], dtype="int64"))
            np.testing.assert_allclose(duty0, np.ones((2,), dtype="float32"))

            np.testing.assert_allclose(x_batch1, x1[2:])
            np.testing.assert_array_equal(y_up1, np.array([2], dtype="int64"))
            np.testing.assert_array_equal(y_down1, np.array([2], dtype="int64"))
            np.testing.assert_array_equal(anchor1, np.array([102], dtype="int64"))
            np.testing.assert_allclose(duty1, np.ones((1,), dtype="float32"))

            np.testing.assert_allclose(x_batch2, x2[:2])
            np.testing.assert_array_equal(y_up2, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(y_down2, np.array([0, 1], dtype="int64"))
            np.testing.assert_array_equal(anchor2, np.array([200, 201], dtype="int64"))
            np.testing.assert_allclose(duty2, np.ones((2,), dtype="float32"))

            np.testing.assert_allclose(x_batch3, x2[2:])
            np.testing.assert_array_equal(y_up3, np.array([2], dtype="int64"))
            np.testing.assert_array_equal(y_down3, np.array([2], dtype="int64"))
            np.testing.assert_array_equal(anchor3, np.array([202], dtype="int64"))
            np.testing.assert_allclose(duty3, np.ones((1,), dtype="float32"))

    def test_build_training_generator_applies_duty_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x = np.asarray(
                [
                    [[[[1.0]]]],
                    [[[[2.0]]]],
                ],
                dtype="float32",
            )
            y_up = np.asarray([0, 1], dtype="int64")
            y_down = np.asarray([1, 0], dtype="int64")
            anchor_ts = np.asarray([0, 10], dtype="int64")
            duty_cycle = np.asarray([1.0, 0.5], dtype="float32")

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
                num_samples=2,
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=2,
                config_hash="hash",
            )

            gen, steps = build_training_generator(
                dataset=dataset,
                start_index=0,
                end_index=2,
                batch_size=2,
                num_classes=2,
                normalization=None,
                sample_weight_cfg=None,
            )

            self.assertEqual(steps, 1)
            batch = next(iter(gen))
            weights = batch[2]
            np.testing.assert_allclose(weights[0], duty_cycle)
            np.testing.assert_allclose(weights[1], duty_cycle)

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
            np.savez_compressed(
                chunk_path,
                x=x,
                y_up=y_up,
                y_down=y_down,
                anchor_ts=anchor_ts,
                duty_cycle=np.ones((x.shape[0],), dtype="float32"),
            )

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

    def test_build_training_generator_for_indices_yields_selected_samples_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            x = np.asarray(
                [
                    [[[[10.0]]]],
                    [[[[11.0]]]],
                    [[[[12.0]]]],
                    [[[[13.0]]]],
                    [[[[14.0]]]],
                    [[[[15.0]]]],
                ],
                dtype="float32",
            )
            y_up = np.asarray([0, 1, 0, 1, 0, 1], dtype="int64")
            y_down = np.asarray([1, 0, 1, 0, 1, 0], dtype="int64")
            anchor_ts = np.arange(6, dtype="int64")
            duty_cycle = np.ones((6,), dtype="float32")

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
                num_samples=int(x.shape[0]),
                start_index=0,
            )
            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[chunk],
                total_samples=int(x.shape[0]),
                config_hash="hash",
            )

            indices = np.asarray([0, 2, 4, 5], dtype="int64")
            gen, steps = build_training_generator_for_indices(
                dataset=dataset,
                indices=indices,
                batch_size=2,
                num_classes=2,
                normalization=None,
                sample_weight_cfg=None,
            )
            self.assertEqual(steps, 2)

            batch0 = next(iter(gen))
            x0 = batch0[0]
            y0_up = batch0[1][0]
            y0_down = batch0[1][1]
            np.testing.assert_allclose(x0, x[[0, 2]])
            np.testing.assert_array_equal(y0_up.argmax(axis=1), y_up[[0, 2]])
            np.testing.assert_array_equal(y0_down.argmax(axis=1), y_down[[0, 2]])

            batch1 = next(iter(gen))
            x1 = batch1[0]
            y1_up = batch1[1][0]
            y1_down = batch1[1][1]
            np.testing.assert_allclose(x1, x[[4, 5]])
            np.testing.assert_array_equal(y1_up.argmax(axis=1), y_up[[4, 5]])
            np.testing.assert_array_equal(y1_down.argmax(axis=1), y_down[[4, 5]])

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
