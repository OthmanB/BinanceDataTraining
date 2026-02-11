"""Tests for bounded-memory snapshot chunk storage compatibility."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from training.snapshot_dataset import (
    CHUNK_STORAGE_NPY_SHARDS_V1,
    SnapshotChunk,
    SnapshotDataset,
    get_chunk_x_shape,
    iter_snapshot_batches,
    load_chunk_anchor_timestamps,
    load_snapshot_dataset,
)
from training.snapshot_store import SnapshotContext


def _write_chunk_npy_shards(base_dir: str, rel_dir: str, n_samples: int) -> dict:
    chunk_dir = os.path.join(base_dir, rel_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    x = np.arange(n_samples * 2, dtype="float32").reshape(n_samples, 1, 1, 2)
    y_up = np.arange(n_samples, dtype="int64") % 3
    y_down = (np.arange(n_samples, dtype="int64") + 1) % 3
    anchor_ts = (np.arange(n_samples, dtype="int64") + 1000) * 10
    duty_cycle = np.linspace(0.1, 1.0, num=n_samples, dtype="float32")

    files = {
        "x": os.path.join(rel_dir, "x.npy"),
        "y_up": os.path.join(rel_dir, "y_up.npy"),
        "y_down": os.path.join(rel_dir, "y_down.npy"),
        "anchor_ts": os.path.join(rel_dir, "anchor_ts.npy"),
        "duty_cycle": os.path.join(rel_dir, "duty_cycle.npy"),
    }
    np.save(os.path.join(base_dir, files["x"]), x, allow_pickle=False)
    np.save(os.path.join(base_dir, files["y_up"]), y_up, allow_pickle=False)
    np.save(os.path.join(base_dir, files["y_down"]), y_down, allow_pickle=False)
    np.save(os.path.join(base_dir, files["anchor_ts"]), anchor_ts, allow_pickle=False)
    np.save(os.path.join(base_dir, files["duty_cycle"]), duty_cycle, allow_pickle=False)
    return files


class TestSnapshotChunkStorage(unittest.TestCase):
    def test_load_snapshot_dataset_supports_npy_shards_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = _write_chunk_npy_shards(tmp_dir, os.path.join("chunks", "chunk0"), n_samples=6)
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config_hash": "hash123",
                        "chunks": [
                            {
                                "start": "2024-01-01 00:00:00",
                                "end": "2024-01-01 01:00:00",
                                "format": CHUNK_STORAGE_NPY_SHARDS_V1,
                                "file": files["x"],
                                "files": files,
                                "num_samples": 6,
                            }
                        ],
                    },
                    handle,
                )

            context = SnapshotContext(
                snapshot_dir=tmp_dir,
                manifest_path=manifest_path,
                config_hash="hash123",
                config_snapshot={},
                snapshot_name="test",
                root_name="test",
            )
            dataset = load_snapshot_dataset(context, config={})

            self.assertEqual(dataset.total_samples, 6)
            self.assertEqual(dataset.chunks[0].storage_format, CHUNK_STORAGE_NPY_SHARDS_V1)
            self.assertEqual(get_chunk_x_shape(dataset.chunks[0]), (6, 1, 1, 2))

            anchor_ts = load_chunk_anchor_timestamps(dataset.chunks[0])
            self.assertEqual(anchor_ts.shape, (6,))

            outputs = list(iter_snapshot_batches(dataset, 2, 5))
            self.assertEqual(len(outputs), 1)
            x, y_up, y_down, _, duty = outputs[0]
            self.assertEqual(x.shape[0], 3)
            self.assertEqual(y_up.shape[0], 3)
            self.assertEqual(y_down.shape[0], 3)
            self.assertEqual(duty.shape[0], 3)

    def test_legacy_npz_chunk_loading_still_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = os.path.join(tmp_dir, "chunk.npz")
            x = np.arange(12, dtype="float32").reshape(3, 1, 2, 2)
            y_up = np.array([0, 1, 2], dtype="int64")
            y_down = np.array([2, 1, 0], dtype="int64")
            anchor_ts = np.array([11, 22, 33], dtype="int64")
            duty_cycle = np.array([1.0, 0.7, 0.3], dtype="float32")
            np.savez_compressed(
                chunk_path,
                x=x,
                y_up=y_up,
                y_down=y_down,
                anchor_ts=anchor_ts,
                duty_cycle=duty_cycle,
            )

            dataset = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[
                    SnapshotChunk(
                        start="2024-01-01 00:00:00",
                        end="2024-01-01 01:00:00",
                        file_path=chunk_path,
                        num_samples=3,
                        start_index=0,
                    )
                ],
                total_samples=3,
                config_hash="hash",
            )

            batches = list(iter_snapshot_batches(dataset, 1, 3))
            self.assertEqual(len(batches), 1)
            x_out, y_up_out, y_down_out, anchor_out, duty_out = batches[0]
            np.testing.assert_array_equal(x_out, x[1:3])
            np.testing.assert_array_equal(y_up_out, y_up[1:3])
            np.testing.assert_array_equal(y_down_out, y_down[1:3])
            np.testing.assert_array_equal(anchor_out, anchor_ts[1:3])
            np.testing.assert_array_equal(duty_out, duty_cycle[1:3])


class TestSnapshotChunkStorageProperties(unittest.TestCase):
    @settings(max_examples=40)
    @given(
        n_samples=st.integers(min_value=1, max_value=60),
        start_ratio=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
        span_ratio=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_iter_snapshot_batches_matches_npz_and_npy_shards(
        self,
        n_samples: int,
        start_ratio: float,
        span_ratio: float,
    ) -> None:
        start_index = int(start_ratio * n_samples)
        span = max(1, int(span_ratio * n_samples))
        end_index = min(n_samples, start_index + span)
        if end_index <= start_index:
            end_index = min(n_samples, start_index + 1)

        x = np.arange(n_samples * 6, dtype="float32").reshape(n_samples, 1, 2, 3)
        y_up = np.arange(n_samples, dtype="int64") % 4
        y_down = (np.arange(n_samples, dtype="int64") + 1) % 4
        anchor_ts = np.arange(5000, 5000 + n_samples, dtype="int64")
        duty_cycle = np.linspace(0.2, 1.0, n_samples, dtype="float32")

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = os.path.join(tmp_dir, "chunk.npz")
            np.savez_compressed(
                npz_path,
                x=x,
                y_up=y_up,
                y_down=y_down,
                anchor_ts=anchor_ts,
                duty_cycle=duty_cycle,
            )

            shard_dir_rel = os.path.join("chunks", "chunk-shard")
            shard_dir = os.path.join(tmp_dir, shard_dir_rel)
            os.makedirs(shard_dir, exist_ok=True)
            files = {
                "x": os.path.join(shard_dir_rel, "x.npy"),
                "y_up": os.path.join(shard_dir_rel, "y_up.npy"),
                "y_down": os.path.join(shard_dir_rel, "y_down.npy"),
                "anchor_ts": os.path.join(shard_dir_rel, "anchor_ts.npy"),
                "duty_cycle": os.path.join(shard_dir_rel, "duty_cycle.npy"),
            }
            np.save(os.path.join(tmp_dir, files["x"]), x, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["y_up"]), y_up, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["y_down"]), y_down, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["anchor_ts"]), anchor_ts, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["duty_cycle"]), duty_cycle, allow_pickle=False)

            dataset_npz = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[
                    SnapshotChunk(
                        start="2024-01-01 00:00:00",
                        end="2024-01-01 01:00:00",
                        file_path=npz_path,
                        num_samples=n_samples,
                        start_index=0,
                    )
                ],
                total_samples=n_samples,
                config_hash="h1",
            )
            dataset_shards = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[
                    SnapshotChunk(
                        start="2024-01-01 00:00:00",
                        end="2024-01-01 01:00:00",
                        file_path=os.path.join(tmp_dir, files["x"]),
                        num_samples=n_samples,
                        start_index=0,
                        storage_format=CHUNK_STORAGE_NPY_SHARDS_V1,
                        array_paths={k: os.path.join(tmp_dir, v) for k, v in files.items()},
                    )
                ],
                total_samples=n_samples,
                config_hash="h2",
            )

            npz_batches = list(iter_snapshot_batches(dataset_npz, start_index, end_index))
            shard_batches = list(iter_snapshot_batches(dataset_shards, start_index, end_index))

            self.assertEqual(len(npz_batches), len(shard_batches))
            for npz_batch, shard_batch in zip(npz_batches, shard_batches):
                for npz_arr, shard_arr in zip(npz_batch, shard_batch):
                    np.testing.assert_array_equal(npz_arr, shard_arr)


if __name__ == "__main__":
    unittest.main()
