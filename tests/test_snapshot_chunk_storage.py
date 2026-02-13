"""Tests for bounded-memory snapshot chunk storage compatibility."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Tuple

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from training.snapshot_dataset import (
    CHUNK_STORAGE_FRAME_STORE_V1,
    CHUNK_STORAGE_NPY_SHARDS_V1,
    SnapshotChunk,
    SnapshotDataset,
    _open_chunk_sample_reader,
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


def _write_chunk_frame_store(base_dir: str, rel_dir: str) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunk_dir = os.path.join(base_dir, rel_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    # Frame timeline: F=6, H=1, W=1, A=1
    frames_base = np.arange(6, dtype="float32").reshape(6, 1, 1, 1)
    frames_confidence = np.linspace(0.1, 0.6, num=6, dtype="float32").reshape(6, 1)
    frames_observed = np.array([[1], [1], [1], [0], [1], [0]], dtype="uint8")
    frames_ts = np.array([100, 110, 120, 130, 140, 150], dtype="int64")

    # Two samples anchored on local frame indices 2 and 4, window_steps=3
    anchor_local_idx = np.array([2, 4], dtype="int32")
    y_up = np.array([1, 2], dtype="int64")
    y_down = np.array([2, 1], dtype="int64")
    anchor_ts = np.array([120, 140], dtype="int64")
    duty_cycle = np.array([1.0, 0.5], dtype="float32")
    aux = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32")

    files = {
        "frames_base": os.path.join(rel_dir, "frames_base.npy"),
        "frames_confidence": os.path.join(rel_dir, "frames_confidence.npy"),
        "frames_observed": os.path.join(rel_dir, "frames_observed.npy"),
        "frames_ts": os.path.join(rel_dir, "frames_ts.npy"),
        "anchor_local_idx": os.path.join(rel_dir, "anchor_local_idx.npy"),
        "aux": os.path.join(rel_dir, "aux.npy"),
        "y_up": os.path.join(rel_dir, "y_up.npy"),
        "y_down": os.path.join(rel_dir, "y_down.npy"),
        "anchor_ts": os.path.join(rel_dir, "anchor_ts.npy"),
        "duty_cycle": os.path.join(rel_dir, "duty_cycle.npy"),
    }

    np.save(os.path.join(base_dir, files["frames_base"]), frames_base, allow_pickle=False)
    np.save(os.path.join(base_dir, files["frames_confidence"]), frames_confidence, allow_pickle=False)
    np.save(os.path.join(base_dir, files["frames_observed"]), frames_observed, allow_pickle=False)
    np.save(os.path.join(base_dir, files["frames_ts"]), frames_ts, allow_pickle=False)
    np.save(os.path.join(base_dir, files["anchor_local_idx"]), anchor_local_idx, allow_pickle=False)
    np.save(os.path.join(base_dir, files["aux"]), aux, allow_pickle=False)
    np.save(os.path.join(base_dir, files["y_up"]), y_up, allow_pickle=False)
    np.save(os.path.join(base_dir, files["y_down"]), y_down, allow_pickle=False)
    np.save(os.path.join(base_dir, files["anchor_ts"]), anchor_ts, allow_pickle=False)
    np.save(os.path.join(base_dir, files["duty_cycle"]), duty_cycle, allow_pickle=False)

    expected_x = np.zeros((2, 3, 1, 1, 4), dtype="float32")
    # sample 0 -> frames [0,1,2]
    expected_x[0, :, 0, 0, 0] = np.array([0.0, 1.0, 2.0], dtype="float32")
    expected_x[0, :, 0, 0, 1] = np.array([0.1, 0.2, 0.3], dtype="float32")
    expected_x[0, :, 0, 0, 2] = 10.0
    expected_x[0, :, 0, 0, 3] = 20.0
    # sample 1 -> frames [2,3,4]
    expected_x[1, :, 0, 0, 0] = np.array([2.0, 3.0, 4.0], dtype="float32")
    expected_x[1, :, 0, 0, 1] = np.array([0.3, 0.4, 0.5], dtype="float32")
    expected_x[1, :, 0, 0, 2] = 30.0
    expected_x[1, :, 0, 0, 3] = 40.0

    return files, expected_x, y_up, y_down, anchor_ts, duty_cycle


class TestSnapshotChunkStorage(unittest.TestCase):
    def test_load_snapshot_dataset_supports_frame_store_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            files, expected_x, expected_y_up, expected_y_down, expected_anchor_ts, expected_duty = _write_chunk_frame_store(
                tmp_dir,
                os.path.join("chunks", "chunk-frame"),
            )

            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config_hash": "hash-frame",
                        "chunks": [
                            {
                                "start": "2024-01-01 00:00:00",
                                "end": "2024-01-01 01:00:00",
                                "format": CHUNK_STORAGE_FRAME_STORE_V1,
                                "file": files["frames_base"],
                                "files": files,
                                "num_samples": 2,
                                "window_steps": 3,
                                "include_mask_channel": True,
                                "num_assets": 1,
                                "aux_dim": 2,
                                "x_shape": [2, 3, 1, 1, 4],
                            }
                        ],
                    },
                    handle,
                )

            context = SnapshotContext(
                snapshot_dir=tmp_dir,
                manifest_path=manifest_path,
                config_hash="hash-frame",
                config_snapshot={},
                snapshot_name="test",
                root_name="test",
            )
            dataset = load_snapshot_dataset(context, config={})

            self.assertEqual(dataset.total_samples, 2)
            self.assertEqual(dataset.chunks[0].storage_format, CHUNK_STORAGE_FRAME_STORE_V1)
            self.assertEqual(get_chunk_x_shape(dataset.chunks[0]), (2, 3, 1, 1, 4))

            outputs = list(iter_snapshot_batches(dataset, 0, 2))
            self.assertEqual(len(outputs), 1)
            x, y_up, y_down, anchor_ts, duty = outputs[0]

            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(y_up, expected_y_up)
            np.testing.assert_array_equal(y_down, expected_y_down)
            np.testing.assert_array_equal(anchor_ts, expected_anchor_ts)
            np.testing.assert_array_equal(duty, expected_duty)

    def test_frame_store_sample_reader_matches_materialized_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            files, expected_x, expected_y_up, expected_y_down, expected_anchor_ts, expected_duty = _write_chunk_frame_store(
                tmp_dir,
                os.path.join("chunks", "chunk-frame"),
            )

            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config_hash": "hash-frame",
                        "chunks": [
                            {
                                "start": "2024-01-01 00:00:00",
                                "end": "2024-01-01 01:00:00",
                                "format": CHUNK_STORAGE_FRAME_STORE_V1,
                                "file": files["frames_base"],
                                "files": files,
                                "num_samples": 2,
                                "window_steps": 3,
                                "include_mask_channel": True,
                                "num_assets": 1,
                                "aux_dim": 2,
                                "x_shape": [2, 3, 1, 1, 4],
                            }
                        ],
                    },
                    handle,
                )

            context = SnapshotContext(
                snapshot_dir=tmp_dir,
                manifest_path=manifest_path,
                config_hash="hash-frame",
                config_snapshot={},
                snapshot_name="test",
                root_name="test",
            )
            dataset = load_snapshot_dataset(context, config={})
            chunk = dataset.chunks[0]

            reader = _open_chunk_sample_reader(chunk)
            try:
                x, y_up, y_down, anchor_ts, duty = reader.get_samples(np.array([0, 1], dtype="int64"))
            finally:
                try:
                    reader.close()
                except Exception:
                    pass

            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(y_up, expected_y_up)
            np.testing.assert_array_equal(y_down, expected_y_down)
            np.testing.assert_array_equal(anchor_ts, expected_anchor_ts)
            np.testing.assert_array_equal(duty, expected_duty)

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

    @settings(max_examples=30)
    @given(data=st.data())
    def test_iter_snapshot_batches_frame_store_matches_manual_reconstruction(self, data: st.DataObject) -> None:
        n_frames = data.draw(st.integers(min_value=4, max_value=12))
        window_steps = data.draw(st.integers(min_value=2, max_value=min(6, n_frames)))
        h_dim = data.draw(st.integers(min_value=1, max_value=2))
        w_dim = data.draw(st.integers(min_value=1, max_value=2))
        num_assets = data.draw(st.integers(min_value=1, max_value=2))
        include_mask = data.draw(st.booleans())
        aux_dim = data.draw(st.integers(min_value=0, max_value=3))

        max_samples = max(1, n_frames - window_steps + 1)
        n_samples = data.draw(st.integers(min_value=1, max_value=min(6, max_samples)))
        anchors = data.draw(
            st.lists(
                st.integers(min_value=window_steps - 1, max_value=n_frames - 1),
                min_size=n_samples,
                max_size=n_samples,
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            rel_dir = os.path.join("chunks", "chunk-frame-prop")
            chunk_dir = os.path.join(tmp_dir, rel_dir)
            os.makedirs(chunk_dir, exist_ok=True)

            frames_base = np.arange(n_frames * h_dim * w_dim * num_assets, dtype="float32").reshape(
                n_frames,
                h_dim,
                w_dim,
                num_assets,
            )
            frames_confidence = np.linspace(0.01, 0.99, num=n_frames * num_assets, dtype="float32").reshape(
                n_frames,
                num_assets,
            )
            frames_observed = np.ones((n_frames, num_assets), dtype="uint8")
            frames_ts = (np.arange(n_frames, dtype="int64") + 1000) * 10
            anchor_local_idx = np.asarray(anchors, dtype="int32")
            y_up = np.arange(n_samples, dtype="int64") % 4
            y_down = (np.arange(n_samples, dtype="int64") + 1) % 4
            anchor_ts = frames_ts[anchor_local_idx]
            duty_cycle = np.linspace(0.1, 1.0, n_samples, dtype="float32")
            aux = np.arange(max(1, n_samples * max(1, aux_dim)), dtype="float32")
            if aux_dim > 0:
                aux = aux[: n_samples * aux_dim].reshape(n_samples, aux_dim)
            else:
                aux = np.zeros((n_samples, 0), dtype="float32")

            files = {
                "frames_base": os.path.join(rel_dir, "frames_base.npy"),
                "frames_confidence": os.path.join(rel_dir, "frames_confidence.npy"),
                "frames_observed": os.path.join(rel_dir, "frames_observed.npy"),
                "frames_ts": os.path.join(rel_dir, "frames_ts.npy"),
                "anchor_local_idx": os.path.join(rel_dir, "anchor_local_idx.npy"),
                "aux": os.path.join(rel_dir, "aux.npy"),
                "y_up": os.path.join(rel_dir, "y_up.npy"),
                "y_down": os.path.join(rel_dir, "y_down.npy"),
                "anchor_ts": os.path.join(rel_dir, "anchor_ts.npy"),
                "duty_cycle": os.path.join(rel_dir, "duty_cycle.npy"),
            }

            np.save(os.path.join(tmp_dir, files["frames_base"]), frames_base, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["frames_confidence"]), frames_confidence, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["frames_observed"]), frames_observed, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["frames_ts"]), frames_ts, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["anchor_local_idx"]), anchor_local_idx, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["aux"]), aux, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["y_up"]), y_up, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["y_down"]), y_down, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["anchor_ts"]), anchor_ts, allow_pickle=False)
            np.save(os.path.join(tmp_dir, files["duty_cycle"]), duty_cycle, allow_pickle=False)

            base_channels = num_assets
            mask_channels = num_assets if include_mask else 0
            total_channels = base_channels + mask_channels + aux_dim

            dataset_frame = SnapshotDataset(
                snapshot_dir=tmp_dir,
                manifest={"chunks": []},
                chunks=[
                    SnapshotChunk(
                        start="2024-01-01 00:00:00",
                        end="2024-01-01 01:00:00",
                        file_path=os.path.join(tmp_dir, files["frames_base"]),
                        num_samples=n_samples,
                        start_index=0,
                        storage_format=CHUNK_STORAGE_FRAME_STORE_V1,
                        array_paths={k: os.path.join(tmp_dir, v) for k, v in files.items()},
                        x_shape=(n_samples, window_steps, h_dim, w_dim, total_channels),
                        window_steps=window_steps,
                        include_mask_channel=include_mask,
                        num_assets=num_assets,
                        aux_dim=aux_dim,
                    )
                ],
                total_samples=n_samples,
                config_hash="h-frame",
            )

            outputs = list(iter_snapshot_batches(dataset_frame, 0, n_samples))
            self.assertEqual(len(outputs), 1)
            x_out, y_up_out, y_down_out, anchor_out, duty_out = outputs[0]

            expected_x = np.zeros((n_samples, window_steps, h_dim, w_dim, total_channels), dtype="float32")
            for i, anchor_idx in enumerate(anchor_local_idx):
                start_idx = int(anchor_idx) - window_steps + 1
                end_idx = int(anchor_idx) + 1

                base_window = frames_base[start_idx:end_idx]
                if include_mask:
                    conf_window = frames_confidence[start_idx:end_idx]
                    mask_window = np.broadcast_to(conf_window[:, None, None, :], (window_steps, h_dim, w_dim, num_assets))
                    x_core = np.concatenate([base_window, mask_window], axis=-1)
                else:
                    x_core = base_window

                if aux_dim > 0:
                    aux_window = np.broadcast_to(aux[i][None, None, None, :], (window_steps, h_dim, w_dim, aux_dim))
                    x_core = np.concatenate([x_core, aux_window], axis=-1)

                expected_x[i] = x_core

            np.testing.assert_array_equal(x_out, expected_x)
            np.testing.assert_array_equal(y_up_out, y_up)
            np.testing.assert_array_equal(y_down_out, y_down)
            np.testing.assert_array_equal(anchor_out, anchor_ts)
            np.testing.assert_array_equal(duty_out, duty_cycle)


if __name__ == "__main__":
    unittest.main()
