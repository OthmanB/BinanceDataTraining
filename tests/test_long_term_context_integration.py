"""Integration tests for long-term context data loading and caching."""

import os
import tempfile
import unittest

import numpy as np

from training.long_term_context import compute_long_term_features_for_dataset
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


if __name__ == "__main__":
    unittest.main()
