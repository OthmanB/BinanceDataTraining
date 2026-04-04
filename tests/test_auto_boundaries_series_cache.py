"""Unit tests for auto boundary fitting from series cache.

These tests avoid network calls by monkeypatching prepare_series_dataset.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from training.auto_boundaries import fit_price_class_boundaries_from_series_cache
from utils.config_loader import ConfigError


def _make_min_config(*, cadence_seconds: int = 10) -> dict:
    return {
        "snapshot": {
            "enabled": True,
            "directory": "/tmp/snapshots",
            "root_name": "baseline",
            "name": "auto",
            "on_config_mismatch": "create_new",
            "max_snapshots": 1,
        },
        "data": {
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "cadence_seconds": cadence_seconds,
            }
        },
        "preprocessing": {
            "train_test_split": {
                "method": "chronological",
                "train_ratio": 0.5,
            }
        },
        "targets": {
            "visible_window_seconds": 30,
            "prediction_horizon_seconds": 20,
        },
    }


class TestAutoBoundariesSeriesCache(unittest.TestCase):
    def test_fit_boundaries_from_series_cache_smoke(self) -> None:
        # Create synthetic mid_prices with varying moves.
        mids = np.linspace(100.0, 120.0, num=500, dtype="float64")
        ts = np.arange(mids.shape[0], dtype="int64")
        vols = np.zeros_like(mids)

        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = os.path.join(tmp, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(chunks_dir, "chunk_0.npz"),
                timestamps=ts,
                mid_prices=mids,
                volumes=vols,
            )

            cfg = _make_min_config()

            fake_dataset = SimpleNamespace(series_dir=tmp)
            with mock.patch("training.auto_boundaries.prepare_series_dataset", return_value=fake_dataset):
                boundaries = fit_price_class_boundaries_from_series_cache(
                    cfg,
                    num_classes=4,
                    fit_on="full",
                    labeling_criteria="max_intensity",
                    max_samples=1000,
                    random_seed=0,
                )

        self.assertEqual(len(boundaries), 3)
        self.assertTrue(all(float(b) > 0.0 for b in boundaries))
        self.assertTrue(all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1)))

    def test_fit_boundaries_requires_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "chunks"), exist_ok=True)
            cfg = _make_min_config()
            fake_dataset = SimpleNamespace(series_dir=tmp)
            with mock.patch("training.auto_boundaries.prepare_series_dataset", return_value=fake_dataset):
                with self.assertRaises(ConfigError):
                    fit_price_class_boundaries_from_series_cache(
                        cfg,
                        num_classes=3,
                        fit_on="full",
                        labeling_criteria="max_intensity",
                        max_samples=1000,
                        random_seed=0,
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
