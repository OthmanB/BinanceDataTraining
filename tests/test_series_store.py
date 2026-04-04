import json
import os
import tempfile
import unittest

from training.series_store import (
    compute_series_config_hash,
    load_or_create_series_manifest,
    maybe_evict_series_caches,
    resolve_series_context,
)
from utils.config_loader import ConfigError


def _build_base_config(snapshot_dir: str) -> dict:
    return {
        "snapshot": {
            "directory": snapshot_dir,
            "root_name": "baseline",
            "name": "auto",
            "on_config_mismatch": "create_new",
        },
        "data": {
            "asset_pairs": {
                "target_asset": "BTCUSDT",
                "correlated_assets": ["ETHUSDT"],
                "alignment": {
                    "method": "interpolate",
                    "missing_policy": "forward_fill",
                    "max_gap_seconds": 120,
                    "bucket_tolerance_seconds": 0.0,
                    "include_mask_channel": False,
                },
            },
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "cadence_seconds": 10,
            },
            "ingestion": {
                "chunk_hours": 24,
            },
            "order_book": {
                "depth_levels": 10,
                "representation": "hybrid",
                "hybrid": {
                    "raw_levels": 5,
                    "aggregated_bins": 5,
                },
            },
        },
        "targets": {
            "labeling": {
                "scheme": "two_head_intensity",
                "use_midpoint": True,
                "handle_gaps": "skip",
            },
            "price_classes": {
                "definition_type": "percentage",
                "boundaries": [0.1, 0.2],
            },
        },
        "preprocessing": {
            "feature_engineering": {
                "enabled": True,
                "momentum_window_seconds": 60,
            },
        },
    }


class TestSeriesStore(unittest.TestCase):
    def test_compute_series_config_hash_ignores_price_class_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            baseline = compute_series_config_hash(config)

            config["targets"]["price_classes"]["boundaries"] = [0.5, 1.0, 2.0]
            changed_boundaries = compute_series_config_hash(config)
            self.assertEqual(baseline, changed_boundaries)

            config["targets"]["labeling"]["handle_gaps"] = "forward_fill"
            changed_labeling = compute_series_config_hash(config)
            self.assertNotEqual(baseline, changed_labeling)

    def test_resolve_series_context_mismatch_raises_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            config["snapshot"]["on_config_mismatch"] = "error"

            context = resolve_series_context(config)
            os.makedirs(context.series_dir, exist_ok=True)
            with open(context.manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"config_hash": "different"}, handle)

            with self.assertRaises(ConfigError):
                resolve_series_context(config)

    def test_maybe_evict_series_caches_removes_oldest_non_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            context = resolve_series_context(config)
            load_or_create_series_manifest(context, config)

            root_dir = os.path.dirname(context.series_dir)
            old_a = os.path.join(root_dir, f"{context.root_name}_old_a")
            old_b = os.path.join(root_dir, f"{context.root_name}_old_b")

            for path, created_at in (
                (old_a, "2020-01-01T00:00:00+00:00"),
                (old_b, "2021-01-01T00:00:00+00:00"),
            ):
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as handle:
                    json.dump({"created_at": created_at, "config_hash": "x"}, handle)

            maybe_evict_series_caches(context, max_caches=2)

            self.assertFalse(os.path.exists(old_a))
            self.assertTrue(os.path.exists(old_b))
            self.assertTrue(os.path.exists(context.series_dir))


if __name__ == "__main__":
    unittest.main()
