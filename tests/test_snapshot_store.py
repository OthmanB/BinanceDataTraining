import json
import os
import tempfile
import unittest

from training.snapshot_store import compute_config_hash, maybe_evict_snapshots, resolve_snapshot_context


def _build_base_config(snapshot_dir: str) -> dict:
    return {
        "snapshot": {
            "enabled": True,
            "directory": snapshot_dir,
            "root_name": "baseline",
            "name": "auto",
            "on_config_mismatch": "create_new",
            "max_snapshots": 2,
        },
        "data": {
            "asset_pairs": {
                "target_asset": "BTCUSDT",
                "correlated_assets": [],
            },
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "cadence_seconds": 10,
            },
            "order_book": {
                "depth_levels": 10,
                "representation": "hybrid",
                "hybrid": {"raw_levels": 5, "aggregated_bins": 5},
            },
            "temporal_features": {"local": [], "global": []},
        },
        "targets": {
            "prediction_horizon_seconds": 60,
            "visible_window_seconds": 30,
            "price_classes": {"definition_type": "percentage", "boundaries": [1.0]},
            "labeling": {
                "scheme": "two_head_intensity",
                "use_midpoint": True,
                "handle_gaps": "skip",
            },
        },
        "preprocessing": {
            "normalization": {
                "method": "min_max",
                "per_asset": False,
                "fit_on_train_only": True,
            },
            "feature_engineering": {"enabled": False},
        },
        "model": {
            "architecture": "CNN_LSTM_MultiClass",
            "input_representation": {
                "temporal_features": {
                    "integration_mode": "none",
                    "use_local_features": False,
                    "use_global_features": False,
                }
            },
            "cnn": {"kernel_sizes": [[3, 3]], "pool_sizes": [[2, 2]]},
            "output": {"type": "two_head_intensity", "num_classes": 2},
        },
    }


class TestSnapshotStore(unittest.TestCase):
    def test_compute_config_hash_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            first = compute_config_hash(config)
            second = compute_config_hash(config)
            self.assertEqual(first, second)

            config["data"]["order_book"]["depth_levels"] = 12
            changed = compute_config_hash(config)
            self.assertNotEqual(first, changed)

    def test_resolve_snapshot_context_auto_name_uses_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            context = resolve_snapshot_context(config)

            self.assertTrue(os.path.exists(context.snapshot_dir))
            self.assertTrue(context.snapshot_name.startswith("baseline_"))

    def test_resolve_snapshot_context_mismatch_creates_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            context = resolve_snapshot_context(config)

            os.makedirs(context.snapshot_dir, exist_ok=True)
            with open(context.manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"config_hash": "different"}, handle)

            new_context = resolve_snapshot_context(config)
            self.assertNotEqual(context.snapshot_dir, new_context.snapshot_dir)

    def test_maybe_evict_snapshots_disabled_when_max_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_base_config(tmp_dir)
            context = resolve_snapshot_context(config)

            root_dir = os.path.dirname(context.snapshot_dir)
            snapshot_a = os.path.join(root_dir, "baseline_a")
            snapshot_b = os.path.join(root_dir, "baseline_b")
            os.makedirs(snapshot_a, exist_ok=True)
            os.makedirs(snapshot_b, exist_ok=True)

            for path in (snapshot_a, snapshot_b):
                manifest_path = os.path.join(path, "manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump({"config_hash": "hash", "created_at": "2025-01-01T00:00:00Z"}, handle)

            maybe_evict_snapshots(context, max_snapshots=0)

            self.assertTrue(os.path.exists(snapshot_a))
            self.assertTrue(os.path.exists(snapshot_b))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
