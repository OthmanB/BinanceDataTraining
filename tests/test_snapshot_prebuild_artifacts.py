"""Tests for snapshot pre-build trial-invariant artifact warming."""

from __future__ import annotations

import types
import unittest
from unittest import mock

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

from training.snapshot_dataset import LabelDistribution, NormalizationStats
from training.pipeline import (
    _precompute_trial_invariant_snapshot_artifacts,
    _resolve_snapshot_training_indices,
    pre_build_snapshots,
)


class TestSnapshotPrebuildArtifacts(unittest.TestCase):
    def _base_config(self) -> dict:
        return {
            "run_mode": {"mode": "trial"},
            "snapshot": {
                "enabled": True,
                "directory": "snapshots",
                "root_name": "unit",
                "name": "run",
            },
            "data": {
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "cadence_seconds": 10,
                }
            },
            "preprocessing": {
                "train_test_split": {
                    "train_ratio": 0.7,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.1,
                },
                "normalization": {
                    "method": "standard",
                    "fit_on_train_only": False,
                },
            },
            "training": {
                "debug_max_samples": 50,
                "validation_split": 0.2,
                "class_weights": {
                    "compute_from_train": True,
                },
                "sequential_training": {
                    "enabled": False,
                    "window_days": 1,
                },
            },
            "model": {
                "output": {
                    "num_classes": 4,
                },
                "long_term": {
                    "enabled": False,
                },
            },
        }

    def test_precompute_trial_invariant_artifacts_warms_expected_caches(self) -> None:
        config = self._base_config()
        fake_dataset = types.SimpleNamespace(total_samples=100, chunks=[types.SimpleNamespace(file_path="x.npz")])
        fake_context = types.SimpleNamespace(snapshot_dir="/tmp/snapshot")
        fake_manifest = {}
        fake_norm = NormalizationStats(method="standard", mean=None, std=None)
        fake_label_dist = LabelDistribution(
            up_counts={0: 10, 1: 10, 2: 10, 3: 10},
            down_counts={0: 10, 1: 10, 2: 10, 3: 10},
            total_samples=40,
            num_classes=4,
        )

        with mock.patch("training.pipeline.resolve_snapshot_context", return_value=fake_context):
            with mock.patch("training.pipeline.load_or_create_manifest", return_value=fake_manifest):
                with mock.patch("training.pipeline._get_normalization_stats", return_value=fake_norm) as norm_mock:
                    with mock.patch("training.pipeline.load_label_stats_from_manifest", return_value=None):
                        with mock.patch(
                            "training.pipeline.compute_label_distribution",
                            return_value=fake_label_dist,
                        ) as label_dist_mock:
                            with mock.patch("training.pipeline.save_label_stats_to_manifest") as save_label_mock:
                                with mock.patch("training.pipeline.is_long_term_enabled", return_value=True):
                                    with mock.patch(
                                        "training.pipeline.compute_long_term_features_for_dataset",
                                        return_value=None,
                                    ) as lt_mock:
                                        _precompute_trial_invariant_snapshot_artifacts(config, fake_dataset)

        self.assertEqual(norm_mock.call_count, 2)
        first_call = norm_mock.call_args_list[0]
        second_call = norm_mock.call_args_list[1]
        self.assertEqual(first_call.args[4], 0)
        self.assertEqual(first_call.args[5], 50)
        self.assertEqual(first_call.args[6], "train")
        self.assertEqual(second_call.args[4], 70)
        self.assertEqual(second_call.args[5], 90)
        self.assertEqual(second_call.args[6], "val")

        label_dist_mock.assert_called_once_with(
            fake_dataset,
            start_index=0,
            end_index=50,
            num_classes=4,
        )
        save_label_mock.assert_called_once()
        lt_mock.assert_called_once_with(config, fake_dataset, cadence_seconds=10)

    def test_precompute_uses_cached_label_stats_and_train_only_normalization(self) -> None:
        config = self._base_config()
        config["preprocessing"]["normalization"]["fit_on_train_only"] = True
        config["training"]["class_weights"]["compute_from_train"] = True

        fake_dataset = types.SimpleNamespace(total_samples=100, chunks=[types.SimpleNamespace(file_path="x.npz")])
        fake_context = types.SimpleNamespace(snapshot_dir="/tmp/snapshot")
        fake_manifest = {}
        fake_norm = NormalizationStats(method="standard", mean=None, std=None)
        cached_label_dist = LabelDistribution(
            up_counts={0: 10, 1: 10, 2: 10, 3: 10},
            down_counts={0: 10, 1: 10, 2: 10, 3: 10},
            total_samples=40,
            num_classes=4,
        )

        with mock.patch("training.pipeline.resolve_snapshot_context", return_value=fake_context):
            with mock.patch("training.pipeline.load_or_create_manifest", return_value=fake_manifest):
                with mock.patch("training.pipeline._get_normalization_stats", return_value=fake_norm) as norm_mock:
                    with mock.patch(
                        "training.pipeline.load_label_stats_from_manifest",
                        return_value=cached_label_dist,
                    ):
                        with mock.patch("training.pipeline.compute_label_distribution") as label_dist_mock:
                            with mock.patch("training.pipeline.save_label_stats_to_manifest") as save_label_mock:
                                with mock.patch("training.pipeline.is_long_term_enabled", return_value=False):
                                    with mock.patch("training.pipeline.compute_long_term_features_for_dataset") as lt_mock:
                                        _precompute_trial_invariant_snapshot_artifacts(config, fake_dataset)

        self.assertEqual(norm_mock.call_count, 1)
        label_dist_mock.assert_not_called()
        save_label_mock.assert_not_called()
        lt_mock.assert_not_called()

    def test_pre_build_snapshots_calls_precompute_for_each_sequential_window(self) -> None:
        config = self._base_config()
        fake_dataset_a = types.SimpleNamespace(total_samples=10, chunks=[types.SimpleNamespace(file_path="a.npz")])
        fake_dataset_b = types.SimpleNamespace(total_samples=20, chunks=[types.SimpleNamespace(file_path="b.npz")])

        windows = [("2024-01-01", "2024-01-01"), ("2024-01-02", "2024-01-02")]
        with mock.patch("training.pipeline._resolve_sequential_windows", return_value=windows):
            with mock.patch(
                "training.pipeline.prepare_snapshot_dataset",
                side_effect=[fake_dataset_a, fake_dataset_b],
            ) as prepare_mock:
                with mock.patch("training.pipeline._precompute_trial_invariant_snapshot_artifacts") as precompute_mock:
                    pre_build_snapshots(config)

        self.assertEqual(prepare_mock.call_count, 2)
        self.assertEqual(precompute_mock.call_count, 2)
        self.assertEqual(precompute_mock.call_args_list[0].args[1], fake_dataset_a)
        self.assertEqual(precompute_mock.call_args_list[1].args[1], fake_dataset_b)


@unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestSnapshotPrebuildProperties(unittest.TestCase):
    @settings(max_examples=50)
    @given(
        n_samples=st.integers(min_value=1, max_value=5000),
        debug_max_samples=st.integers(min_value=0, max_value=5000),
        train_units=st.integers(min_value=0, max_value=10),
        val_units=st.integers(min_value=0, max_value=10),
        test_units=st.integers(min_value=1, max_value=10),
    )
    def test_resolve_snapshot_training_indices_invariants(
        self,
        n_samples: int,
        debug_max_samples: int,
        train_units: int,
        val_units: int,
        test_units: int,
    ) -> None:
        total = train_units + val_units + test_units
        train_ratio = float(train_units / total)
        val_ratio = float(val_units / total)
        test_ratio = float(test_units / total)

        config = {
            "training": {
                "debug_max_samples": debug_max_samples,
                "validation_split": val_ratio,
            },
            "preprocessing": {
                "train_test_split": {
                    "train_ratio": train_ratio,
                    "validation_ratio": val_ratio,
                    "test_ratio": test_ratio,
                }
            },
        }

        result = _resolve_snapshot_training_indices(config, n_samples)

        train_end = int(n_samples * train_ratio)
        expected_effective = min(train_end, debug_max_samples)
        expected_val_end = min(train_end + int(n_samples * val_ratio), n_samples)

        if train_end <= 0 or expected_effective <= 0:
            self.assertIsNone(result)
            return

        self.assertIsNotNone(result)
        assert result is not None
        effective_train_n, val_start, val_end = result
        self.assertEqual(effective_train_n, expected_effective)
        self.assertEqual(val_start, train_end)
        self.assertEqual(val_end, expected_val_end)
        self.assertGreater(effective_train_n, 0)
        self.assertLessEqual(effective_train_n, n_samples)
        self.assertLessEqual(val_start, val_end)
        self.assertLessEqual(val_end, n_samples)


if __name__ == "__main__":
    unittest.main()
