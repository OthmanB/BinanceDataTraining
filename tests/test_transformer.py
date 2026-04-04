import unittest

import numpy as np

from preprocessing.transformer import (
    _apply_gap_handling_to_snapshots,
    _validate_snapshot_timestamps_for_target_asset,
    run_preprocessing_pipeline,
)


class TestTransformer(unittest.TestCase):
    def test_run_preprocessing_pipeline_skips_target_build_when_no_samples(self) -> None:
        data_object = {
            "metadata": {"num_samples": 0},
            "order_books": {},
            "temporal_features": {},
            "targets": {},
            "external_data": {},
        }

        out = run_preprocessing_pipeline({}, data_object)
        self.assertIs(out, data_object)
        self.assertEqual(out["metadata"]["num_samples"], 0)

    def test_validate_snapshot_timestamps_for_target_asset_raises_on_non_monotonic(self) -> None:
        data_cfg = {
            "validation": {
                "check_missing_data": True,
                "max_gap_seconds": 60,
                "fail_on_invalid": True,
            }
        }

        with self.assertRaises(ValueError) as exc_info:
            _validate_snapshot_timestamps_for_target_asset(
                data_cfg,
                "BTCUSDT",
                [
                    np.datetime64("2024-01-01T00:00:10"),
                    np.datetime64("2024-01-01T00:00:00"),
                ],
            )

        self.assertIn("Non-monotonic", str(exc_info.exception))

    def test_apply_gap_handling_forward_fill_inserts_missing_steps(self) -> None:
        data_cfg = {
            "time_range": {"cadence_seconds": 10},
            "validation": {"max_gap_seconds": 40},
        }
        targets_cfg = {
            "labeling": {"handle_gaps": "forward_fill"},
        }
        timestamps = [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-01T00:00:30"),
        ]
        features = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
        mids = [100.0, 130.0]

        new_ts, new_features, new_mid = _apply_gap_handling_to_snapshots(
            data_cfg,
            targets_cfg,
            timestamps,
            features,
            mids,
        )

        self.assertEqual(len(new_ts), 4)
        self.assertEqual(len(new_features), 4)
        self.assertEqual(len(new_mid), 4)
        self.assertEqual(new_mid[1], 100.0)
        self.assertEqual(new_mid[2], 100.0)
        self.assertEqual(new_mid[3], 130.0)


if __name__ == "__main__":
    unittest.main()
