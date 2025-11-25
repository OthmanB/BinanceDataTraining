import unittest

import numpy as np

from preprocessing.temporal_features import attach_temporal_features


class TestTemporalFeatures(unittest.TestCase):
    def _build_base_config(self) -> dict:
        return {
            "data": {
                "asset_pairs": {"target_asset": "BTCUSDT"},
                "temporal_features": {
                    "local": [
                        "hour_of_day",
                        "day_of_week",
                        "minute_of_hour",
                    ],
                    "global": [
                        "days_since_start",
                        "market_session",
                    ],
                    "market_session": {
                        "utc_offset_hours": 0,
                        "sessions": [
                            {"name": "asian", "start_hour": 0, "end_hour": 8},
                            {"name": "european", "start_hour": 8, "end_hour": 16},
                            {"name": "american", "start_hour": 16, "end_hour": 24},
                        ],
                    },
                },
            },
        }

    def test_attach_temporal_features_shapes_and_values(self) -> None:
        config = self._build_base_config()

        snapshot_timestamps = np.array(
            [
                "2024-01-01T01:00:00",  # asian
                "2024-01-02T09:00:00",  # european
                "2024-01-03T17:00:00",  # american
            ],
            dtype="datetime64[ns]",
        )

        data_object = {
            "metadata": {
                "num_samples": 3,
                "anchor_indices": [0, 1, 2],
            },
            "order_books": {
                "BTCUSDT": {
                    "snapshot_timestamps": snapshot_timestamps,
                },
            },
            "temporal_features": {},
            "targets": {},
            "external_data": {},
        }

        updated = attach_temporal_features(config, data_object)
        tf = updated["temporal_features"]

        self.assertIn("local", tf)
        self.assertIn("global", tf)

        local = tf["local"]
        global_ = tf["global"]

        self.assertEqual(local.shape, (3, 6))
        self.assertEqual(global_.shape, (3, 4))

        days_since_start = global_[:, 0]
        np.testing.assert_array_equal(days_since_start, np.array([0.0, 1.0, 2.0], dtype="float32"))

        sessions = global_[:, 1:]
        expected_sessions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(sessions, expected_sessions)

    def test_attach_temporal_features_skips_when_no_samples(self) -> None:
        config = self._build_base_config()

        data_object = {
            "metadata": {
                "num_samples": 0,
                "anchor_indices": [],
            },
            "order_books": {},
            "temporal_features": {},
            "targets": {},
            "external_data": {},
        }

        # Should return without raising and without modifying temporal_features.
        updated = attach_temporal_features(config, data_object)
        self.assertIs(updated, data_object)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
