import unittest

import numpy as np

from preprocessing.snapshot_sequence_builder import (
    build_hybrid_depth_sequence_tensor,
    build_top_of_book_sequence_tensor,
)


def _build_config() -> dict:
    return {
        "data": {
            "time_range": {
                "cadence_seconds": 10,
            },
            "order_book": {
                "depth_levels": 4,
                "representation": "hybrid",
                "hybrid": {
                    "raw_levels": 2,
                    "aggregated_bins": 2,
                    "bin_strategy": "equal_width",
                },
            },
        },
        "targets": {
            "visible_window_seconds": 20,
        },
    }


class TestSnapshotSequenceBuilder(unittest.TestCase):
    def test_build_top_of_book_sequence_tensor_happy_path(self) -> None:
        config = _build_config()
        features = [
            [1.0, 10.0, 2.0, 20.0],
            [3.0, 30.0, 4.0, 40.0],
            [5.0, 50.0, 6.0, 60.0],
        ]
        x = build_top_of_book_sequence_tensor(
            config,
            snapshot_features=features,
            anchor_indices=[1, 2],
            sample_indices=[0, 1],
            height=2,
            width=2,
            channels=1,
        )

        self.assertEqual(x.shape, (2, 2, 2, 2, 1))
        self.assertEqual(float(x[0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(x[0, 0, 0, 1, 0]), 10.0)
        self.assertEqual(float(x[0, 0, 1, 0, 0]), 2.0)
        self.assertEqual(float(x[1, 1, 1, 1, 0]), 60.0)

    def test_build_top_of_book_sequence_tensor_rejects_invalid_window_multiple(self) -> None:
        config = _build_config()
        config["targets"]["visible_window_seconds"] = 25

        with self.assertRaises(ValueError):
            build_top_of_book_sequence_tensor(
                config,
                snapshot_features=[[1.0, 1.0, 1.0, 1.0]],
                anchor_indices=[0],
                sample_indices=[0],
                height=2,
                width=2,
                channels=1,
            )

    def test_build_hybrid_depth_sequence_tensor_empty_sample_indices(self) -> None:
        config = _build_config()
        depth_data = [
            {
                "bid_prices": np.array([100.0, 99.0, 98.0, 97.0]),
                "bid_quantities": np.array([1.0, 1.0, 1.0, 1.0]),
                "ask_prices": np.array([101.0, 102.0, 103.0, 104.0]),
                "ask_quantities": np.array([1.0, 1.0, 1.0, 1.0]),
            }
        ]

        x = build_hybrid_depth_sequence_tensor(
            config,
            snapshot_depth_data=depth_data,
            anchor_indices=[0],
            sample_indices=[],
        )

        self.assertEqual(x.shape, (0, 2, 4, 4, 1))

    def test_build_hybrid_depth_sequence_tensor_rejects_empty_depth_data(self) -> None:
        config = _build_config()
        with self.assertRaises(ValueError):
            build_hybrid_depth_sequence_tensor(
                config,
                snapshot_depth_data=[],
                anchor_indices=[0],
                sample_indices=[0],
            )


if __name__ == "__main__":
    unittest.main()
