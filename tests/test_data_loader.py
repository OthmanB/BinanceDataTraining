import unittest
from unittest.mock import patch

from data.data_loader import (
    _get_asset_pairs_from_config,
    create_empty_data_object_from_config,
    load_order_book_data,
)


def _build_base_config() -> dict:
    return {
        "data": {
            "asset_pairs": {
                "target_asset": "BTCUSDT",
                "correlated_assets": ["ETHUSDT", "SOLUSDT"],
            },
            "time_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "cadence_seconds": 10,
            },
            "order_book": {
                "depth_levels": 20,
            },
        },
        "targets": {
            "prediction_horizon_seconds": 60,
        },
    }


class TestDataLoader(unittest.TestCase):
    def test_get_asset_pairs_from_config_includes_target_and_correlated(self) -> None:
        config = _build_base_config()
        assets = _get_asset_pairs_from_config(config)
        self.assertEqual(assets, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    def test_get_asset_pairs_from_config_skips_none_target(self) -> None:
        config = _build_base_config()
        config["data"]["asset_pairs"]["target_asset"] = None
        assets = _get_asset_pairs_from_config(config)
        self.assertEqual(assets, ["ETHUSDT", "SOLUSDT"])

    def test_create_empty_data_object_from_config_populates_metadata(self) -> None:
        config = _build_base_config()
        data_object = create_empty_data_object_from_config(config)

        metadata = data_object["metadata"]
        self.assertEqual(metadata["asset_pairs"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(metadata["num_samples"], 0)
        self.assertEqual(metadata["order_book_depth"], 20)

        targets = data_object["targets"]
        self.assertEqual(targets["asset"], "BTCUSDT")
        self.assertEqual(targets["delta_t_seconds"], 60)

    def test_create_empty_data_object_from_config_missing_data_raises(self) -> None:
        with self.assertRaises(KeyError):
            create_empty_data_object_from_config({})

    @patch("data.data_loader.fetch_order_book_rows")
    @patch("data.data_loader.check_greptime_connectivity")
    def test_load_order_book_data_sets_num_samples_for_target(
        self,
        mock_connectivity,
        mock_fetch_rows,
    ) -> None:
        config = _build_base_config()
        mock_fetch_rows.return_value = {
            "BTCUSDT": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "ETHUSDT": [[10, 11, 12]],
        }

        data_object = load_order_book_data(config)

        mock_connectivity.assert_called_once_with(config)
        mock_fetch_rows.assert_called_once_with(config)
        self.assertEqual(data_object["metadata"]["num_samples"], 3)
        self.assertEqual(sorted(data_object["order_books"].keys()), ["BTCUSDT", "ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
