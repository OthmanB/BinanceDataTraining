import unittest
from unittest import mock

from data.greptime_client import fetch_order_book_rows


class TestGreptimeClient(unittest.TestCase):
    def _build_base_schema(self) -> dict:
        return {
            "timestamp_column": "ts",
            "bid_price_column": "bid_price",
            "bid_quantity_column": "bid_quantity",
            "ask_price_column": "ask_price",
            "ask_quantity_column": "ask_quantity",
            "batch_id_column": "batch_id",
        }

    @mock.patch("data.greptime_client.requests.post")
    def test_fetch_order_book_rows_multi_database_time_split(self, mock_post) -> None:
        def fake_post(url, data=None, headers=None):  # type: ignore[override]
            if "db1" in url:
                rows = [["db1_row1"], ["db1_row2"]]
            else:
                rows = [["db2_row1"], ["db2_row2"]]

            payload = {
                "output": [
                    {"records": {"rows": rows}},
                ],
                "execution_time_ms": 1,
            }

            class _Resp:
                def __init__(self, p):
                    self.ok = True
                    self._payload = p
                    self.status_code = 200
                    self.text = "OK"

                def json(self):
                    return self._payload

            return _Resp(payload)

        mock_post.side_effect = fake_post

        config = {
            "data": {
                "asset_pairs": {
                    "target_asset": "BTCUSDT",
                    "correlated_assets": [],
                },
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                },
                "order_book": {
                    "schema": self._build_base_schema(),
                },
                "multi_database": {
                    "enabled": True,
                    "strategy": "time_split",
                    "connections": [
                        {
                            "name": "historical",
                            "database_uri": "http://db1",
                            "table_prefix": "orderbook_",
                            "time_range": {
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-05",
                            },
                        },
                        {
                            "name": "recent",
                            "database_uri": "http://db2",
                            "table_prefix": "orderbook_",
                            "time_range": {
                                "start_date": "2024-01-06",
                                "end_date": "2024-01-10",
                            },
                        },
                    ],
                },
            },
        }

        rows_by_asset = fetch_order_book_rows(config)
        self.assertIn("BTCUSDT", rows_by_asset)
        rows = rows_by_asset["BTCUSDT"]

        self.assertEqual(
            rows,
            [
                ["db1_row1"],
                ["db1_row2"],
                ["db2_row1"],
                ["db2_row2"],
            ],
        )

    def test_fetch_order_book_rows_overlapping_ranges_raise(self) -> None:
        config = {
            "data": {
                "asset_pairs": {
                    "target_asset": "BTCUSDT",
                    "correlated_assets": [],
                },
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                },
                "order_book": {
                    "schema": self._build_base_schema(),
                },
                "multi_database": {
                    "enabled": True,
                    "strategy": "time_split",
                    "connections": [
                        {
                            "name": "db1",
                            "database_uri": "http://db1",
                            "table_prefix": "orderbook_",
                            "time_range": {
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-08",
                            },
                        },
                        {
                            "name": "db2",
                            "database_uri": "http://db2",
                            "table_prefix": "orderbook_",
                            "time_range": {
                                "start_date": "2024-01-07",
                                "end_date": "2024-01-10",
                            },
                        },
                    ],
                },
            },
        }

        with self.assertRaises(ValueError):
            fetch_order_book_rows(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
