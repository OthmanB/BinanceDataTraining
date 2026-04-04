import unittest
from unittest import mock
from typing import Any

import data.greptime_client as greptime_client
from data.greptime_client import fetch_order_book_rows, stream_order_book_chunks


class TestGreptimeClient(unittest.TestCase):
    def _build_base_schema(self) -> dict[str, Any]:
        return {
            "timestamp_column": "ts",
            "bid_price_column": "bid_price",
            "bid_quantity_column": "bid_quantity",
            "ask_price_column": "ask_price",
            "ask_quantity_column": "ask_quantity",
            "batch_id_column": "batch_id",
        }

    @mock.patch("requests.sessions.Session.post")
    def test_fetch_order_book_rows_multi_database_time_split(self, mock_post) -> None:
        def fake_post(url, data=None, headers=None, timeout=None):  # type: ignore[override]
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
                "ingestion": {
                    "chunk_hours": 240,  # 10 days - larger than date range to avoid chunking in test
                    "chunk_delay_seconds": 0.0,
                    "max_concurrent_chunk_fetches": 1,
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
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
                            "time_range": {
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-05",
                            },
                        },
                        {
                            "name": "recent",
                            "database_uri": "http://db2",
                            "table_prefix": "orderbook_",
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
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
                "ingestion": {
                    "chunk_hours": 24,
                    "chunk_delay_seconds": 0.0,
                    "max_concurrent_chunk_fetches": 1,
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
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
                            "time_range": {
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-08",
                            },
                        },
                        {
                            "name": "db2",
                            "database_uri": "http://db2",
                            "table_prefix": "orderbook_",
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
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

    def test_fetch_order_book_rows_touching_ranges_raise(self) -> None:
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
                "ingestion": {
                    "chunk_hours": 24,
                    "chunk_delay_seconds": 0.0,
                    "max_concurrent_chunk_fetches": 1,
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
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
                            "time_range": {
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-08",
                            },
                        },
                        {
                            "name": "db2",
                            "database_uri": "http://db2",
                            "table_prefix": "orderbook_",
                            "request_timeout_seconds": 30,
                            "connect_timeout_seconds": 10,
                            "max_retries": 3,
                            "retry_backoff_factor": 0.5,
                            "time_range": {
                                "start_date": "2024-01-08",
                                "end_date": "2024-01-10",
                            },
                        },
                    ],
                },
            },
        }

        with self.assertRaises(ValueError):
            fetch_order_book_rows(config)

    def test_stream_order_book_chunks_rejects_concurrent(self) -> None:
        config = {
            "data": {
                "asset_pairs": {
                    "target_asset": "BTCUSDT",
                    "correlated_assets": [],
                },
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                },
                "ingestion": {
                    "chunk_hours": 24,
                    "chunk_delay_seconds": 0.0,
                    "max_concurrent_chunk_fetches": 2,
                },
                "order_book": {
                    "schema": self._build_base_schema(),
                },
                "connection": {
                    "database_uri": "http://db",
                    "table_prefix": "orderbook_",
                    "request_timeout_seconds": 30,
                    "connect_timeout_seconds": 10,
                    "max_retries": 3,
                    "retry_backoff_factor": 0.5,
                },
                "multi_database": {
                    "enabled": False,
                },
            },
        }

        with self.assertRaises(ValueError):
            next(stream_order_book_chunks(config))

    def test_retry_session_uses_configured_budget(self) -> None:
        timeout_cfg = {
            "request_timeout_seconds": 30,
            "connect_timeout_seconds": 10,
            "max_retries": 3,
            "retry_backoff_factor": 0.5,
        }

        session = greptime_client._build_retry_session(timeout_cfg)
        self.addCleanup(session.close)

        adapter: Any = session.get_adapter("http://")
        retry = adapter.__dict__.get("max_retries")
        self.assertEqual(retry.total, 3)
        self.assertEqual(retry.connect, 3)
        self.assertEqual(retry.read, 3)
        self.assertEqual(retry.status, 3)
        self.assertEqual(retry.backoff_factor, 0.5)
        self.assertEqual(retry.backoff_max, 3.0)
        self.assertIn("POST", retry.allowed_methods)

    def test_fetch_chunk_uses_retry_session_and_configured_request_timeout(self) -> None:
        timeout_cfg = {
            "request_timeout_seconds": 30,
            "connect_timeout_seconds": 10,
            "max_retries": 3,
            "retry_backoff_factor": 0.5,
        }
        schema_cfg = self._build_base_schema()

        response = mock.Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = {
            "output": [{"records": {"rows": [["row1"]]}}],
            "execution_time_ms": 1,
        }

        session = mock.Mock()
        session.post.return_value = response

        with mock.patch("data.greptime_client._build_retry_session", return_value=session) as build_retry_session:
            rows = greptime_client._fetch_order_book_rows_for_chunk(
                base_uri="http://db",
                table_prefix="orderbook_",
                asset="BTCUSDT",
                chunk_start="2024-01-01 00:00:00",
                chunk_end="2024-01-01 23:59:59",
                schema_cfg=schema_cfg,
                timeout_cfg=timeout_cfg,
                end_inclusive=True,
            )

        self.assertEqual(rows, [["row1"]])
        build_retry_session.assert_called_once_with(timeout_cfg)
        session.post.assert_called_once()
        self.assertEqual(session.post.call_args.kwargs["timeout"], (10.0, 30.0))
        session.close.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
