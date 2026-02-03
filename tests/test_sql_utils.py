"""Tests for SQL sanitization utilities.

Tests cover:
- Datetime literal validation
- SQL identifier validation
- SQL injection attempt detection
- Query building functions
"""

import unittest

from data.sql_utils import (
    SQLValidationError,
    validate_datetime_literal,
    validate_identifier,
    build_order_book_query,
    build_connectivity_check_query,
)


class TestValidateDatetimeLiteral(unittest.TestCase):
    """Tests for validate_datetime_literal function."""

    def test_valid_date_format(self) -> None:
        """Accept YYYY-MM-DD format."""
        result = validate_datetime_literal("2025-01-15")
        self.assertEqual(result, "2025-01-15")

    def test_valid_datetime_format(self) -> None:
        """Accept YYYY-MM-DD HH:MM:SS format."""
        result = validate_datetime_literal("2025-01-15 10:30:45")
        self.assertEqual(result, "2025-01-15 10:30:45")

    def test_strips_whitespace(self) -> None:
        """Whitespace should be stripped."""
        result = validate_datetime_literal("  2025-01-15  ")
        self.assertEqual(result, "2025-01-15")

    def test_invalid_format_slash_separator(self) -> None:
        """Reject date with slash separators."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("2025/01/15")
        self.assertIn("Invalid", str(ctx.exception))

    def test_invalid_format_wrong_order(self) -> None:
        """Reject DD-MM-YYYY format."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("15-01-2025")
        self.assertIn("Invalid", str(ctx.exception))

    def test_sql_injection_attempt_single_quote(self) -> None:
        """Block SQL injection with single quote."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("2025-01-15'; DROP TABLE users; --")
        self.assertIn("Invalid", str(ctx.exception))

    def test_sql_injection_attempt_comment(self) -> None:
        """Block SQL injection with comment."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("2025-01-15 --")
        self.assertIn("Invalid", str(ctx.exception))

    def test_sql_injection_attempt_union(self) -> None:
        """Block SQL injection with UNION."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("2025-01-15 UNION SELECT * FROM secrets")
        self.assertIn("Invalid", str(ctx.exception))

    def test_non_string_raises(self) -> None:
        """Non-string input should raise."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal(12345)  # type: ignore[arg-type]
        self.assertIn("must be a string", str(ctx.exception))

    def test_custom_field_name_in_error(self) -> None:
        """Custom field name appears in error message."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_datetime_literal("bad", field_name="start_date")
        self.assertIn("start_date", str(ctx.exception))


class TestValidateIdentifier(unittest.TestCase):
    """Tests for validate_identifier function."""

    def test_valid_simple_identifier(self) -> None:
        """Accept simple alphanumeric identifier."""
        result = validate_identifier("orderbook_btcusdt")
        self.assertEqual(result, "orderbook_btcusdt")

    def test_valid_starts_with_underscore(self) -> None:
        """Accept identifier starting with underscore."""
        result = validate_identifier("_private_col")
        self.assertEqual(result, "_private_col")

    def test_valid_single_letter(self) -> None:
        """Accept single letter identifier."""
        result = validate_identifier("a")
        self.assertEqual(result, "a")

    def test_valid_with_numbers(self) -> None:
        """Accept identifier with numbers."""
        result = validate_identifier("col123")
        self.assertEqual(result, "col123")

    def test_invalid_starts_with_number(self) -> None:
        """Reject identifier starting with number."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("123table")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_invalid_contains_space(self) -> None:
        """Reject identifier with space."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("my table")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_invalid_contains_hyphen(self) -> None:
        """Reject identifier with hyphen."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("my-table")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_sql_injection_semicolon(self) -> None:
        """Block SQL injection with semicolon."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("table; DROP TABLE users")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_sql_injection_single_quote(self) -> None:
        """Block SQL injection with single quote."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("table'")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_sql_injection_double_dash(self) -> None:
        """Block SQL injection with comment."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("table--comment")
        self.assertIn("Invalid SQL", str(ctx.exception))

    def test_empty_string_raises(self) -> None:
        """Empty string should raise."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_non_string_raises(self) -> None:
        """Non-string input should raise."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier(123)  # type: ignore[arg-type]
        self.assertIn("must be a string", str(ctx.exception))

    def test_custom_field_name_in_error(self) -> None:
        """Custom field name appears in error message."""
        with self.assertRaises(SQLValidationError) as ctx:
            validate_identifier("bad;injection", field_name="table_name")
        self.assertIn("table_name", str(ctx.exception))


class TestBuildOrderBookQuery(unittest.TestCase):
    """Tests for build_order_book_query function."""

    def test_basic_query_structure(self) -> None:
        """Verify basic query structure."""
        sql = build_order_book_query(
            table_name="orderbook_btcusdt",
            columns=["ts", "bid_price", "ask_price"],
            timestamp_column="ts",
            start_datetime="2025-01-01",
            end_datetime="2025-01-02",
            end_inclusive=True,
            bid_price_column="bid_price",
            ask_price_column="ask_price",
        )
        self.assertIn("SELECT ts, bid_price, ask_price", sql)
        self.assertIn("FROM orderbook_btcusdt", sql)
        self.assertIn("WHERE ts >= '2025-01-01'", sql)
        self.assertIn("AND ts <= '2025-01-02'", sql)
        self.assertIn("AND bid_price > 0", sql)
        self.assertIn("AND ask_price > 0", sql)
        self.assertIn("ORDER BY ts ASC", sql)

    def test_exclusive_end_boundary(self) -> None:
        """Verify exclusive end uses < operator."""
        sql = build_order_book_query(
            table_name="test_table",
            columns=["col1"],
            timestamp_column="ts",
            start_datetime="2025-01-01",
            end_datetime="2025-01-02",
            end_inclusive=False,
            bid_price_column="bid",
            ask_price_column="ask",
        )
        self.assertIn("ts < '2025-01-02'", sql)
        self.assertNotIn("ts <= '2025-01-02'", sql)

    def test_inclusive_end_boundary(self) -> None:
        """Verify inclusive end uses <= operator."""
        sql = build_order_book_query(
            table_name="test_table",
            columns=["col1"],
            timestamp_column="ts",
            start_datetime="2025-01-01",
            end_datetime="2025-01-02",
            end_inclusive=True,
            bid_price_column="bid",
            ask_price_column="ask",
        )
        self.assertIn("ts <= '2025-01-02'", sql)

    def test_multiple_columns(self) -> None:
        """Verify multiple columns are joined correctly."""
        sql = build_order_book_query(
            table_name="test",
            columns=["a", "b", "c", "d"],
            timestamp_column="ts",
            start_datetime="2025-01-01",
            end_datetime="2025-01-02",
            end_inclusive=True,
            bid_price_column="bid",
            ask_price_column="ask",
        )
        self.assertIn("SELECT a, b, c, d", sql)

    def test_invalid_table_name_raises(self) -> None:
        """Invalid table name should raise SQLValidationError."""
        with self.assertRaises(SQLValidationError):
            build_order_book_query(
                table_name="table; DROP TABLE x",
                columns=["col"],
                timestamp_column="ts",
                start_datetime="2025-01-01",
                end_datetime="2025-01-02",
                end_inclusive=True,
                bid_price_column="bid",
                ask_price_column="ask",
            )

    def test_invalid_column_raises(self) -> None:
        """Invalid column name should raise SQLValidationError."""
        with self.assertRaises(SQLValidationError):
            build_order_book_query(
                table_name="test",
                columns=["valid_col", "invalid--col"],
                timestamp_column="ts",
                start_datetime="2025-01-01",
                end_datetime="2025-01-02",
                end_inclusive=True,
                bid_price_column="bid",
                ask_price_column="ask",
            )

    def test_invalid_datetime_raises(self) -> None:
        """Invalid datetime should raise SQLValidationError."""
        with self.assertRaises(SQLValidationError):
            build_order_book_query(
                table_name="test",
                columns=["col"],
                timestamp_column="ts",
                start_datetime="2025-01-01'; DROP TABLE x; --",
                end_datetime="2025-01-02",
                end_inclusive=True,
                bid_price_column="bid",
                ask_price_column="ask",
            )


class TestBuildConnectivityCheckQuery(unittest.TestCase):
    """Tests for build_connectivity_check_query function."""

    def test_basic_query(self) -> None:
        """Verify connectivity check query structure."""
        sql = build_connectivity_check_query("orderbook_btcusdt")
        self.assertEqual(sql, "SELECT 1 FROM orderbook_btcusdt LIMIT 1")

    def test_invalid_table_raises(self) -> None:
        """Invalid table name should raise SQLValidationError."""
        with self.assertRaises(SQLValidationError):
            build_connectivity_check_query("bad; DROP TABLE x")


if __name__ == "__main__":
    unittest.main()
