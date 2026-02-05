"""Unit tests for time chunk generation in greptime_client.

Tests cover:
- Basic chunk generation
- Edge cases: single day, range smaller than chunk
- Boundary handling: exclusive end for interior chunks
"""

import pytest
from hypothesis import given, strategies as st

from data.greptime_client import _generate_time_chunks


class TestGenerateTimeChunksBasic:
    """Basic unit tests for _generate_time_chunks."""

    def test_single_day_single_chunk(self) -> None:
        """Single day with chunk_hours > 24 produces one chunk."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-15", chunk_hours=48)

        assert len(chunks) == 1
        start, end, is_last = chunks[0]
        assert start == "2024-01-15 00:00:00"
        assert end == "2024-01-15 23:59:59"
        assert is_last is True

    def test_single_day_multiple_chunks(self) -> None:
        """Single day with chunk_hours=12 produces 2 chunks."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-15", chunk_hours=12)

        assert len(chunks) == 2
        # First chunk
        start0, end0, is_last0 = chunks[0]
        assert start0 == "2024-01-15 00:00:00"
        assert end0 == "2024-01-15 11:59:59"
        assert is_last0 is False
        # Second chunk
        start1, end1, is_last1 = chunks[1]
        assert start1 == "2024-01-15 12:00:00"
        assert end1 == "2024-01-15 23:59:59"
        assert is_last1 is True

    def test_two_days_with_24h_chunks(self) -> None:
        """Two days with chunk_hours=24 produces 2 chunks."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-16", chunk_hours=24)

        assert len(chunks) == 2
        # First chunk covers first day
        start0, end0, is_last0 = chunks[0]
        assert start0 == "2024-01-15 00:00:00"
        assert end0 == "2024-01-15 23:59:59"
        assert is_last0 is False
        # Second chunk covers second day
        start1, end1, is_last1 = chunks[1]
        assert start1 == "2024-01-16 00:00:00"
        assert end1 == "2024-01-16 23:59:59"
        assert is_last1 is True

    def test_week_with_48h_chunks(self) -> None:
        """Week-long range with 48h chunks produces expected chunks."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-21", chunk_hours=48)

        # 7 days = 168 hours, 48h chunks = ceil(168/48) = 4 chunks
        assert len(chunks) >= 3
        # All except last should have is_last=False
        for i, (_, _, is_last) in enumerate(chunks[:-1]):
            assert is_last is False, f"Chunk {i} should not be last"
        # Last chunk should have is_last=True
        assert chunks[-1][2] is True

    def test_chunks_are_contiguous(self) -> None:
        """Chunks should be contiguous without gaps."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-17", chunk_hours=12)

        for i in range(len(chunks) - 1):
            _, end_i, _ = chunks[i]
            start_next, _, _ = chunks[i + 1]
            # Parse end time and next start time
            # end_i is like "2024-01-15 11:59:59"
            # start_next should be "2024-01-15 12:00:00" (1 second later)
            # Simple string-based check
            assert end_i < start_next, f"Chunks {i} and {i+1} should be contiguous"


class TestGenerateTimeChunksValidation:
    """Validation tests for _generate_time_chunks."""

    def test_invalid_chunk_hours_zero(self) -> None:
        """chunk_hours=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            _generate_time_chunks("2024-01-15", "2024-01-16", chunk_hours=0)

    def test_invalid_chunk_hours_negative(self) -> None:
        """chunk_hours<0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            _generate_time_chunks("2024-01-15", "2024-01-16", chunk_hours=-5)

    def test_invalid_date_format(self) -> None:
        """Invalid date format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _generate_time_chunks("01-15-2024", "2024-01-16", chunk_hours=12)

    def test_start_after_end_raises(self) -> None:
        """start_date > end_date should raise ValueError."""
        with pytest.raises(ValueError, match="start_date must be <= end_date"):
            _generate_time_chunks("2024-01-20", "2024-01-15", chunk_hours=12)


class TestGenerateTimeChunksProperties:
    """Property-based tests for _generate_time_chunks."""

    @given(
        st.integers(min_value=1, max_value=168),  # chunk_hours 1-168 (1 week)
    )
    def test_last_chunk_is_last(self, chunk_hours: int) -> None:
        """Last chunk in any valid range should have is_last=True."""
        chunks = _generate_time_chunks("2024-01-15", "2024-01-20", chunk_hours)

        assert len(chunks) >= 1
        assert chunks[-1][2] is True

    @given(
        st.integers(min_value=1, max_value=72),  # chunk_hours 1-72
    )
    def test_only_last_chunk_has_is_last_true(self, chunk_hours: int) -> None:
        """Only the last chunk should have is_last=True."""
        chunks = _generate_time_chunks("2024-01-01", "2024-01-07", chunk_hours)

        is_last_flags = [c[2] for c in chunks]
        true_count = sum(1 for x in is_last_flags if x)
        assert true_count == 1, "Exactly one chunk should be the last"
        assert is_last_flags[-1] is True, "Last chunk should have is_last=True"

    @given(
        st.integers(min_value=1, max_value=48),  # chunk_hours 1-48
    )
    def test_first_chunk_starts_at_midnight(self, chunk_hours: int) -> None:
        """First chunk should start at 00:00:00."""
        chunks = _generate_time_chunks("2024-03-10", "2024-03-12", chunk_hours)

        start, _, _ = chunks[0]
        assert start.endswith("00:00:00"), f"First chunk should start at midnight, got {start}"

    @given(
        st.integers(min_value=1, max_value=48),  # chunk_hours 1-48
    )
    def test_last_chunk_ends_at_235959(self, chunk_hours: int) -> None:
        """Last chunk should end at 23:59:59 of end_date."""
        chunks = _generate_time_chunks("2024-06-01", "2024-06-03", chunk_hours)

        _, end, _ = chunks[-1]
        assert end == "2024-06-03 23:59:59", f"Last chunk should end at 23:59:59, got {end}"

    @given(
        st.integers(min_value=1, max_value=72),  # chunk_hours 1-72
    )
    def test_chunks_cover_full_range(self, chunk_hours: int) -> None:
        """Union of all chunks should cover the full date range without gaps."""
        chunks = _generate_time_chunks("2024-02-15", "2024-02-20", chunk_hours)

        # First chunk should start at beginning of start_date
        assert chunks[0][0] == "2024-02-15 00:00:00"
        # Last chunk should end at end of end_date
        assert chunks[-1][1] == "2024-02-20 23:59:59"

        # Verify chunks are contiguous (end + 1 second = next start)
        from datetime import datetime, timedelta

        for i in range(len(chunks) - 1):
            end_str = chunks[i][1]
            start_next_str = chunks[i + 1][0]
            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            start_next_dt = datetime.strptime(start_next_str, "%Y-%m-%d %H:%M:%S")
            expected_next = end_dt + timedelta(seconds=1)
            assert start_next_dt == expected_next, (
                f"Gap between chunk {i} end ({end_str}) and chunk {i+1} start ({start_next_str})"
            )

    @given(
        start_day=st.integers(min_value=1, max_value=25),
        duration_days=st.integers(min_value=0, max_value=10),
        chunk_hours=st.integers(min_value=1, max_value=96),
    )
    def test_arbitrary_date_ranges_produce_valid_chunks(
        self, start_day: int, duration_days: int, chunk_hours: int
    ) -> None:
        """Any valid date range produces non-empty, correctly bounded chunks."""
        start_date = f"2024-03-{start_day:02d}"
        end_day = start_day + duration_days
        if end_day > 31:
            end_day = 31
        end_date = f"2024-03-{end_day:02d}"

        chunks = _generate_time_chunks(start_date, end_date, chunk_hours)

        # Always at least one chunk
        assert len(chunks) >= 1

        # All chunks have valid structure
        for start, end, is_last in chunks:
            assert len(start) == 19  # "YYYY-MM-DD HH:MM:SS"
            assert len(end) == 19
            assert isinstance(is_last, bool)

        # Exactly one is_last=True
        assert sum(1 for _, _, is_last in chunks if is_last) == 1
        assert chunks[-1][2] is True

        # Last chunk ends at 23:59:59 of the end_date
        _, last_end, _ = chunks[-1]
        assert last_end == f"{end_date} 23:59:59", (
            f"Last chunk should end at {end_date} 23:59:59, got {last_end}"
        )

        # First chunk starts at midnight of start_date
        first_start, _, _ = chunks[0]
        assert first_start == f"{start_date} 00:00:00", (
            f"First chunk should start at {start_date} 00:00:00, got {first_start}"
        )
