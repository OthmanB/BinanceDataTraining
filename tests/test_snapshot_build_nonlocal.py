"""Tests for prev_tail_* nonlocal scoping bug in _build_snapshot_chunks."""

from __future__ import annotations

import inspect
import unittest

from training.snapshot_dataset import _build_snapshot_chunks

__all__ = ["TestSnapshotBuildNonlocal"]


class TestSnapshotBuildNonlocal(unittest.TestCase):
    """Test that process_chunk properly declares prev_tail_* as nonlocal."""

    def test_nonlocal_declaration_exists(self) -> None:
        """
        Verify that _build_snapshot_chunks defines process_chunk with
        nonlocal declarations for prev_tail_base, prev_tail_confidence,
        prev_tail_observed, and prev_tail_ts.
        
        Without these nonlocal declarations, Python treats assignments
        to these variables as local, causing UnboundLocalError when
        the code reads them before assignment (which happens in the
        second chunk iteration).
        
        This is a static verification that the fix is in place.
        """
        source = inspect.getsource(_build_snapshot_chunks)
        
        self.assertIn(
            "nonlocal prev_tail_base, prev_tail_confidence, prev_tail_observed, prev_tail_ts",
            source,
            "process_chunk must declare prev_tail_base, prev_tail_confidence, prev_tail_observed, "
            "prev_tail_ts as nonlocal in a single declaration"
        )

    def test_build_snapshot_chunks_imports(self) -> None:
        """
        Verify that _build_snapshot_chunks can be imported without
        syntax or indentation errors in the nonlocal declarations.
        """
        self.assertTrue(
            callable(_build_snapshot_chunks),
            "_build_snapshot_chunks must be a callable function"
        )


if __name__ == "__main__":
    unittest.main()
