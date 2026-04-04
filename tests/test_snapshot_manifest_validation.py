"""Tests for snapshot cache manifest validation (WRF-8)."""

import os
import tempfile
import unittest

import numpy as np

from training.snapshot_dataset import load_snapshot_dataset, prepare_snapshot_dataset
from training.snapshot_store import SnapshotContext
from utils.config_loader import ConfigError


class TestSnapshotManifestValidation(unittest.TestCase):
    """Test manifest validation and missing chunk handling."""

    def _build_context(self, manifest_dir: str, config_hash: str) -> SnapshotContext:
        manifest_path = os.path.join(manifest_dir, "manifest.json")
        return SnapshotContext(
            snapshot_dir=manifest_dir,
            manifest_path=manifest_path,
            config_hash=config_hash,
            config_snapshot={},
            snapshot_name="test_snapshot",
            root_name="test_root",
        )

    def test_load_snapshot_dataset_logs_warning_for_missing_chunks(self) -> None:
        """Verify warning logs include chunk details when files are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_dir = os.path.join(tmp_dir, "snapshots", "test_hash")
            os.makedirs(manifest_dir, exist_ok=True)

            chunks = []
            for i in range(10):
                chunk_path = os.path.join(manifest_dir, f"chunk_{i}.npz")
                np.savez_compressed(
                    chunk_path,
                    x=np.zeros((10, 1, 1, 1, 1), dtype="float32"),
                    y_up=np.zeros((10,), dtype="int64"),
                    y_down=np.zeros((10,), dtype="int64"),
                    anchor_ts=np.arange(10, dtype="int64"),
                    duty_cycle=np.ones((10,), dtype="float32"),
                )
                chunks.append(
                    {
                        "start": f"2024-01-01 {i:02d}:00:00",
                        "end": f"2024-01-01 {i:02d}:59:59",
                        "file": f"chunk_{i}.npz",
                        "num_samples": 10,
                        "format": "npz",
                    }
                )

            chunks.append(
                {
                    "start": "2024-01-01 10:00:00",
                    "end": "2024-01-01 10:59:59",
                    "file": "missing_chunk.npz",
                    "num_samples": 10,
                    "format": "npz",
                }
            )

            manifest = {
                "config_hash": "test_hash",
                "complete": True,
                "chunks": chunks,
            }

            manifest_path = os.path.join(manifest_dir, "manifest.json")
            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            context = self._build_context(manifest_dir, "test_hash")

            config = {"snapshot": {}}

            with self.assertLogs("training.snapshot_dataset", level="WARNING") as cm:
                dataset = load_snapshot_dataset(context, config)

            self.assertEqual(len(dataset.chunks), 10)
            self.assertEqual(dataset.total_samples, 100)

            log_output = "\n".join(cm.output)
            self.assertIn("chunk_id=2024-01-01 10:00:00-2024-01-01 10:59:59", log_output)
            self.assertIn("format=npz", log_output)
            self.assertIn("num_samples=10", log_output)

    def test_load_snapshot_dataset_raises_when_missing_chunk_ratio_exceeds_threshold(
        self,
    ) -> None:
        """Verify threshold validation raises when too many chunks are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_dir = os.path.join(tmp_dir, "snapshots", "test_hash")
            os.makedirs(manifest_dir, exist_ok=True)

            chunks = []
            for i in range(12):
                chunks.append(
                    {
                        "start": f"2024-01-01 {i:02d}:00:00",
                        "end": f"2024-01-01 {i:02d}:59:59",
                        "file": f"missing_chunk_{i}.npz",
                        "num_samples": 100,
                        "format": "npz",
                    }
                )

            manifest = {
                "config_hash": "test_hash",
                "complete": True,
                "chunks": chunks,
            }

            manifest_path = os.path.join(manifest_dir, "manifest.json")
            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            context = self._build_context(manifest_dir, "test_hash")

            config = {"snapshot": {}}

            with self.assertRaises(ConfigError) as cm:
                load_snapshot_dataset(context, config)

            error_msg = str(cm.exception)
            self.assertIn("missing chunk ratio exceeds threshold", error_msg)
            self.assertIn("missing=12/12", error_msg)

    def test_load_snapshot_dataset_succeeds_when_missing_ratio_below_threshold(
        self,
    ) -> None:
        """Verify load succeeds when missing chunk ratio is below 10%."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_dir = os.path.join(tmp_dir, "snapshots", "test_hash")
            os.makedirs(manifest_dir, exist_ok=True)

            chunks = []
            for i in range(10):
                chunk_path = os.path.join(manifest_dir, f"chunk_{i}.npz")
                np.savez_compressed(
                    chunk_path,
                    x=np.zeros((10, 1, 1, 1, 1), dtype="float32"),
                    y_up=np.zeros((10,), dtype="int64"),
                    y_down=np.zeros((10,), dtype="int64"),
                    anchor_ts=np.arange(10, dtype="int64"),
                    duty_cycle=np.ones((10,), dtype="float32"),
                )
                chunks.append(
                    {
                        "start": f"2024-01-01 {i:02d}:00:00",
                        "end": f"2024-01-01 {i:02d}:59:59",
                        "file": f"chunk_{i}.npz",
                        "num_samples": 10,
                        "format": "npz",
                    }
                )

            manifest = {
                "config_hash": "test_hash",
                "complete": True,
                "chunks": chunks,
            }

            manifest_path = os.path.join(manifest_dir, "manifest.json")
            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            context = self._build_context(manifest_dir, "test_hash")

            config = {"snapshot": {}}

            with self.assertLogs("training.snapshot_dataset", level="INFO") as cm:
                dataset = load_snapshot_dataset(context, config)

            self.assertEqual(len(dataset.chunks), 10)
            self.assertEqual(dataset.total_samples, 100)

            log_output = "\n".join(cm.output)
            self.assertIn("chunks=10/10", log_output)
            self.assertIn("samples=100", log_output)
            self.assertIn("missing_chunks=0", log_output)

    def test_load_snapshot_dataset_summary_logging_with_some_missing_chunks(
        self,
    ) -> None:
        """Verify summary logging includes loaded/total counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_dir = os.path.join(tmp_dir, "snapshots", "test_hash")
            os.makedirs(manifest_dir, exist_ok=True)

            chunks = []
            for i in range(10):
                chunk_path = os.path.join(manifest_dir, f"chunk_{i}.npz")
                np.savez_compressed(
                    chunk_path,
                    x=np.zeros((10, 1, 1, 1, 1), dtype="float32"),
                    y_up=np.zeros((10,), dtype="int64"),
                    y_down=np.zeros((10,), dtype="int64"),
                    anchor_ts=np.arange(10, dtype="int64"),
                    duty_cycle=np.ones((10,), dtype="float32"),
                )
                chunks.append(
                    {
                        "start": f"2024-01-01 {i:02d}:00:00",
                        "end": f"2024-01-01 {i:02d}:59:59",
                        "file": f"chunk_{i}.npz",
                        "num_samples": 10,
                        "format": "npz",
                    }
                )

            chunks.append(
                {
                    "start": "2024-01-01 10:00:00",
                    "end": "2024-01-01 10:59:59",
                    "file": "missing_chunk.npz",
                    "num_samples": 10,
                    "format": "npz",
                }
            )

            manifest = {
                "config_hash": "test_hash",
                "complete": True,
                "chunks": chunks,
            }

            manifest_path = os.path.join(manifest_dir, "manifest.json")
            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            context = self._build_context(manifest_dir, "test_hash")

            config = {"snapshot": {}}

            with self.assertLogs("training.snapshot_dataset", level="INFO") as cm:
                dataset = load_snapshot_dataset(context, config)

            self.assertEqual(len(dataset.chunks), 10)
            self.assertEqual(dataset.total_samples, 100)

            log_output = "\n".join(cm.output)
            self.assertIn("chunks=10/11", log_output)
            self.assertIn("samples=100", log_output)
            self.assertIn("missing_chunks=1", log_output)

    def test_prepare_snapshot_dataset_invalidates_manifest_with_missing_chunks(self) -> None:
        """Verify prepare_snapshot_dataset() invalidates manifest when chunk files missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_root = os.path.join(tmp_dir, "snapshots")
            os.makedirs(snapshot_root, exist_ok=True)

            manifest_dir = os.path.join(snapshot_root, "test_hash")
            os.makedirs(manifest_dir, exist_ok=True)

            chunks = []
            for i in range(3):
                chunk_path = os.path.join(manifest_dir, f"chunk_{i}.npz")
                np.savez_compressed(
                    chunk_path,
                    x=np.zeros((5, 1, 1, 1, 1), dtype="float32"),
                    y_up=np.zeros((5,), dtype="int64"),
                    y_down=np.zeros((5,), dtype="int64"),
                    anchor_ts=np.arange(5, dtype="int64"),
                    duty_cycle=np.ones((5,), dtype="float32"),
                )
                chunks.append(
                    {
                        "start": f"2024-01-01 {i:02d}:00:00",
                        "end": f"2024-01-01 {i:02d}:59:59",
                        "file": f"chunk_{i}.npz",
                        "num_samples": 5,
                        "format": "npz",
                    }
                )

            chunks.append(
                {
                    "start": "2024-01-01 03:00:00",
                    "end": "2024-01-01 03:59:59",
                    "file": "missing_chunk.npz",
                    "num_samples": 5,
                    "format": "npz",
                }
            )

            manifest = {
                "config_hash": "test_hash",
                "complete": True,
                "chunks": chunks,
            }

            manifest_path = os.path.join(manifest_dir, "manifest.json")
            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            context = self._build_context(manifest_dir, "test_hash")

            config = {"snapshot": {"max_snapshots": 10}}

            from unittest.mock import patch, MagicMock

            with patch("training.snapshot_dataset.resolve_snapshot_context", return_value=context):
                with patch("training.snapshot_dataset._build_snapshot_chunks") as mock_build:
                    with patch("training.snapshot_dataset.load_snapshot_dataset") as mock_load:
                        mock_build.return_value = {"complete": False, "chunks": []}
                        mock_dataset = MagicMock()
                        mock_load.return_value = mock_dataset
                        
                        result = prepare_snapshot_dataset(config)
                        
                        mock_build.assert_called_once()
                        self.assertEqual(result, mock_dataset)


if __name__ == "__main__":
    unittest.main()
