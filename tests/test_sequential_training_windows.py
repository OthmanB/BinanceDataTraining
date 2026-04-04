"""Tests for sequential snapshot training window planning."""

from __future__ import annotations

import unittest

from training.pipeline import _resolve_sequential_windows
from utils.config_loader import ConfigError


class TestSequentialTrainingWindows(unittest.TestCase):
    def _base_config(self) -> dict:
        return {
            "data": {
                "time_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                }
            },
            "training": {},
        }

    def test_resolve_returns_none_when_disabled(self) -> None:
        config = self._base_config()
        self.assertIsNone(_resolve_sequential_windows(config))

    def test_resolve_generates_expected_windows(self) -> None:
        config = self._base_config()
        config["training"]["sequential_training"] = {
            "enabled": True,
            "window_days": 3,
        }

        windows = _resolve_sequential_windows(config)

        self.assertEqual(
            windows,
            [
                ("2024-01-01", "2024-01-03"),
                ("2024-01-04", "2024-01-06"),
                ("2024-01-07", "2024-01-09"),
                ("2024-01-10", "2024-01-10"),
            ],
        )

    def test_resolve_raises_on_invalid_window_days(self) -> None:
        config = self._base_config()
        config["training"]["sequential_training"] = {
            "enabled": True,
            "window_days": 0,
        }

        with self.assertRaises(ConfigError):
            _resolve_sequential_windows(config)


if __name__ == "__main__":
    unittest.main()
