"""Tests for runtime device availability validation."""

from __future__ import annotations

import logging
import types
import unittest
from unittest import mock

from main import _validate_runtime_device_availability
from utils.config_loader import ConfigError


class _FakeTFConfig:
    def __init__(self, devices: list[object]) -> None:
        self._devices = devices
        self.memory_growth_calls: list[tuple[object, bool]] = []
        self.experimental = types.SimpleNamespace(set_memory_growth=self._set_memory_growth)

    def list_physical_devices(self, device_type: str) -> list[object]:
        if device_type == "GPU":
            return self._devices
        return []

    def _set_memory_growth(self, device: object, enabled: bool) -> None:
        self.memory_growth_calls.append((device, enabled))


class TestRuntimeDeviceValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("tests.runtime_device")

    def test_cpu_mode_skips_gpu_checks(self) -> None:
        config = {"training": {"runtime": {"device": "cpu"}}}
        _validate_runtime_device_availability(config, self.logger)

    def test_gpu_mode_raises_when_no_visible_gpus(self) -> None:
        fake_tf = types.SimpleNamespace(config=_FakeTFConfig([]))
        config = {"training": {"runtime": {"device": "gpu"}}}

        with mock.patch.dict("sys.modules", {"tensorflow": fake_tf}):
            with self.assertRaises(ConfigError):
                _validate_runtime_device_availability(config, self.logger)

    def test_gpu_mode_passes_when_gpu_visible(self) -> None:
        fake_device = types.SimpleNamespace(name="/physical_device:GPU:0")
        fake_config = _FakeTFConfig([fake_device])
        fake_tf = types.SimpleNamespace(config=fake_config)
        config = {"training": {"runtime": {"device": "gpu"}}}

        with mock.patch.dict("sys.modules", {"tensorflow": fake_tf}):
            _validate_runtime_device_availability(config, self.logger)

        self.assertEqual(fake_config.memory_growth_calls, [])

    def test_gpu_mode_enables_memory_growth_when_requested(self) -> None:
        fake_device = types.SimpleNamespace(name="/physical_device:GPU:0")
        fake_config = _FakeTFConfig([fake_device])
        fake_tf = types.SimpleNamespace(config=fake_config)
        config = {
            "training": {
                "runtime": {
                    "device": "gpu",
                    "gpu_memory_growth": True,
                }
            }
        }

        with mock.patch.dict("sys.modules", {"tensorflow": fake_tf}):
            _validate_runtime_device_availability(config, self.logger)

        self.assertEqual(fake_config.memory_growth_calls, [(fake_device, True)])

    def test_gpu_mode_rejects_invalid_allocator_value(self) -> None:
        config = {
            "training": {
                "runtime": {
                    "device": "gpu",
                    "gpu_allocator": "invalid",
                }
            }
        }

        with self.assertRaises(ConfigError):
            _validate_runtime_device_availability(config, self.logger)


if __name__ == "__main__":
    unittest.main()
