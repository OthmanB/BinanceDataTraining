"""Unit tests for training.distributed config parsing (no TensorFlow required)."""

import unittest

from utils.config_loader import ConfigError
from training.distributed import parse_distributed_config, _parse_resources


class TestParseResources(unittest.TestCase):
    """Tests for _parse_resources validation."""

    def test_empty_list_returns_empty(self):
        self.assertEqual(_parse_resources([]), [])

    def test_valid_gpu_resources(self):
        result = _parse_resources(["gpu:0", "gpu:1"])
        self.assertEqual(result, ["/GPU:0", "/GPU:1"])

    def test_valid_cpu_resource(self):
        result = _parse_resources(["cpu:0"])
        self.assertEqual(result, ["/CPU:0"])

    def test_mixed_resources(self):
        result = _parse_resources(["gpu:0", "cpu:0", "gpu:1"])
        self.assertEqual(result, ["/GPU:0", "/CPU:0", "/GPU:1"])

    def test_case_insensitive(self):
        result = _parse_resources(["GPU:0", "Gpu:1"])
        self.assertEqual(result, ["/GPU:0", "/GPU:1"])

    def test_invalid_format_raises(self):
        with self.assertRaises(ConfigError):
            _parse_resources(["not_a_device"])

    def test_missing_index_raises(self):
        with self.assertRaises(ConfigError):
            _parse_resources(["gpu:"])

    def test_non_numeric_index_raises(self):
        with self.assertRaises(ConfigError):
            _parse_resources(["gpu:abc"])


class TestParseDistributedConfig(unittest.TestCase):
    """Tests for parse_distributed_config."""

    def test_missing_distributed_returns_none(self):
        result = parse_distributed_config({})
        self.assertIsNone(result)

    def test_disabled_returns_none(self):
        cfg = {"distributed": {"enabled": False, "strategy": "mirrored", "resources": []}}
        result = parse_distributed_config(cfg)
        self.assertIsNone(result)

    def test_enabled_mirrored_no_resources(self):
        cfg = {"distributed": {"enabled": True, "strategy": "mirrored", "resources": []}}
        result = parse_distributed_config(cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "mirrored")
        self.assertEqual(result["devices"], [])

    def test_enabled_mirrored_with_resources(self):
        cfg = {"distributed": {"enabled": True, "strategy": "mirrored", "resources": ["gpu:0", "gpu:1"]}}
        result = parse_distributed_config(cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "mirrored")
        self.assertEqual(result["devices"], ["/GPU:0", "/GPU:1"])

    def test_invalid_strategy_raises(self):
        cfg = {"distributed": {"enabled": True, "strategy": "unknown", "resources": []}}
        with self.assertRaises(ConfigError):
            parse_distributed_config(cfg)

    def test_resources_not_list_raises(self):
        cfg = {"distributed": {"enabled": True, "strategy": "mirrored", "resources": "gpu:0"}}
        with self.assertRaises(ConfigError):
            parse_distributed_config(cfg)

    def test_non_dict_distributed_returns_none(self):
        cfg = {"distributed": "invalid"}
        result = parse_distributed_config(cfg)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
