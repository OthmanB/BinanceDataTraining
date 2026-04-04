"""Unit tests for sequential HPO metric aggregation helpers."""

from __future__ import annotations

import unittest

from training.pipeline import _aggregate_hpo_window_metrics, _resolve_hpo_metric_weight


class TestSequentialHPOMetricAggregation(unittest.TestCase):
    def test_weight_uses_validation_count_for_val_metric(self) -> None:
        weight = _resolve_hpo_metric_weight("val_loss", effective_train_n=1000, val_count=200)
        self.assertEqual(weight, 200.0)

    def test_weight_uses_train_count_for_non_val_metric(self) -> None:
        weight = _resolve_hpo_metric_weight("loss", effective_train_n=1000, val_count=200)
        self.assertEqual(weight, 1000.0)

    def test_aggregate_returns_weighted_average(self) -> None:
        value = _aggregate_hpo_window_metrics(
            [
                (0.20, 100.0),
                (0.40, 300.0),
            ]
        )
        self.assertAlmostEqual(value, 0.35)

    def test_aggregate_falls_back_to_simple_average_without_weights(self) -> None:
        value = _aggregate_hpo_window_metrics(
            [
                (1.0, 0.0),
                (3.0, 0.0),
                (5.0, 0.0),
            ]
        )
        self.assertAlmostEqual(value, 3.0)

    def test_aggregate_returns_none_for_empty_metrics(self) -> None:
        self.assertIsNone(_aggregate_hpo_window_metrics([]))


if __name__ == "__main__":
    unittest.main()
