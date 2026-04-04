"""Unit tests for resolving best HPO params into a final training config."""

from __future__ import annotations

import types
import unittest


class TestHPOBestParamsResolution(unittest.TestCase):
    def test_overrides_batch_size_with_effective(self) -> None:
        from models.hyperparameter_tuning import _resolve_best_params_for_final_training  # noqa: PLC0415

        trial = types.SimpleNamespace(
            params={"batch_size": 19, "learning_rate": 0.001},
            user_attrs={"batch_size_effective": 8},
        )

        resolved, requested, effective = _resolve_best_params_for_final_training(trial)
        self.assertEqual(requested, 19)
        self.assertEqual(effective, 8)
        self.assertEqual(int(resolved["batch_size"]), 8)
        self.assertEqual(float(resolved["learning_rate"]), 0.001)

    def test_keeps_requested_when_effective_missing(self) -> None:
        from models.hyperparameter_tuning import _resolve_best_params_for_final_training  # noqa: PLC0415

        trial = types.SimpleNamespace(
            params={"batch_size": 16},
            user_attrs={},
        )

        resolved, requested, effective = _resolve_best_params_for_final_training(trial)
        self.assertEqual(requested, 16)
        self.assertIsNone(effective)
        self.assertEqual(int(resolved["batch_size"]), 16)

    def test_ignores_non_positive_or_invalid_effective(self) -> None:
        from models.hyperparameter_tuning import _resolve_best_params_for_final_training  # noqa: PLC0415

        trial_zero = types.SimpleNamespace(
            params={"batch_size": 12},
            user_attrs={"batch_size_effective": 0},
        )
        resolved, requested, effective = _resolve_best_params_for_final_training(trial_zero)
        self.assertEqual(requested, 12)
        self.assertEqual(effective, 0)
        self.assertEqual(int(resolved["batch_size"]), 12)

        trial_bad = types.SimpleNamespace(
            params={"batch_size": 12},
            user_attrs={"batch_size_effective": "nope"},
        )
        resolved, requested, effective = _resolve_best_params_for_final_training(trial_bad)
        self.assertEqual(requested, 12)
        self.assertIsNone(effective)
        self.assertEqual(int(resolved["batch_size"]), 12)


if __name__ == "__main__":
    unittest.main()
