import unittest

from preprocessing.validator import DataValidationError, validate_data_object


def _valid_data_object() -> dict:
    return {
        "metadata": {},
        "order_books": {},
        "temporal_features": {},
        "targets": {},
        "external_data": {},
    }


class TestValidator(unittest.TestCase):
    def test_validate_data_object_accepts_valid_structure(self) -> None:
        validate_data_object(_valid_data_object())

    def test_validate_data_object_raises_on_missing_key(self) -> None:
        bad = _valid_data_object()
        del bad["targets"]

        with self.assertRaises(DataValidationError) as exc_info:
            validate_data_object(bad)

        self.assertIn("targets", str(exc_info.exception))

    def test_validate_data_object_raises_on_wrong_type(self) -> None:
        bad = _valid_data_object()
        bad["metadata"] = []

        with self.assertRaises(DataValidationError) as exc_info:
            validate_data_object(bad)

        self.assertIn("metadata", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
