import unittest

from preprocessing.train_test_split import chronological_split_indices


class TestChronologicalSplit(unittest.TestCase):
    def test_basic_split_shapes_and_order(self) -> None:
        train, val, test = chronological_split_indices(10, 0.7, 0.2, 0.1)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(test), 1)

        self.assertEqual(train, list(range(0, 7)))
        self.assertEqual(val, list(range(7, 9)))
        self.assertEqual(test, list(range(9, 10)))

    def test_invalid_ratios_raise(self) -> None:
        with self.assertRaises(ValueError):
            chronological_split_indices(10, 0.5, 0.5, 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
