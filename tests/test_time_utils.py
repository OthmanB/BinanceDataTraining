import unittest

import numpy as np

from preprocessing.time_utils import normalize_timestamp_array


class TestTimeUtils(unittest.TestCase):
    def test_normalize_timestamp_array_empty(self) -> None:
        out = normalize_timestamp_array([])
        self.assertEqual(out.dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(out.size, 0)

    def test_normalize_timestamp_array_seconds_epoch(self) -> None:
        out = normalize_timestamp_array([1_700_000_000, 1_700_000_001])
        expected = np.asarray(["2023-11-14T22:13:20", "2023-11-14T22:13:21"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(out, expected)

    def test_normalize_timestamp_array_milliseconds_epoch(self) -> None:
        out = normalize_timestamp_array([1_700_000_000_000, 1_700_000_001_000])
        expected = np.asarray(["2023-11-14T22:13:20", "2023-11-14T22:13:21"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(out, expected)

    def test_normalize_timestamp_array_datetime64_input(self) -> None:
        raw = [np.datetime64("2024-01-01T00:00:00"), np.datetime64("2024-01-01T00:00:01")]
        out = normalize_timestamp_array(raw)
        self.assertEqual(out.dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(str(out[0]), "2024-01-01T00:00:00.000000000")


if __name__ == "__main__":
    unittest.main()
