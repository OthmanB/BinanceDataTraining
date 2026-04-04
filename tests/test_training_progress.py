import builtins
import unittest
from unittest.mock import patch

from observability.training_progress import create_training_progress_callback


class _FakeWriter:
    def __init__(self) -> None:
        self.progress_updates = []
        self.epoch_metrics = []

    def update_training_progress(self, *, epochs_done: int, epochs_total: int, batches_done: int, batches_total: int) -> None:
        self.progress_updates.append((epochs_done, epochs_total, batches_done, batches_total))

    def update_epoch_metrics(self, epoch: int, metrics: dict) -> None:
        self.epoch_metrics.append((epoch, metrics))


class TestTrainingProgress(unittest.TestCase):
    def test_create_training_progress_callback_updates_writer(self) -> None:
        writer = _FakeWriter()
        callback = create_training_progress_callback(writer, epochs=3, steps_per_epoch=4)
        if callback is None:
            self.skipTest("TensorFlow is not available in this environment")

        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_end(0)
        callback.on_batch_end(1)
        callback.on_epoch_end(0, logs={"loss": 0.5, "text": "ignored"})

        self.assertGreaterEqual(len(writer.progress_updates), 4)
        self.assertEqual(writer.progress_updates[-1], (1, 3, 4, 4))
        self.assertEqual(writer.epoch_metrics, [(1, {"loss": 0.5})])

    def test_create_training_progress_callback_returns_none_when_tensorflow_missing(self) -> None:
        writer = _FakeWriter()
        real_import = builtins.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tensorflow":
                raise ImportError("tensorflow unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_import):
            callback = create_training_progress_callback(writer, epochs=1, steps_per_epoch=1)

        self.assertIsNone(callback)


if __name__ == "__main__":
    unittest.main()
