"""Keras callback to update run state during training."""

from __future__ import annotations

from typing import Any, Optional

from .run_state import RunStateWriter


def create_training_progress_callback(
    writer: RunStateWriter,
    epochs: int,
    steps_per_epoch: int,
) -> Optional[Any]:
    """Create a training progress callback when TensorFlow is available."""
    try:
        from tensorflow import keras  # type: ignore[import]
    except Exception:
        return None

    class _ProgressCallback(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._epochs = epochs
            self._steps = steps_per_epoch
            self._batch_in_epoch = 0
            self._epoch_index = 0

        def on_train_begin(self, logs: Optional[dict] = None) -> None:
            writer.update_training_progress(epochs_done=0, epochs_total=self._epochs, batches_done=0, batches_total=self._steps)

        def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
            self._batch_in_epoch = 0
            self._epoch_index = epoch
            writer.update_training_progress(
                epochs_done=epoch,
                epochs_total=self._epochs,
                batches_done=0,
                batches_total=self._steps,
            )

        def on_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
            self._batch_in_epoch += 1
            writer.update_training_progress(
                epochs_done=self._epoch_index,
                epochs_total=self._epochs,
                batches_done=self._batch_in_epoch,
                batches_total=self._steps,
            )

        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
            writer.update_training_progress(
                epochs_done=epoch + 1,
                epochs_total=self._epochs,
                batches_done=self._steps,
                batches_total=self._steps,
            )

    return _ProgressCallback()


__all__ = ["create_training_progress_callback"]
