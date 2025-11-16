"""Model evaluation skeleton."""

from typing import Any, Dict
import logging
from pathlib import Path
import tempfile

import numpy as np

from preprocessing.train_test_split import chronological_split_indices


logger = logging.getLogger(__name__)


def evaluate_model(config: Dict[str, Any], model: Any, data_object: Dict[str, Any]) -> None:
    """Evaluate a trained model.

    Phase 3: placeholder that logs invocation only.
    """

    metadata = data_object["metadata"]
    n_samples = int(metadata["num_samples"])

    if n_samples <= 0:
        logger.info("evaluate_model invoked with num_samples=0, skipping evaluation.")
        return

    # Recompute chronological train/validation/test splits from configuration.
    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    _, _, test_idx = chronological_split_indices(
        n_samples,
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if not test_idx:
        logger.info("No test samples available for evaluation; skipping evaluation.")
        return

    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])

    # Limit evaluation to a reasonable number of samples.
    eval_n = min(len(test_idx), debug_max_samples)

    model_cfg = config["model"]
    cnn_cfg = model_cfg["cnn"]
    kernel_sizes = cnn_cfg["kernel_sizes"]
    pool_sizes = cnn_cfg["pool_sizes"]

    if not isinstance(kernel_sizes, list) or not kernel_sizes:
        raise ValueError("model.cnn.kernel_sizes must be a non-empty list in configuration")
    if not isinstance(pool_sizes, list) or not pool_sizes:
        raise ValueError("model.cnn.pool_sizes must be a non-empty list in configuration")

    heights = [int(k[0]) for k in kernel_sizes]
    widths = [int(k[1]) for k in kernel_sizes]

    pool_heights = [int(p[0]) for p in pool_sizes]
    pool_widths = [int(p[1]) for p in pool_sizes]

    min_height = 1
    for ph in pool_heights:
        min_height *= ph

    min_width = 1
    for pw in pool_widths:
        min_width *= pw

    height = max(max(heights), min_height)
    width = max(max(widths), min_width)
    channels = 1

    x_eval = np.zeros((eval_n, height, width, channels), dtype="float32")

    # Map snapshot-level top-of-book features for the target asset into the
    # evaluation tensor, using the same snapshot index space as the labels.
    try:
        data_cfg = config["data"]
        asset_pairs_cfg = data_cfg["asset_pairs"]
        target_asset = str(asset_pairs_cfg["target_asset"])
        order_books = data_object["order_books"]
        target_book = order_books.get(target_asset, {})
        snapshot_features = target_book.get("snapshot_features") or []
    except KeyError:
        snapshot_features = []

    if snapshot_features:
        logger.info(
            "Building evaluation inputs from snapshot_features (target_asset=%s). test_samples=%s, eval_n=%s",
            target_asset,
            len(test_idx),
            eval_n,
        )
        for i in range(eval_n):
            idx = test_idx[i]
            if idx >= len(snapshot_features):
                break

            features_i = snapshot_features[idx]
            if not isinstance(features_i, (list, tuple)) or len(features_i) < 4:
                continue

            bid_price, bid_qty, ask_price, ask_qty = features_i[:4]

            try:
                bid_price_f = float(bid_price)
                bid_qty_f = float(bid_qty)
                ask_price_f = float(ask_price)
                ask_qty_f = float(ask_qty)
            except (TypeError, ValueError):
                continue

            x_eval[i, 0, 0, 0] = bid_price_f
            if width > 1:
                x_eval[i, 0, 1, 0] = bid_qty_f
            if height > 1:
                x_eval[i, 1, 0, 0] = ask_price_f
            if height > 1 and width > 1:
                x_eval[i, 1, 1, 0] = ask_qty_f
    else:
        logger.info("No snapshot_features available for evaluation; using synthetic evaluation inputs.")
        x_eval = np.random.randn(eval_n, height, width, channels).astype("float32")

    num_classes = int(model_cfg["output"]["num_classes"])

    # Labels are built during preprocessing and stored in the DataObject.
    targets = data_object.get("targets")
    if targets is None:
        raise ValueError("data_object.targets must be populated by the preprocessing pipeline")

    labels_list = targets.get("labels")
    if labels_list is None:
        raise ValueError("data_object.targets.labels must be populated by the preprocessing pipeline")

    if len(labels_list) < len(test_idx):
        raise ValueError(
            "data_object.targets.labels length must be at least the number of samples used for splitting; "
            f"got labels={len(labels_list)}, n_samples={len(test_idx)}",
        )

    labels_arr = np.asarray(labels_list, dtype="int64")

    if labels_arr.min() < 0 or labels_arr.max() >= num_classes:
        raise ValueError(
            "Target label values must be in the range [0, num_classes-1]; "
            f"observed min={labels_arr.min()}, max={labels_arr.max()}, num_classes={num_classes}",
        )

    # Restrict to the evaluation subset defined by the test indices and debug_max_samples.
    eval_indices = test_idx[:eval_n]
    y_true_int = labels_arr[eval_indices]

    # Run model predictions on the evaluation set.
    try:
        y_prob = model.predict(x_eval, verbose=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model prediction failed during evaluation: %s", exc)
        return

    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        logger.warning(
            "Unexpected prediction shape during evaluation. expected=(eval_n,%s), got=%s",
            num_classes,
            y_prob.shape,
        )
        return

    y_pred_int = np.argmax(y_prob, axis=1)

    # Compute evaluation metrics.
    accuracy = float(np.mean(y_pred_int == y_true_int)) if eval_n > 0 else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_int, y_pred_int):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    for cls in range(num_classes):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    logger.info(
        "Evaluation metrics (Phase 4 minimal). eval_n=%s, accuracy=%s, macro_precision=%s, macro_recall=%s, macro_f1=%s",
        eval_n,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    )

    # Log evaluation metrics and optional confusion matrix to MLFlow.
    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for evaluation metric logging: %s", exc)
        return

    metrics = {
        "eval_accuracy": accuracy,
        "eval_macro_precision": macro_precision,
        "eval_macro_recall": macro_recall,
        "eval_macro_f1": macro_f1,
    }

    for cls, (prec, rec, f1) in enumerate(
        zip(per_class_precision, per_class_recall, per_class_f1),
    ):
        metrics[f"eval_precision_class_{cls}"] = prec
        metrics[f"eval_recall_class_{cls}"] = rec
        metrics[f"eval_f1_class_{cls}"] = f1

    for name, value in metrics.items():
        try:
            mlflow.log_metric(name, float(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log MLFlow evaluation metric %s: %s", name, exc)

    # Optional confusion matrix artifact logging.
    try:
        mlflow_cfg = config.get("mlflow", {})
        artifact_logging_cfg = mlflow_cfg.get("artifact_logging", {})
        log_confusion = bool(artifact_logging_cfg.get("confusion_matrix"))
    except Exception:  # noqa: BLE001
        log_confusion = False

    if log_confusion:
        tmp_dir = Path(tempfile.mkdtemp())
        cm_path = tmp_dir / "confusion_matrix.csv"
        try:
            np.savetxt(cm_path, confusion, fmt="%d", delimiter=",")
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            logger.info("Logged evaluation confusion matrix artifact to MLFlow at %s", cm_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log confusion matrix artifact to MLFlow: %s", exc)


__all__ = ["evaluate_model"]
