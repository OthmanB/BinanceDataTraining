"""Model evaluation skeleton."""

from __future__ import annotations

from typing import Any, Dict
import logging
from pathlib import Path
import tempfile

import numpy as np

from preprocessing.train_test_split import chronological_split_indices
from preprocessing.snapshot_sequence_builder import build_top_of_book_sequence_tensor
from .calibration import compute_calibration_metrics


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

    eval_cfg = config["evaluation"]
    missing_snapshot_strategy = str(eval_cfg["missing_snapshot_strategy"])
    if missing_snapshot_strategy not in ("fail", "skip", "synthetic"):
        raise ValueError(
            "evaluation.missing_snapshot_strategy must be one of 'fail', 'skip', or 'synthetic'",
        )

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
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in the evaluation pipeline")

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

    eval_indices = test_idx[:eval_n]
    x_eval = None

    if snapshot_features:
        logger.info(
            "Building evaluation inputs from snapshot_features (target_asset=%s). test_samples=%s, eval_n=%s",
            target_asset,
            len(test_idx),
            eval_n,
        )

        anchor_indices = metadata.get("anchor_indices")
        if anchor_indices is None:
            raise ValueError(
                "metadata.anchor_indices must be populated by the preprocessing pipeline when snapshot_features are present",
            )

        x_eval = build_top_of_book_sequence_tensor(
            config=config,
            snapshot_features=snapshot_features,
            anchor_indices=list(anchor_indices),
            sample_indices=eval_indices,
            height=height,
            width=width,
            channels=channels,
        )
    else:
        if missing_snapshot_strategy == "fail":
            raise ValueError(
                "No snapshot_features available for evaluation inputs for target asset; "
                "set evaluation.missing_snapshot_strategy to 'skip' or 'synthetic' to change this behavior.",
            )

        if missing_snapshot_strategy == "skip":
            logger.info(
                "No snapshot_features available for evaluation inputs; skipping evaluation stage because "
                "evaluation.missing_snapshot_strategy='skip'.",
            )
            return

        data_cfg = config["data"]
        time_range_cfg = data_cfg["time_range"]
        cadence_seconds = int(time_range_cfg["cadence_seconds"])
        if cadence_seconds <= 0:
            raise ValueError("data.time_range.cadence_seconds must be positive")

        targets_cfg = config["targets"]
        visible_window_seconds = int(targets_cfg["visible_window_seconds"])
        if visible_window_seconds <= 0:
            raise ValueError("targets.visible_window_seconds must be positive")
        if visible_window_seconds % cadence_seconds != 0:
            raise ValueError(
                "targets.visible_window_seconds must be an integer multiple of data.time_range.cadence_seconds",
            )

        window_steps = visible_window_seconds // cadence_seconds
        if window_steps <= 0:
            raise ValueError(
                "Derived visible window length in steps must be at least one snapshot; "
                f"visible_window_seconds={visible_window_seconds}, cadence_seconds={cadence_seconds}",
            )

        # missing_snapshot_strategy == "synthetic"
        logger.info(
            "No snapshot_features available for evaluation inputs; using synthetic inputs because "
            "evaluation.missing_snapshot_strategy='synthetic'.",
        )
        x_eval = np.random.randn(eval_n, window_steps, height, width, channels).astype("float32")

    if x_eval is None:
        raise ValueError("Evaluation inputs could not be constructed; x_eval is None")

    if x_eval.ndim != 5:
        raise ValueError(
            "Evaluation input tensor must have shape (N, T, H, W, C); "
            f"got x_eval.ndim={x_eval.ndim}, shape={x_eval.shape!r}",
        )

    num_classes = int(output_cfg["num_classes"])

    # Labels are built during preprocessing and stored in the DataObject.
    targets = data_object.get("targets")
    if targets is None:
        raise ValueError("data_object.targets must be populated by the preprocessing pipeline")

    labels_up_list = targets.get("labels_up_intensity")
    labels_down_list = targets.get("labels_down_intensity")
    if labels_up_list is None or labels_down_list is None:
        raise ValueError(
            "data_object.targets.labels_up_intensity and labels_down_intensity must be populated by the preprocessing pipeline",
        )

    if len(labels_up_list) < len(test_idx) or len(labels_down_list) < len(test_idx):
        raise ValueError(
            "Intensity label arrays must have length at least the number of samples used for splitting; "
            f"got labels_up={len(labels_up_list)}, labels_down={len(labels_down_list)}, n_samples={len(test_idx)}",
        )

    labels_up_arr = np.asarray(labels_up_list, dtype="int64")
    labels_down_arr = np.asarray(labels_down_list, dtype="int64")

    if labels_up_arr.min() < 0 or labels_up_arr.max() >= num_classes:
        raise ValueError(
            "labels_up_intensity values must be in the range [0, num_classes-1]; "
            f"observed min={labels_up_arr.min()}, max={labels_up_arr.max()}, num_classes={num_classes}",
        )
    if labels_down_arr.min() < 0 or labels_down_arr.max() >= num_classes:
        raise ValueError(
            "labels_down_intensity values must be in the range [0, num_classes-1]; "
            f"observed min={labels_down_arr.min()}, max={labels_down_arr.max()}, num_classes={num_classes}",
        )

    # Restrict to the evaluation subset defined by the test indices and debug_max_samples.
    y_true_up = labels_up_arr[eval_indices]
    y_true_down = labels_down_arr[eval_indices]

    try:
        y_pred = model.predict(x_eval, verbose=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model prediction failed during evaluation: %s", exc)
        return

    if not isinstance(y_pred, (list, tuple)) or len(y_pred) != 2:
        logger.warning("Expected model.predict to return two outputs for two_head_intensity, got %r", type(y_pred))
        return

    y_prob_up, y_prob_down = y_pred

    if (
        y_prob_up.ndim != 2
        or y_prob_up.shape[1] != num_classes
        or y_prob_down.ndim != 2
        or y_prob_down.shape[1] != num_classes
    ):
        logger.warning(
            "Unexpected prediction shapes during evaluation. expected=(eval_n,%s) for each head, got up=%s, down=%s",
            num_classes,
            getattr(y_prob_up, "shape", None),
            getattr(y_prob_down, "shape", None),
        )
        return

    # Up-intensity head metrics.
    y_pred_up = np.argmax(y_prob_up, axis=1)
    accuracy_up = float(np.mean(y_pred_up == y_true_up)) if eval_n > 0 else 0.0

    per_class_precision_up = []
    per_class_recall_up = []
    per_class_f1_up = []

    confusion_up = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_up, y_pred_up):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion_up[t, p] += 1

    for cls in range(num_classes):
        tp = float(confusion_up[cls, cls])
        fp = float(confusion_up[:, cls].sum() - tp)
        fn = float(confusion_up[cls, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

        per_class_precision_up.append(precision)
        per_class_recall_up.append(recall)
        per_class_f1_up.append(f1)

    macro_precision_up = float(np.mean(per_class_precision_up)) if per_class_precision_up else 0.0
    macro_recall_up = float(np.mean(per_class_recall_up)) if per_class_recall_up else 0.0
    macro_f1_up = float(np.mean(per_class_f1_up)) if per_class_f1_up else 0.0

    # Down-intensity head metrics.
    y_pred_down = np.argmax(y_prob_down, axis=1)
    accuracy_down = float(np.mean(y_pred_down == y_true_down)) if eval_n > 0 else 0.0

    per_class_precision_down = []
    per_class_recall_down = []
    per_class_f1_down = []

    confusion_down = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_down, y_pred_down):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion_down[t, p] += 1

    for cls in range(num_classes):
        tp = float(confusion_down[cls, cls])
        fp = float(confusion_down[:, cls].sum() - tp)
        fn = float(confusion_down[cls, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

        per_class_precision_down.append(precision)
        per_class_recall_down.append(recall)
        per_class_f1_down.append(f1)

    macro_precision_down = float(np.mean(per_class_precision_down)) if per_class_precision_down else 0.0
    macro_recall_down = float(np.mean(per_class_recall_down)) if per_class_recall_down else 0.0
    macro_f1_down = float(np.mean(per_class_f1_down)) if per_class_f1_down else 0.0

    calib_cfg = eval_cfg["calibration_analysis"]
    calib_enabled = bool(calib_cfg["enabled"])
    calibration_results_up: Dict[str, Any] | None = None
    calibration_results_down: Dict[str, Any] | None = None

    if calib_enabled:
        n_bins = int(calib_cfg["n_bins"])
        if n_bins <= 0:
            raise ValueError("evaluation.calibration_analysis.n_bins must be a positive integer")

        try:
            y_true_up_onehot = np.eye(num_classes, dtype="float64")[y_true_up]
            calibration_results_up = compute_calibration_metrics(
                y_true=y_true_up_onehot,
                y_prob=y_prob_up,
                num_bins=n_bins,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute up-head calibration metrics during evaluation: %s", exc)
            calibration_results_up = None

        try:
            y_true_down_onehot = np.eye(num_classes, dtype="float64")[y_true_down]
            calibration_results_down = compute_calibration_metrics(
                y_true=y_true_down_onehot,
                y_prob=y_prob_down,
                num_bins=n_bins,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute down-head calibration metrics during evaluation: %s", exc)
            calibration_results_down = None

    logger.info(
        "Evaluation metrics (two_head_intensity). eval_n=%s, up_accuracy=%s, down_accuracy=%s, up_macro_precision=%s, down_macro_precision=%s, up_macro_recall=%s, down_macro_recall=%s, up_macro_f1=%s, down_macro_f1=%s",
        eval_n,
        accuracy_up,
        accuracy_down,
        macro_precision_up,
        macro_precision_down,
        macro_recall_up,
        macro_recall_down,
        macro_f1_up,
        macro_f1_down,
    )

    try:
        import mlflow  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import MLFlow for evaluation metric logging: %s", exc)
        return

    metrics = {
        "eval_up_accuracy": accuracy_up,
        "eval_up_macro_precision": macro_precision_up,
        "eval_up_macro_recall": macro_recall_up,
        "eval_up_macro_f1": macro_f1_up,
        "eval_down_accuracy": accuracy_down,
        "eval_down_macro_precision": macro_precision_down,
        "eval_down_macro_recall": macro_recall_down,
        "eval_down_macro_f1": macro_f1_down,
    }

    for cls, (prec, rec, f1) in enumerate(
        zip(per_class_precision_up, per_class_recall_up, per_class_f1_up),
    ):
        metrics[f"eval_up_precision_class_{cls}"] = prec
        metrics[f"eval_up_recall_class_{cls}"] = rec
        metrics[f"eval_up_f1_class_{cls}"] = f1

    for cls, (prec, rec, f1) in enumerate(
        zip(per_class_precision_down, per_class_recall_down, per_class_f1_down),
    ):
        metrics[f"eval_down_precision_class_{cls}"] = prec
        metrics[f"eval_down_recall_class_{cls}"] = rec
        metrics[f"eval_down_f1_class_{cls}"] = f1

    if calibration_results_up is not None:
        try:
            metrics["eval_up_brier_score"] = float(calibration_results_up["brier_score"])
            metrics["eval_up_ece"] = float(calibration_results_up["ece"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract up-head calibration metrics for MLFlow logging: %s", exc)

    if calibration_results_down is not None:
        try:
            metrics["eval_down_brier_score"] = float(calibration_results_down["brier_score"])
            metrics["eval_down_ece"] = float(calibration_results_down["ece"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract down-head calibration metrics for MLFlow logging: %s", exc)

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
        cm_up_path = tmp_dir / "confusion_matrix_up.csv"
        cm_down_path = tmp_dir / "confusion_matrix_down.csv"
        try:
            np.savetxt(cm_up_path, confusion_up, fmt="%d", delimiter=",")
            np.savetxt(cm_down_path, confusion_down, fmt="%d", delimiter=",")
            mlflow.log_artifact(str(cm_up_path), artifact_path="evaluation")
            mlflow.log_artifact(str(cm_down_path), artifact_path="evaluation")
            logger.info("Logged evaluation up/down confusion matrix artifacts to MLFlow at %s and %s", cm_up_path, cm_down_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log confusion matrix artifacts to MLFlow: %s", exc)

    if calib_enabled and (calibration_results_up is not None or calibration_results_down is not None):
        try:
            tmp_dir = Path(tempfile.mkdtemp())

            if calibration_results_up is not None:
                calib_up_path = tmp_dir / "calibration_curve_up.csv"
                edges = calibration_results_up["bin_edges"]
                conf = calibration_results_up["bin_confidence"]
                acc = calibration_results_up["bin_accuracy"]
                count = calibration_results_up["bin_count"]

                left_edges = edges[:-1]
                right_edges = edges[1:]
                data = np.column_stack([left_edges, right_edges, conf, acc, count])
                header = "left_edge,right_edge,bin_confidence,bin_accuracy,bin_count"
                np.savetxt(
                    calib_up_path,
                    data,
                    fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%d"],
                    delimiter=",",
                    header=header,
                    comments="",
                )

                mlflow.log_artifact(str(calib_up_path), artifact_path="evaluation")
                logger.info("Logged evaluation up-head calibration curve artifact to MLFlow at %s", calib_up_path)

            if calibration_results_down is not None:
                calib_down_path = tmp_dir / "calibration_curve_down.csv"
                edges = calibration_results_down["bin_edges"]
                conf = calibration_results_down["bin_confidence"]
                acc = calibration_results_down["bin_accuracy"]
                count = calibration_results_down["bin_count"]

                left_edges = edges[:-1]
                right_edges = edges[1:]
                data = np.column_stack([left_edges, right_edges, conf, acc, count])
                header = "left_edge,right_edge,bin_confidence,bin_accuracy,bin_count"
                np.savetxt(
                    calib_down_path,
                    data,
                    fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%d"],
                    delimiter=",",
                    header=header,
                    comments="",
                )

                mlflow.log_artifact(str(calib_down_path), artifact_path="evaluation")
                logger.info("Logged evaluation down-head calibration curve artifact to MLFlow at %s", calib_down_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log calibration curve artifacts to MLFlow: %s", exc)


__all__ = ["evaluate_model"]
