"""Model evaluation skeleton."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging
from pathlib import Path
import tempfile
import os

import numpy as np

from preprocessing.train_test_split import chronological_split_indices
from training.snapshot_dataset import (
    NormalizationStats,
    compute_normalization_stats,
    get_mask_channel_info,
    iter_snapshot_batches,
    load_normalization_stats,
    prepare_snapshot_dataset,
    save_normalization_stats,
)
from training.snapshot_store import load_or_create_manifest, resolve_snapshot_context, save_manifest
from preprocessing.snapshot_sequence_builder import (
    build_top_of_book_sequence_tensor,
    build_hybrid_depth_sequence_tensor,
)
from preprocessing.feature_engineering import FeatureEngineer
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

    # Map snapshot-level features for the target asset into the evaluation
    # tensor, using the same snapshot index space as the labels.
    data_cfg = config.get("data", {})
    asset_pairs_cfg = data_cfg.get("asset_pairs", {})
    target_asset = str(asset_pairs_cfg.get("target_asset") or "unknown")
    order_books = data_object.get("order_books", {})
    target_book: Dict[str, Any] = order_books.get(target_asset, {})
    snapshot_features: list[Any] = target_book.get("snapshot_features") or []
    snapshot_depth_data: list[Any] = target_book.get("snapshot_depth_data") or []

    eval_indices = test_idx[:eval_n]
    x_eval = None

    order_book_cfg = data_cfg.get("order_book", {})
    representation = str(order_book_cfg.get("representation", "top_of_book"))

    anchor_indices = metadata.get("anchor_indices")
    if anchor_indices is None:
        raise ValueError(
            "metadata.anchor_indices must be populated by the preprocessing pipeline when snapshot features are present",
        )

    if representation == "hybrid":
        if snapshot_depth_data:
            logger.info(
                "Building evaluation inputs from snapshot_depth_data (target_asset=%s). test_samples=%s, eval_n=%s",
                target_asset,
                len(test_idx),
                eval_n,
            )
            x_eval = build_hybrid_depth_sequence_tensor(
                config=config,
                snapshot_depth_data=snapshot_depth_data,
                anchor_indices=list(anchor_indices),
                sample_indices=eval_indices,
            )
        else:
            if missing_snapshot_strategy == "fail":
                raise ValueError(
                    "No snapshot_depth_data available for evaluation inputs for target asset; "
                    "set evaluation.missing_snapshot_strategy to 'skip' or 'synthetic' to change this behavior.",
                )

            if missing_snapshot_strategy == "skip":
                logger.info(
                    "No snapshot_depth_data available for evaluation inputs; skipping evaluation stage because "
                    "evaluation.missing_snapshot_strategy='skip'.",
                )
                return

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

            logger.info(
                "No snapshot_depth_data available for evaluation inputs; using synthetic inputs because "
                "evaluation.missing_snapshot_strategy='synthetic'.",
            )
            effective_levels = snapshot_depth_data[0]["bid_prices"].shape[0] if snapshot_depth_data else 1
            x_eval = np.random.randn(eval_n, window_steps, effective_levels, 4, 1).astype("float32")
    else:
        if snapshot_features:
            logger.info(
                "Building evaluation inputs from snapshot_features (target_asset=%s). test_samples=%s, eval_n=%s",
                target_asset,
                len(test_idx),
                eval_n,
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

            logger.info(
                "No snapshot_features available for evaluation inputs; using synthetic inputs because "
                "evaluation.missing_snapshot_strategy='synthetic'.",
            )
            x_eval = np.random.randn(eval_n, window_steps, height, width, channels).astype("float32")

    if x_eval is None:
        raise ValueError("Evaluation inputs could not be constructed; x_eval is None")

    # Optionally integrate feature engineering derived features into the input
    # channels, mirroring the training pipeline behavior.
    fe_cfg = config["preprocessing"].get("feature_engineering", {})
    if isinstance(fe_cfg, dict) and fe_cfg.get("enabled"):
        try:
            feature_engineer = FeatureEngineer(config)

            snapshot_derived_features = target_book.get("snapshot_derived_features")
            volume_proxy = target_book.get("volume_proxy")
            mid_prices_list = target_book.get("mid_prices")

            if snapshot_derived_features and volume_proxy and mid_prices_list:
                mid_prices_arr = np.asarray(mid_prices_list, dtype="float64")
                anchor_indices_list = list(anchor_indices)
                cadence_seconds = int(data_cfg["time_range"]["cadence_seconds"])

                all_features = feature_engineer.compute_all_features(
                    snapshot_depth_data=snapshot_depth_data,
                    mid_prices=mid_prices_arr,
                    anchor_indices=anchor_indices_list,
                    cadence_seconds=cadence_seconds,
                )

                if all_features is not None and all_features.shape[0] > 0:
                    fe_eval = all_features[eval_indices].astype("float32")

                    if x_eval.ndim == 5:
                        _, t_steps, h_dim, w_dim, _ = x_eval.shape
                        fe_eval_exp = fe_eval[:, None, None, None, :]
                        fe_eval_broadcast = np.broadcast_to(
                            fe_eval_exp,
                            (fe_eval.shape[0], t_steps, h_dim, w_dim, fe_eval.shape[1]),
                        )
                        x_eval = np.concatenate(
                            [x_eval, fe_eval_broadcast.astype("float32")], axis=-1
                        )

                        logger.info(
                            "Integrated feature engineering features into evaluation inputs: "
                            "n_features=%s, x_eval.shape=%s",
                            fe_eval.shape[1],
                            x_eval.shape,
                        )
            else:
                logger.info(
                    "Feature engineering skipped for evaluation: missing snapshot_derived_features, "
                    "volume_proxy, or mid_prices from preprocessing."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Feature engineering integration failed during evaluation: %s. Continuing without derived features.",
                exc,
            )

    # Optionally integrate temporal features into the evaluation input channels
    # according to the model.input_representation.temporal_features
    # configuration.
    ir_cfg = model_cfg.get("input_representation")
    if ir_cfg is None:
        raise ValueError("model.input_representation must be defined in configuration")
    tf_cfg = ir_cfg.get("temporal_features")
    if tf_cfg is None:
        raise ValueError("model.input_representation.temporal_features must be defined in configuration")

    integration_mode = str(tf_cfg["integration_mode"])
    use_local = bool(tf_cfg["use_local_features"])
    use_global = bool(tf_cfg["use_global_features"])

    if integration_mode not in ("none", "concat_channels"):
        raise ValueError(
            "model.input_representation.temporal_features.integration_mode must be one of 'none' or 'concat_channels'; "
            f"got {integration_mode!r}",
        )

    if integration_mode == "concat_channels":
        temporal_features = data_object.get("temporal_features")
        if not isinstance(temporal_features, dict):
            raise ValueError(
                "data_object.temporal_features must be a dict when temporal feature integration is enabled; "
                f"got {type(temporal_features)!r}",
            )

        local_arr = temporal_features.get("local")
        global_arr = temporal_features.get("global")

        if use_local:
            if local_arr is None:
                raise ValueError(
                    "Temporal feature integration is configured to use_local_features=True but "
                    "data_object.temporal_features.local is missing.",
                )
        if use_global:
            if global_arr is None:
                raise ValueError(
                    "Temporal feature integration is configured to use_global_features=True but "
                    "data_object.temporal_features.global is missing.",
                )

        feature_matrices = []
        if use_local and local_arr is not None:
            local_np = np.asarray(local_arr, dtype="float32")
            feature_matrices.append(local_np)
        if use_global and global_arr is not None:
            global_np = np.asarray(global_arr, dtype="float32")
            feature_matrices.append(global_np)

        if feature_matrices:
            tf_all = np.concatenate(feature_matrices, axis=1)
            if tf_all.shape[0] != n_samples:
                raise ValueError(
                    "Temporal feature matrices must have one row per sample; "
                    f"got tf_all.shape={tf_all.shape}, num_samples={n_samples}",
                )

            tf_eval = tf_all[eval_indices]
            tf_eval = np.asarray(tf_eval, dtype="float32")

            if x_eval.ndim != 5:
                raise ValueError(
                    "Evaluation input tensor must have rank 5 before temporal feature integration; "
                    f"got x_eval.ndim={x_eval.ndim}, shape={x_eval.shape!r}",
                )

            _, t_steps, h_dim, w_dim, _ = x_eval.shape
            tf_eval_exp = tf_eval[:, None, None, None, :]
            tf_eval_broadcast = np.broadcast_to(
                tf_eval_exp,
                (tf_eval.shape[0], t_steps, h_dim, w_dim, tf_eval.shape[1]),
            )
            x_eval = np.concatenate([x_eval, tf_eval_broadcast.astype("float32")], axis=-1)

            logger.info(
                "Integrated temporal features into evaluation inputs via concat_channels: "
                "eval_n=%s, feature_dim=%s",
                eval_n,
                tf_all.shape[1],
            )

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


def evaluate_snapshot_model(config: Dict[str, Any], model: Any) -> None:
    """Evaluate a trained model using snapshot datasets."""

    snapshot_dataset = prepare_snapshot_dataset(config)
    n_samples = int(snapshot_dataset.total_samples)
    if n_samples <= 0:
        logger.info("Snapshot evaluation invoked with num_samples=0; skipping.")
        return

    split_cfg = config["preprocessing"]["train_test_split"]
    train_ratio = float(split_cfg["train_ratio"])
    validation_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * validation_ratio)
    if val_end > n_samples:
        val_end = n_samples
    test_start = val_end
    test_end = n_samples

    if test_end <= test_start:
        logger.info("No test samples available for snapshot evaluation; skipping.")
        return

    training_cfg = config["training"]
    debug_max_samples = int(training_cfg["debug_max_samples"])
    if debug_max_samples > 0:
        test_end = min(test_end, test_start + debug_max_samples)

    if test_end <= test_start:
        logger.info("Snapshot evaluation skipped: debug_max_samples limits test set to zero.")
        return

    model_cfg = config["model"]
    output_cfg = model_cfg["output"]
    output_type = str(output_cfg["type"])
    if output_type != "two_head_intensity":
        raise ValueError("Only model.output.type='two_head_intensity' is supported in snapshot evaluation")

    num_classes = int(output_cfg["num_classes"])
    if num_classes <= 1:
        raise ValueError("model.output.num_classes must be >= 2 for evaluation")

    context = resolve_snapshot_context(config)
    manifest = load_or_create_manifest(context, config)

    normalization_cfg = config["preprocessing"]["normalization"]
    method = str(normalization_cfg["method"])
    fit_on_train_only = bool(normalization_cfg["fit_on_train_only"])

    mask_start, mask_count = get_mask_channel_info(config)

    stats_path = os.path.join(context.snapshot_dir, "normalization_stats_train.npz")
    if os.path.exists(stats_path):
        train_stats = load_normalization_stats(stats_path)
    else:
        train_stats = compute_normalization_stats(
            snapshot_dataset,
            0,
            train_end,
            method,
            mask_start=mask_start,
            mask_count=mask_count,
        )
        save_normalization_stats(stats_path, train_stats)

    stats_meta = manifest.get("normalization_stats", {})
    stats_meta["train"] = {
        "method": train_stats.method,
        "path": stats_path,
        "start_index": 0,
        "end_index": train_end,
    }
    manifest["normalization_stats"] = stats_meta
    save_manifest(context, manifest)

    eval_stats: NormalizationStats
    if fit_on_train_only:
        eval_stats = train_stats
    else:
        eval_stats = train_stats

    eval_cfg = config["evaluation"]
    calib_cfg = eval_cfg["calibration_analysis"]
    calib_enabled = bool(calib_cfg["enabled"])
    n_bins = int(calib_cfg["n_bins"])
    if n_bins <= 0:
        raise ValueError("evaluation.calibration_analysis.n_bins must be positive")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype="float64")

    def _init_calibration_state() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        return (
            np.zeros(n_bins, dtype="float64"),
            np.zeros(n_bins, dtype="float64"),
            np.zeros(n_bins, dtype="int64"),
            0.0,
            0,
        )

    up_conf_sum, up_acc_sum, up_bin_count, up_brier_sum, up_count = _init_calibration_state()
    down_conf_sum, down_acc_sum, down_bin_count, down_brier_sum, down_count = _init_calibration_state()

    confusion_up = np.zeros((num_classes, num_classes), dtype=int)
    confusion_down = np.zeros((num_classes, num_classes), dtype=int)

    total_eval = 0
    correct_up = 0
    correct_down = 0

    batch_size = int(training_cfg["batch_size"])
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    for x_chunk, y_up_chunk, y_down_chunk, _ in iter_snapshot_batches(
        snapshot_dataset, test_start, test_end
    ):
        x_chunk = _apply_normalization_snapshot(x_chunk, eval_stats, mask_start, mask_count)
        n_chunk = x_chunk.shape[0]

        for offset in range(0, n_chunk, batch_size):
            x_batch = x_chunk[offset : offset + batch_size]
            y_true_up = y_up_chunk[offset : offset + batch_size]
            y_true_down = y_down_chunk[offset : offset + batch_size]

            y_pred = model.predict(x_batch, batch_size=batch_size, verbose=0)
            if not isinstance(y_pred, (list, tuple)) or len(y_pred) != 2:
                raise ValueError("Expected model.predict to return two outputs for two_head_intensity")

            y_prob_up, y_prob_down = y_pred
            y_prob_up = np.asarray(y_prob_up, dtype="float64")
            y_prob_down = np.asarray(y_prob_down, dtype="float64")

            if y_prob_up.shape[1] != num_classes or y_prob_down.shape[1] != num_classes:
                raise ValueError("Prediction output classes do not match model.output.num_classes")

            y_pred_up = np.argmax(y_prob_up, axis=1)
            y_pred_down = np.argmax(y_prob_down, axis=1)

            correct_up += int(np.sum(y_pred_up == y_true_up))
            correct_down += int(np.sum(y_pred_down == y_true_down))
            total_eval += int(y_true_up.shape[0])

            np.add.at(confusion_up, (y_true_up, y_pred_up), 1)
            np.add.at(confusion_down, (y_true_down, y_pred_down), 1)

            if calib_enabled:
                y_true_up_onehot = np.eye(num_classes, dtype="float64")[y_true_up]
                y_true_down_onehot = np.eye(num_classes, dtype="float64")[y_true_down]

                up_brier_sum += float(
                    np.sum(np.sum((y_prob_up - y_true_up_onehot) ** 2, axis=1))
                )
                down_brier_sum += float(
                    np.sum(np.sum((y_prob_down - y_true_down_onehot) ** 2, axis=1))
                )
                up_count += int(y_true_up.shape[0])
                down_count += int(y_true_down.shape[0])

                up_bins = _assign_calibration_bins_with_truth(
                    y_prob_up,
                    y_true_up_onehot,
                    bin_edges,
                )
                down_bins = _assign_calibration_bins_with_truth(
                    y_prob_down,
                    y_true_down_onehot,
                    bin_edges,
                )

                up_conf_sum += up_bins[0]
                up_acc_sum += up_bins[1]
                up_bin_count += up_bins[2]

                down_conf_sum += down_bins[0]
                down_acc_sum += down_bins[1]
                down_bin_count += down_bins[2]

    if total_eval <= 0:
        logger.info("Snapshot evaluation found no samples after batching; skipping.")
        return

    accuracy_up = float(correct_up / total_eval)
    accuracy_down = float(correct_down / total_eval)

    per_class_precision_up, per_class_recall_up, per_class_f1_up = _compute_class_metrics(confusion_up)
    per_class_precision_down, per_class_recall_down, per_class_f1_down = _compute_class_metrics(confusion_down)

    macro_precision_up = float(np.mean(per_class_precision_up)) if per_class_precision_up else 0.0
    macro_recall_up = float(np.mean(per_class_recall_up)) if per_class_recall_up else 0.0
    macro_f1_up = float(np.mean(per_class_f1_up)) if per_class_f1_up else 0.0

    macro_precision_down = float(np.mean(per_class_precision_down)) if per_class_precision_down else 0.0
    macro_recall_down = float(np.mean(per_class_recall_down)) if per_class_recall_down else 0.0
    macro_f1_down = float(np.mean(per_class_f1_down)) if per_class_f1_down else 0.0

    calibration_results_up = None
    calibration_results_down = None
    if calib_enabled:
        calibration_results_up = _finalize_calibration(
            bin_edges,
            up_conf_sum,
            up_acc_sum,
            up_bin_count,
            up_brier_sum,
            up_count,
        )
        calibration_results_down = _finalize_calibration(
            bin_edges,
            down_conf_sum,
            down_acc_sum,
            down_bin_count,
            down_brier_sum,
            down_count,
        )

    logger.info(
        "Snapshot evaluation metrics. eval_n=%s, up_accuracy=%s, down_accuracy=%s, "
        "up_macro_precision=%s, down_macro_precision=%s, up_macro_recall=%s, down_macro_recall=%s, "
        "up_macro_f1=%s, down_macro_f1=%s",
        total_eval,
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
        logger.warning("Failed to import MLFlow for snapshot evaluation metric logging: %s", exc)
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
        zip(per_class_precision_up, per_class_recall_up, per_class_f1_up)
    ):
        metrics[f"eval_up_precision_class_{cls}"] = prec
        metrics[f"eval_up_recall_class_{cls}"] = rec
        metrics[f"eval_up_f1_class_{cls}"] = f1

    for cls, (prec, rec, f1) in enumerate(
        zip(per_class_precision_down, per_class_recall_down, per_class_f1_down)
    ):
        metrics[f"eval_down_precision_class_{cls}"] = prec
        metrics[f"eval_down_recall_class_{cls}"] = rec
        metrics[f"eval_down_f1_class_{cls}"] = f1

    if calibration_results_up is not None:
        metrics["eval_up_brier_score"] = float(calibration_results_up["brier_score"])
        metrics["eval_up_ece"] = float(calibration_results_up["ece"])
    if calibration_results_down is not None:
        metrics["eval_down_brier_score"] = float(calibration_results_down["brier_score"])
        metrics["eval_down_ece"] = float(calibration_results_down["ece"])

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
            logger.info(
                "Logged snapshot evaluation confusion matrices to MLFlow at %s and %s",
                cm_up_path,
                cm_down_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log confusion matrix artifacts to MLFlow: %s", exc)

    if calib_enabled and (calibration_results_up is not None or calibration_results_down is not None):
        try:
            tmp_dir = Path(tempfile.mkdtemp())

            if calibration_results_up is not None:
                calib_up_path = tmp_dir / "calibration_curve_up.csv"
                _write_calibration_curve(calibration_results_up, calib_up_path)
                mlflow.log_artifact(str(calib_up_path), artifact_path="evaluation")

            if calibration_results_down is not None:
                calib_down_path = tmp_dir / "calibration_curve_down.csv"
                _write_calibration_curve(calibration_results_down, calib_down_path)
                mlflow.log_artifact(str(calib_down_path), artifact_path="evaluation")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log calibration curve artifacts to MLFlow: %s", exc)


def _compute_class_metrics(confusion: np.ndarray) -> Tuple[list, list, list]:
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    num_classes = confusion.shape[0]
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

    return per_class_precision, per_class_recall, per_class_f1


def _finalize_calibration(
    bin_edges: np.ndarray,
    bin_conf_sum: np.ndarray,
    bin_acc_sum: np.ndarray,
    bin_count: np.ndarray,
    brier_sum: float,
    sample_count: int,
) -> Dict[str, Any]:
    bin_confidence = np.zeros_like(bin_conf_sum)
    bin_accuracy = np.zeros_like(bin_acc_sum)

    nonzero = bin_count > 0
    bin_confidence[nonzero] = bin_conf_sum[nonzero] / bin_count[nonzero]
    bin_accuracy[nonzero] = bin_acc_sum[nonzero] / bin_count[nonzero]

    total = float(bin_count.sum())
    if total > 0.0:
        abs_diff = np.abs(bin_confidence - bin_accuracy)
        weights = bin_count.astype("float64") / total
        ece = float(np.sum(abs_diff * weights))
    else:
        ece = 0.0

    if sample_count > 0:
        brier_score = float(brier_sum / float(sample_count))
    else:
        brier_score = 0.0

    return {
        "brier_score": brier_score,
        "ece": ece,
        "bin_edges": bin_edges,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
    }


def _write_calibration_curve(calibration_results: Dict[str, Any], path: Path) -> None:
    edges = calibration_results["bin_edges"]
    conf = calibration_results["bin_confidence"]
    acc = calibration_results["bin_accuracy"]
    count = calibration_results["bin_count"]

    left_edges = edges[:-1]
    right_edges = edges[1:]
    data = np.column_stack([left_edges, right_edges, conf, acc, count])
    header = "left_edge,right_edge,bin_confidence,bin_accuracy,bin_count"
    np.savetxt(
        path,
        data,
        fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%d"],
        delimiter=",",
        header=header,
        comments="",
    )


def _apply_normalization_snapshot(
    x: np.ndarray,
    stats: NormalizationStats,
    mask_start: int,
    mask_count: int,
) -> np.ndarray:
    x_non_mask, mask = _strip_mask_channels(x, mask_start, mask_count)
    x_flat = x_non_mask.reshape(x_non_mask.shape[0], -1)

    if stats.method == "min_max":
        if stats.min is None or stats.max is None:
            raise ValueError("Missing min/max normalization stats")
        denom = stats.max - stats.min
        denom = np.where(denom == 0, 1.0, denom)
        x_norm = (x_flat - stats.min) / denom
    elif stats.method == "standard":
        if stats.mean is None or stats.std is None:
            raise ValueError("Missing mean/std normalization stats")
        std = np.where(stats.std == 0, 1.0, stats.std)
        x_norm = (x_flat - stats.mean) / std
    else:
        raise ValueError(f"Unsupported normalization method for snapshot evaluation: {stats.method}")

    x_norm = x_norm.reshape(x_non_mask.shape).astype("float32")
    if mask is None:
        return x_norm

    left = x_norm[..., :mask_start]
    right = x_norm[..., mask_start:]
    return np.concatenate([left, mask, right], axis=-1)


def _strip_mask_channels(
    x: np.ndarray,
    mask_start: int,
    mask_count: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mask_count <= 0:
        return x, None
    mask_end = mask_start + mask_count
    mask = x[..., mask_start:mask_end]
    x_non_mask = np.concatenate([x[..., :mask_start], x[..., mask_end:]], axis=-1)
    return x_non_mask, mask


def _assign_calibration_bins_with_truth(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_flat = y_prob.ravel()
    true_flat = y_true.ravel()
    num_bins = bin_edges.shape[0] - 1
    bin_idx = np.searchsorted(bin_edges, probs_flat, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    bin_count = np.bincount(bin_idx, minlength=num_bins).astype("int64")
    bin_conf_sum = np.bincount(bin_idx, weights=probs_flat, minlength=num_bins).astype("float64")
    bin_acc_sum = np.bincount(bin_idx, weights=true_flat, minlength=num_bins).astype("float64")

    return bin_conf_sum, bin_acc_sum, bin_count


__all__ = ["evaluate_model", "evaluate_snapshot_model"]
