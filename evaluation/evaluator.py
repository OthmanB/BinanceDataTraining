"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
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
from training.long_term_context import (
    compute_long_term_features_for_dataset,
    is_long_term_enabled,
    load_anchor_timestamps,
    load_snapshot_series,
)
from preprocessing.snapshot_sequence_builder import (
    build_top_of_book_sequence_tensor,
    build_hybrid_depth_sequence_tensor,
)
from preprocessing.feature_engineering import FeatureEngineer
from .calibration import (
    compute_calibration_metrics,
    fit_temperature,
    probs_to_logits_proxy,
    logits_to_calibrated_probs,
)
from .temporal_degradation import TemporalDegradationResult, WindowMetrics


logger = logging.getLogger(__name__)

CALIBRATION_MEMORY_WARN_THRESHOLD = 1_000_000
BACKTEST_MEMORY_WARN_THRESHOLD = 1_000_000


def evaluate_model(config: Dict[str, Any], model: Any, data_object: Dict[str, Any]) -> None:
    """Evaluate a trained model."""

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
    data_cfg = config["data"]
    asset_pairs_cfg = data_cfg["asset_pairs"]
    target_asset = str(asset_pairs_cfg["target_asset"])
    order_books = data_object.get("order_books", {})
    target_book: Dict[str, Any] = order_books.get(target_asset, {})
    snapshot_features: list[Any] = target_book.get("snapshot_features") or []
    snapshot_depth_data: list[Any] = target_book.get("snapshot_depth_data") or []

    eval_indices = test_idx[:eval_n]
    x_eval = None

    order_book_cfg = data_cfg["order_book"]
    representation = str(order_book_cfg["representation"])

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
    fe_cfg = config["preprocessing"]["feature_engineering"]
    if bool(fe_cfg["enabled"]):
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
    ir_cfg = model_cfg["input_representation"]
    tf_cfg = ir_cfg["temporal_features"]

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
    calibration_fit_summary: Dict[str, Any] | None = None  # Only populated in evaluate_snapshot_model

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

    if calibration_fit_summary is not None:
        try:
            tmp_dir = Path(tempfile.mkdtemp())
            summary_path = tmp_dir / "post_hoc_calibration_summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(calibration_fit_summary, handle, indent=2, sort_keys=True)
            mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
            logger.info(
                "Logged post-hoc calibration summary artifact to MLFlow at %s",
                summary_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log post-hoc calibration summary artifact to MLFlow: %s", exc)

    # Optional confusion matrix artifact logging.
    try:
        mlflow_cfg = config["mlflow"]
        artifact_logging_cfg = mlflow_cfg["artifact_logging"]
        log_confusion = bool(artifact_logging_cfg["confusion_matrix"])
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

    # Backtesting integration
    backtest_cfg = eval_cfg["backtesting"]
    if bool(backtest_cfg["enabled"]):
        try:
            from .backtesting import run_backtest, log_backtest_to_mlflow

            # Get mid prices for backtesting
            mid_prices_list = target_book.get("mid_prices")
            if mid_prices_list is None or len(mid_prices_list) == 0:
                logger.warning("Backtesting skipped: no mid_prices available in target_book")
            else:
                mid_prices = np.asarray(mid_prices_list, dtype="float64")

                # Map eval_indices to prices using anchor_indices
                anchor_indices_list = list(anchor_indices)
                eval_prices = np.array([
                    mid_prices[anchor_indices_list[idx]] if idx < len(anchor_indices_list) and anchor_indices_list[idx] < len(mid_prices) else 0.0
                    for idx in eval_indices
                ])

                horizon_steps = int(backtest_cfg["horizon_steps"])
                if horizon_steps <= 0:
                    raise ValueError("evaluation.backtesting.horizon_steps must be positive")

                # Run backtest
                backtest_result = run_backtest(
                    config=config,
                    y_prob_up=y_prob_up,
                    y_prob_down=y_prob_down,
                    prices=eval_prices,
                    horizon_steps=horizon_steps,
                )

                # Log to MLflow
                log_backtest_to_mlflow(backtest_result)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Backtesting failed during evaluation: %s", exc)


def evaluate_snapshot_model(config: Dict[str, Any], model: Any) -> None:
    """Evaluate a trained model using snapshot datasets."""

    writer = None
    try:
        from observability.run_state import get_run_state_writer

        writer = get_run_state_writer()
    except Exception:
        writer = None

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

    long_term_features: Optional[np.ndarray] = None
    long_term_enabled = is_long_term_enabled(config)
    if long_term_enabled:
        cadence_seconds = int(config["data"]["time_range"]["cadence_seconds"])
        long_term_features = compute_long_term_features_for_dataset(
            config,
            snapshot_dataset,
            cadence_seconds=cadence_seconds,
        )
        if long_term_features is None:
            raise ValueError("Long-term features enabled but computation returned None")
        if long_term_features.shape[0] != n_samples:
            raise ValueError(
                "Long-term feature rows do not match snapshot dataset sample count: "
                f"features={long_term_features.shape[0]}, samples={n_samples}"
            )

    try:
        input_count = len(getattr(model, "inputs", []))
    except Exception:
        input_count = 1
    if long_term_features is not None and input_count != 2:
        raise ValueError(
            "Long-term features are enabled but model does not expose two inputs. "
            "Disable model.long_term or rebuild the model with dual inputs."
        )
    if long_term_features is None and input_count == 2:
        raise ValueError(
            "Model expects long-term inputs but model.long_term is disabled. "
            "Enable model.long_term and rebuild the snapshot dataset."
        )

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

    batch_size = int(training_cfg["batch_size"])
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")

    total_eval_samples = test_end - test_start
    total_eval_batches = int(np.ceil(total_eval_samples / float(batch_size))) if total_eval_samples > 0 else 0
    eval_batches_done = 0
    if writer is not None:
        try:
            writer.update_eval_progress(processed=0, total=total_eval_batches)
        except Exception:
            pass

    eval_cfg = config["evaluation"]
    calib_cfg = eval_cfg["calibration_analysis"]
    calib_enabled = bool(calib_cfg["enabled"])
    n_bins = int(calib_cfg["n_bins"])
    if n_bins <= 0:
        raise ValueError("evaluation.calibration_analysis.n_bins must be positive")

    post_hoc_cfg = eval_cfg["post_hoc_calibration"]
    post_hoc_enabled = bool(post_hoc_cfg["enabled"])

    long_term_test_features: Optional[np.ndarray] = None
    if long_term_features is not None:
        long_term_test_features = long_term_features[test_start:test_end]
        if long_term_test_features.shape[0] != (test_end - test_start):
            raise ValueError(
                "Long-term feature slice does not match test range length: "
                f"features={long_term_test_features.shape[0]}, expected={test_end - test_start}"
            )

    backtest_cfg = eval_cfg["backtesting"]
    backtest_enabled = bool(backtest_cfg["enabled"])
    backtest_horizon_steps = int(backtest_cfg["horizon_steps"])
    if backtest_enabled and backtest_horizon_steps <= 0:
        raise ValueError("evaluation.backtesting.horizon_steps must be positive")

    backtest_prices: Optional[np.ndarray] = None
    backtest_timestamps: Optional[np.ndarray] = None
    backtest_prob_up_parts: List[np.ndarray] = []
    backtest_prob_down_parts: List[np.ndarray] = []

    def _map_anchor_timestamps_to_prices(
        series_timestamps: np.ndarray,
        series_prices: np.ndarray,
        anchor_timestamps: np.ndarray,
    ) -> np.ndarray:
        if series_timestamps.ndim != 1 or series_prices.ndim != 1:
            raise ValueError("Series timestamps and prices must be 1D arrays")
        if series_timestamps.shape[0] != series_prices.shape[0]:
            raise ValueError("Series timestamps and prices length mismatch")
        if series_timestamps.shape[0] == 0:
            raise ValueError("Series timestamps are empty; cannot map anchor timestamps")
        if anchor_timestamps.ndim != 1:
            raise ValueError("Anchor timestamps must be a 1D array")

        if np.any(series_timestamps[1:] < series_timestamps[:-1]):
            raise ValueError("Series timestamps must be sorted in ascending order")

        indices = np.searchsorted(series_timestamps, anchor_timestamps)
        valid = indices < series_timestamps.shape[0]
        matches = np.zeros_like(valid, dtype=bool)
        if np.any(valid):
            matches[valid] = series_timestamps[indices[valid]] == anchor_timestamps[valid]
        invalid = ~valid | ~matches
        if np.any(invalid):
            missing_count = int(np.sum(invalid))
            raise ValueError(
                "Anchor timestamps do not align with series timestamps: "
                f"missing={missing_count}, total={anchor_timestamps.shape[0]}"
            )

        return series_prices[indices]

    if backtest_enabled:
        series_timestamps, series_mid_prices, _ = load_snapshot_series(snapshot_dataset)
        anchor_timestamps = load_anchor_timestamps(snapshot_dataset)
        if anchor_timestamps.shape[0] != n_samples:
            raise ValueError(
                "Anchor timestamps length does not match snapshot dataset sample count: "
                f"anchors={anchor_timestamps.shape[0]}, samples={n_samples}"
            )

        price_by_anchor = _map_anchor_timestamps_to_prices(
            series_timestamps.astype("int64"),
            series_mid_prices.astype("float64"),
            anchor_timestamps.astype("int64"),
        )
        backtest_prices = price_by_anchor[test_start:test_end]
        backtest_timestamps = anchor_timestamps[test_start:test_end]
        if backtest_prices.shape[0] != (test_end - test_start):
            raise ValueError("Backtesting price slice does not match test range length")

        if backtest_prices.shape[0] > BACKTEST_MEMORY_WARN_THRESHOLD:
            logger.warning(
                "Backtesting will buffer %s samples in memory; consider reducing evaluation range",
                backtest_prices.shape[0],
            )

    # Temporal degradation configuration
    temporal_cfg = eval_cfg["temporal_degradation"]
    temporal_enabled = bool(temporal_cfg["enabled"])
    temporal_num_windows = int(temporal_cfg["num_windows"])
    temporal_overlap = float(temporal_cfg["overlap_fraction"])
    temporal_log_per_window = bool(temporal_cfg["log_per_window_metrics"])

    if temporal_enabled:
        if temporal_num_windows < 1:
            raise ValueError("evaluation.temporal_degradation.num_windows must be >= 1")
        if not 0.0 <= temporal_overlap < 0.5:
            raise ValueError("evaluation.temporal_degradation.overlap_fraction must be in [0.0, 0.5)")

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
    up_conf_sum_cal, up_acc_sum_cal, up_bin_count_cal, up_brier_sum_cal, up_count_cal = _init_calibration_state()
    down_conf_sum_cal, down_acc_sum_cal, down_bin_count_cal, down_brier_sum_cal, down_count_cal = _init_calibration_state()

    confusion_up = np.zeros((num_classes, num_classes), dtype=int)
    confusion_down = np.zeros((num_classes, num_classes), dtype=int)

    total_eval = 0
    correct_up = 0
    correct_down = 0

    temperature_up: Optional[float] = None
    temperature_down: Optional[float] = None
    calibration_fit_summary: Optional[Dict[str, Any]] = None

    temporal_windows: list[Tuple[int, int]] = []
    temporal_confusions_up: list[np.ndarray] = []
    temporal_confusions_down: list[np.ndarray] = []
    temporal_counts: list[int] = []

    if temporal_enabled:
        total_samples = test_end - test_start
        if total_samples <= 0:
            raise ValueError("Temporal degradation requires at least one evaluation sample")

        if temporal_num_windows == 1:
            window_size = total_samples
            step_size = total_samples
        else:
            effective_units = 1 + (temporal_num_windows - 1) * (1 - temporal_overlap)
            window_size = int(np.ceil(total_samples / effective_units))
            step_size = int(window_size * (1 - temporal_overlap))
            step_size = max(1, step_size)

        for i in range(temporal_num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, total_samples)
            if start_idx >= total_samples:
                break
            temporal_windows.append((start_idx, end_idx))
            temporal_confusions_up.append(np.zeros((num_classes, num_classes), dtype=np.int64))
            temporal_confusions_down.append(np.zeros((num_classes, num_classes), dtype=np.int64))
            temporal_counts.append(0)

    def _collect_calibration_predictions(
        start_index: int,
        end_index: int,
        lt_features: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probs_up_parts = []
        probs_down_parts = []
        labels_up_parts = []
        labels_down_parts = []

        expected_len = end_index - start_index
        if lt_features is not None and lt_features.shape[0] != expected_len:
            raise ValueError(
                "Long-term feature slice length does not match calibration range: "
                f"features={lt_features.shape[0]}, expected={expected_len}"
            )

        current_idx = 0

        for x_chunk, y_up_chunk, y_down_chunk, _, _ in iter_snapshot_batches(
            snapshot_dataset, start_index, end_index
        ):
            x_chunk = _apply_normalization_snapshot(x_chunk, eval_stats, mask_start, mask_count)
            n_chunk = x_chunk.shape[0]

            for offset in range(0, n_chunk, batch_size):
                x_batch = x_chunk[offset : offset + batch_size]
                y_true_up = y_up_chunk[offset : offset + batch_size]
                y_true_down = y_down_chunk[offset : offset + batch_size]

                model_input: Any
                if lt_features is not None:
                    lt_batch = lt_features[current_idx : current_idx + x_batch.shape[0]]
                    if lt_batch.shape[0] != x_batch.shape[0]:
                        raise ValueError("Long-term feature batch size mismatch during calibration")
                    model_input = [x_batch, lt_batch]
                else:
                    model_input = x_batch

                y_pred = model.predict(model_input, batch_size=batch_size, verbose=0)
                if not isinstance(y_pred, (list, tuple)) or len(y_pred) != 2:
                    raise ValueError("Expected model.predict to return two outputs for two_head_intensity")

                y_prob_up, y_prob_down = y_pred
                y_prob_up = np.asarray(y_prob_up, dtype="float64")
                y_prob_down = np.asarray(y_prob_down, dtype="float64")

                if y_prob_up.shape[1] != num_classes or y_prob_down.shape[1] != num_classes:
                    raise ValueError("Prediction output classes do not match model.output.num_classes")

                probs_up_parts.append(y_prob_up)
                probs_down_parts.append(y_prob_down)
                labels_up_parts.append(np.asarray(y_true_up, dtype="int64"))
                labels_down_parts.append(np.asarray(y_true_down, dtype="int64"))

                current_idx += x_batch.shape[0]

        if not probs_up_parts:
            raise ValueError("No predictions collected for post-hoc calibration fitting")

        probs_up = np.concatenate(probs_up_parts, axis=0)
        probs_down = np.concatenate(probs_down_parts, axis=0)
        labels_up = np.concatenate(labels_up_parts, axis=0)
        labels_down = np.concatenate(labels_down_parts, axis=0)

        if probs_up.shape[0] != labels_up.shape[0] or probs_down.shape[0] != labels_down.shape[0]:
            raise ValueError("Calibration data size mismatch between predictions and labels")

        return probs_up, probs_down, labels_up, labels_down

    if post_hoc_enabled:
        method = str(post_hoc_cfg["method"])
        if method != "temperature_scaling":
            raise ValueError(
                "evaluation.post_hoc_calibration.method must be 'temperature_scaling' when enabled"
            )

        fit_on_validation = bool(post_hoc_cfg["fit_on_validation"])
        bounds_cfg = post_hoc_cfg["temperature_bounds"]
        min_temp = float(bounds_cfg["min"])
        max_temp = float(bounds_cfg["max"])
        min_samples = int(post_hoc_cfg["min_samples"])

        if min_samples <= 0:
            raise ValueError("evaluation.post_hoc_calibration.min_samples must be positive")
        if min_temp <= 0.0 or max_temp <= 0.0 or min_temp >= max_temp:
            raise ValueError(
                "evaluation.post_hoc_calibration.temperature_bounds must satisfy 0 < min < max"
            )

        fit_start = train_end if fit_on_validation else test_start
        fit_end = val_end if fit_on_validation else test_end
        fit_count = fit_end - fit_start
        if fit_count <= 0:
            raise ValueError("Post-hoc calibration fit range is empty")

        if fit_count > CALIBRATION_MEMORY_WARN_THRESHOLD:
            logger.warning(
                "Post-hoc calibration will buffer %s samples in memory; consider reducing evaluation range",
                fit_count,
            )

        long_term_fit_features: Optional[np.ndarray] = None
        if long_term_features is not None:
            long_term_fit_features = long_term_features[fit_start:fit_end]

        probs_up_fit, probs_down_fit, labels_up_fit, labels_down_fit = _collect_calibration_predictions(
            fit_start,
            fit_end,
            long_term_fit_features,
        )

        fit_samples = int(probs_up_fit.shape[0])
        if fit_samples < min_samples:
            raise ValueError(
                "Post-hoc calibration requires at least {min_samples} samples; got {fit_samples}".format(
                    min_samples=min_samples,
                    fit_samples=fit_samples,
                )
            )

        logits_up = probs_to_logits_proxy(probs_up_fit)
        logits_down = probs_to_logits_proxy(probs_down_fit)

        scaler_up = fit_temperature(
            logits_up,
            labels_up_fit,
            num_bins=n_bins,
            bounds=(min_temp, max_temp),
        )
        scaler_down = fit_temperature(
            logits_down,
            labels_down_fit,
            num_bins=n_bins,
            bounds=(min_temp, max_temp),
        )

        if not scaler_up.fitted or not scaler_down.fitted:
            raise ValueError("Post-hoc calibration failed to fit temperature scalers")

        temperature_up = scaler_up.temperature
        temperature_down = scaler_down.temperature

        calibration_fit_summary = {
            "fit_split": "validation" if fit_on_validation else "test",
            "num_samples": fit_samples,
            "num_bins": n_bins,
            "temperature_bounds": {"min": min_temp, "max": max_temp},
            "up": scaler_up.to_dict(),
            "down": scaler_down.to_dict(),
        }

    sample_offset = 0

    for x_chunk, y_up_chunk, y_down_chunk, _, _ in iter_snapshot_batches(
        snapshot_dataset, test_start, test_end
    ):
        x_chunk = _apply_normalization_snapshot(x_chunk, eval_stats, mask_start, mask_count)
        n_chunk = x_chunk.shape[0]

        for offset in range(0, n_chunk, batch_size):
            x_batch = x_chunk[offset : offset + batch_size]
            y_true_up = y_up_chunk[offset : offset + batch_size]
            y_true_down = y_down_chunk[offset : offset + batch_size]

            model_input: Any
            if long_term_test_features is not None:
                lt_batch = long_term_test_features[sample_offset : sample_offset + x_batch.shape[0]]
                if lt_batch.shape[0] != x_batch.shape[0]:
                    raise ValueError("Long-term feature batch size mismatch during evaluation")
                model_input = [x_batch, lt_batch]
            else:
                model_input = x_batch

            y_pred = model.predict(model_input, batch_size=batch_size, verbose=0)
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

            batch_start = sample_offset
            batch_end = batch_start + y_true_up.shape[0]

            if temporal_enabled and temporal_windows:
                for window_index, (win_start, win_end) in enumerate(temporal_windows):
                    overlap_start = max(win_start, batch_start)
                    overlap_end = min(win_end, batch_end)
                    if overlap_start >= overlap_end:
                        continue

                    local_start = overlap_start - batch_start
                    local_end = overlap_end - batch_start

                    y_true_up_slice = y_true_up[local_start:local_end]
                    y_pred_up_slice = y_pred_up[local_start:local_end]
                    y_true_down_slice = y_true_down[local_start:local_end]
                    y_pred_down_slice = y_pred_down[local_start:local_end]

                    np.add.at(
                        temporal_confusions_up[window_index],
                        (y_true_up_slice, y_pred_up_slice),
                        1,
                    )
                    np.add.at(
                        temporal_confusions_down[window_index],
                        (y_true_down_slice, y_pred_down_slice),
                        1,
                    )
                    temporal_counts[window_index] += int(local_end - local_start)

            y_true_up_onehot = np.eye(num_classes, dtype="float64")[y_true_up]
            y_true_down_onehot = np.eye(num_classes, dtype="float64")[y_true_down]

            if calib_enabled:
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

            y_prob_up_cal: Optional[np.ndarray] = None
            y_prob_down_cal: Optional[np.ndarray] = None

            if post_hoc_enabled:
                if temperature_up is None or temperature_down is None:
                    raise ValueError("Post-hoc calibration is enabled but temperatures are missing")

                y_prob_up_cal = logits_to_calibrated_probs(
                    probs_to_logits_proxy(y_prob_up),
                    temperature_up,
                )
                y_prob_down_cal = logits_to_calibrated_probs(
                    probs_to_logits_proxy(y_prob_down),
                    temperature_down,
                )

                up_brier_sum_cal += float(
                    np.sum(np.sum((y_prob_up_cal - y_true_up_onehot) ** 2, axis=1))
                )
                down_brier_sum_cal += float(
                    np.sum(np.sum((y_prob_down_cal - y_true_down_onehot) ** 2, axis=1))
                )
                up_count_cal += int(y_true_up.shape[0])
                down_count_cal += int(y_true_down.shape[0])

                up_bins_cal = _assign_calibration_bins_with_truth(
                    y_prob_up_cal,
                    y_true_up_onehot,
                    bin_edges,
                )
                down_bins_cal = _assign_calibration_bins_with_truth(
                    y_prob_down_cal,
                    y_true_down_onehot,
                    bin_edges,
                )

                up_conf_sum_cal += up_bins_cal[0]
                up_acc_sum_cal += up_bins_cal[1]
                up_bin_count_cal += up_bins_cal[2]

                down_conf_sum_cal += down_bins_cal[0]
                down_acc_sum_cal += down_bins_cal[1]
                down_bin_count_cal += down_bins_cal[2]

            if backtest_enabled:
                if post_hoc_enabled:
                    if y_prob_up_cal is None or y_prob_down_cal is None:
                        raise ValueError("Backtesting requires calibrated probabilities but none were produced")
                    backtest_prob_up_parts.append(y_prob_up_cal)
                    backtest_prob_down_parts.append(y_prob_down_cal)
                else:
                    backtest_prob_up_parts.append(y_prob_up)
                    backtest_prob_down_parts.append(y_prob_down)

            sample_offset += int(y_true_up.shape[0])
            eval_batches_done += 1
            if writer is not None:
                try:
                    writer.update_eval_progress(processed=eval_batches_done, total=total_eval_batches)
                except Exception:
                    pass

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
    calibration_results_up_cal = None
    calibration_results_down_cal = None
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
    if post_hoc_enabled:
        calibration_results_up_cal = _finalize_calibration(
            bin_edges,
            up_conf_sum_cal,
            up_acc_sum_cal,
            up_bin_count_cal,
            up_brier_sum_cal,
            up_count_cal,
        )
        calibration_results_down_cal = _finalize_calibration(
            bin_edges,
            down_conf_sum_cal,
            down_acc_sum_cal,
            down_bin_count_cal,
            down_brier_sum_cal,
            down_count_cal,
        )

    if backtest_enabled:
        if backtest_prices is None or backtest_timestamps is None:
            raise ValueError("Backtesting is enabled but price data is missing")
        if not backtest_prob_up_parts or not backtest_prob_down_parts:
            raise ValueError("Backtesting is enabled but no probabilities were collected")

        backtest_prob_up = np.concatenate(backtest_prob_up_parts, axis=0)
        backtest_prob_down = np.concatenate(backtest_prob_down_parts, axis=0)

        expected_len = backtest_prices.shape[0]
        if backtest_prob_up.shape[0] != expected_len or backtest_prob_down.shape[0] != expected_len:
            raise ValueError(
                "Backtesting probability length mismatch: "
                f"up={backtest_prob_up.shape[0]}, down={backtest_prob_down.shape[0]}, expected={expected_len}"
            )

        from .backtesting import log_backtest_to_mlflow, run_backtest

        backtest_result = run_backtest(
            config=config,
            y_prob_up=backtest_prob_up,
            y_prob_down=backtest_prob_down,
            prices=backtest_prices,
            horizon_steps=backtest_horizon_steps,
            timestamps=backtest_timestamps,
        )
        log_backtest_to_mlflow(backtest_result)

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
    if calibration_results_up_cal is not None:
        metrics["eval_up_brier_score_calibrated"] = float(calibration_results_up_cal["brier_score"])
        metrics["eval_up_ece_calibrated"] = float(calibration_results_up_cal["ece"])
    if calibration_results_down_cal is not None:
        metrics["eval_down_brier_score_calibrated"] = float(calibration_results_down_cal["brier_score"])
        metrics["eval_down_ece_calibrated"] = float(calibration_results_down_cal["ece"])

    if calibration_fit_summary is not None:
        up_summary = calibration_fit_summary.get("up", {})
        down_summary = calibration_fit_summary.get("down", {})
        metrics["calibration_temperature_up"] = float(up_summary.get("temperature", 0.0))
        metrics["calibration_temperature_down"] = float(down_summary.get("temperature", 0.0))
        metrics["calibration_fit_up_ece_pre"] = float(up_summary.get("pre_calibration_ece", 0.0))
        metrics["calibration_fit_up_ece_post"] = float(up_summary.get("post_calibration_ece", 0.0))
        metrics["calibration_fit_down_ece_pre"] = float(down_summary.get("pre_calibration_ece", 0.0))
        metrics["calibration_fit_down_ece_post"] = float(down_summary.get("post_calibration_ece", 0.0))
        metrics["calibration_fit_samples"] = float(calibration_fit_summary.get("num_samples", 0))

    for name, value in metrics.items():
        try:
            mlflow.log_metric(name, float(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log MLFlow evaluation metric %s: %s", name, exc)

    # Optional confusion matrix artifact logging.
    try:
        mlflow_cfg = config["mlflow"]
        artifact_logging_cfg = mlflow_cfg["artifact_logging"]
        log_confusion = bool(artifact_logging_cfg["confusion_matrix"])
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

    if (
        (calib_enabled and (calibration_results_up is not None or calibration_results_down is not None))
        or (
            post_hoc_enabled
            and (calibration_results_up_cal is not None or calibration_results_down_cal is not None)
        )
    ):
        try:
            tmp_dir = Path(tempfile.mkdtemp())

            if calib_enabled and calibration_results_up is not None:
                calib_up_path = tmp_dir / "calibration_curve_up.csv"
                _write_calibration_curve(calibration_results_up, calib_up_path)
                mlflow.log_artifact(str(calib_up_path), artifact_path="evaluation")

            if calib_enabled and calibration_results_down is not None:
                calib_down_path = tmp_dir / "calibration_curve_down.csv"
                _write_calibration_curve(calibration_results_down, calib_down_path)
                mlflow.log_artifact(str(calib_down_path), artifact_path="evaluation")

            if post_hoc_enabled and calibration_results_up_cal is not None:
                calib_up_cal_path = tmp_dir / "calibration_curve_up_calibrated.csv"
                _write_calibration_curve(calibration_results_up_cal, calib_up_cal_path)
                mlflow.log_artifact(str(calib_up_cal_path), artifact_path="evaluation")

            if post_hoc_enabled and calibration_results_down_cal is not None:
                calib_down_cal_path = tmp_dir / "calibration_curve_down_calibrated.csv"
                _write_calibration_curve(calibration_results_down_cal, calib_down_cal_path)
                mlflow.log_artifact(str(calib_down_cal_path), artifact_path="evaluation")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log calibration curve artifacts to MLFlow: %s", exc)

    # Temporal degradation analysis
    if temporal_enabled:
        try:
            if not temporal_windows:
                logger.warning("Temporal degradation enabled but no windows were computed; skipping analysis.")
            else:
                expected_samples = test_end - test_start
                if sample_offset != expected_samples:
                    logger.warning(
                        "Temporal degradation processed %s samples but expected %s.",
                        sample_offset,
                        expected_samples,
                    )

                def _build_window_metrics(
                    confusion: np.ndarray,
                    sample_count: int,
                    window_index: int,
                    start_index: int,
                    end_index: int,
                ) -> WindowMetrics:
                    if sample_count <= 0:
                        return WindowMetrics(
                            window_index=window_index,
                            start_index=start_index,
                            end_index=end_index,
                            accuracy=0.0,
                            precision_macro=0.0,
                            recall_macro=0.0,
                            f1_macro=0.0,
                            num_samples=0,
                            per_class_accuracy=[],
                        )

                    accuracy = float(np.trace(confusion) / float(sample_count))
                    per_class_precision, per_class_recall, per_class_f1 = _compute_class_metrics(confusion)

                    precision_macro = float(np.mean(per_class_precision)) if per_class_precision else 0.0
                    recall_macro = float(np.mean(per_class_recall)) if per_class_recall else 0.0
                    f1_macro = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

                    per_class_accuracy = []
                    for cls in range(num_classes):
                        class_total = float(confusion[cls, :].sum())
                        class_acc = float(confusion[cls, cls] / class_total) if class_total > 0 else 0.0
                        per_class_accuracy.append(class_acc)

                    return WindowMetrics(
                        window_index=window_index,
                        start_index=start_index,
                        end_index=end_index,
                        accuracy=accuracy,
                        precision_macro=precision_macro,
                        recall_macro=recall_macro,
                        f1_macro=f1_macro,
                        num_samples=sample_count,
                        per_class_accuracy=per_class_accuracy,
                    )

                def _build_temporal_result(
                    window_metrics: list[WindowMetrics],
                ) -> TemporalDegradationResult:
                    if not window_metrics:
                        return TemporalDegradationResult(
                            window_metrics=[],
                            overall_trend=0.0,
                            degradation_rate=0.0,
                            first_window_accuracy=0.0,
                            last_window_accuracy=0.0,
                            total_degradation=0.0,
                        )

                    accuracies = [w.accuracy for w in window_metrics]
                    first_acc = accuracies[0]
                    last_acc = accuracies[-1]
                    total_degradation = first_acc - last_acc

                    if len(accuracies) > 1:
                        x = np.arange(len(accuracies), dtype="float64")
                        y = np.array(accuracies, dtype="float64")

                        x_mean = np.mean(x)
                        y_mean = np.mean(y)
                        numerator = np.sum((x - x_mean) * (y - y_mean))
                        denominator = np.sum((x - x_mean) ** 2)

                        slope = float(numerator / denominator) if denominator > 0 else 0.0
                        degradation_rate = -slope
                    else:
                        slope = 0.0
                        degradation_rate = 0.0

                    return TemporalDegradationResult(
                        window_metrics=window_metrics,
                        overall_trend=slope,
                        degradation_rate=degradation_rate,
                        first_window_accuracy=first_acc,
                        last_window_accuracy=last_acc,
                        total_degradation=total_degradation,
                    )

                window_metrics_up = []
                window_metrics_down = []

                for window_index, (win_start, win_end) in enumerate(temporal_windows):
                    window_metrics_up.append(
                        _build_window_metrics(
                            temporal_confusions_up[window_index],
                            temporal_counts[window_index],
                            window_index,
                            win_start,
                            win_end,
                        )
                    )
                    window_metrics_down.append(
                        _build_window_metrics(
                            temporal_confusions_down[window_index],
                            temporal_counts[window_index],
                            window_index,
                            win_start,
                            win_end,
                        )
                    )

                temporal_result_up = _build_temporal_result(window_metrics_up)
                temporal_result_down = _build_temporal_result(window_metrics_down)

                # Log summary metrics to MLflow
                mlflow.log_metric("temporal_up_total_degradation", temporal_result_up.total_degradation)
                mlflow.log_metric("temporal_up_degradation_rate", temporal_result_up.degradation_rate)
                mlflow.log_metric("temporal_up_overall_trend", temporal_result_up.overall_trend)
                mlflow.log_metric("temporal_up_first_window_accuracy", temporal_result_up.first_window_accuracy)
                mlflow.log_metric("temporal_up_last_window_accuracy", temporal_result_up.last_window_accuracy)

                mlflow.log_metric("temporal_down_total_degradation", temporal_result_down.total_degradation)
                mlflow.log_metric("temporal_down_degradation_rate", temporal_result_down.degradation_rate)
                mlflow.log_metric("temporal_down_overall_trend", temporal_result_down.overall_trend)
                mlflow.log_metric("temporal_down_first_window_accuracy", temporal_result_down.first_window_accuracy)
                mlflow.log_metric("temporal_down_last_window_accuracy", temporal_result_down.last_window_accuracy)

                # Log per-window metrics if configured
                if temporal_log_per_window:
                    for wm in temporal_result_up.window_metrics:
                        prefix = f"temporal_up_window_{wm.window_index}"
                        mlflow.log_metric(f"{prefix}_accuracy", wm.accuracy)
                        mlflow.log_metric(f"{prefix}_f1_macro", wm.f1_macro)
                        mlflow.log_metric(f"{prefix}_num_samples", float(wm.num_samples))

                    for wm in temporal_result_down.window_metrics:
                        prefix = f"temporal_down_window_{wm.window_index}"
                        mlflow.log_metric(f"{prefix}_accuracy", wm.accuracy)
                        mlflow.log_metric(f"{prefix}_f1_macro", wm.f1_macro)
                        mlflow.log_metric(f"{prefix}_num_samples", float(wm.num_samples))

                # Save full results as JSON artifact
                tmp_dir = Path(tempfile.mkdtemp())
                temporal_path = tmp_dir / "temporal_degradation_analysis.json"
                temporal_artifact = {
                    "up": temporal_result_up.to_dict(),
                    "down": temporal_result_down.to_dict(),
                    "config": {
                        "num_windows": temporal_num_windows,
                        "overlap_fraction": temporal_overlap,
                        "total_samples": expected_samples,
                    },
                }
                with temporal_path.open("w", encoding="utf-8") as f:
                    json.dump(temporal_artifact, f, indent=2, sort_keys=True)
                mlflow.log_artifact(str(temporal_path), artifact_path="evaluation")

                logger.info(
                    "Temporal degradation analysis complete. Up: degradation=%.4f, Down: degradation=%.4f",
                    temporal_result_up.total_degradation,
                    temporal_result_down.total_degradation,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Temporal degradation analysis failed: %s", exc)


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
