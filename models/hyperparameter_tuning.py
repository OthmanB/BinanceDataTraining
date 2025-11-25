"""Hyperparameter optimization using Optuna.

This module defines the interface for running hyperparameter optimization
driven entirely by the ``hyperparameter_optimization`` section of the
configuration.
"""

from typing import Any, Dict, Optional
import copy
import logging


logger = logging.getLogger(__name__)


def _sample_hyperparameters(trial: Any, hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    search_space = hpo_cfg["search_space"]

    def _ensure_list(name: str) -> Any:
        if name not in search_space:
            raise ValueError(f"Missing hyperparameter_optimization.search_space entry for {name!r}")
        value = search_space[name]
        if not isinstance(value, list):
            raise ValueError(
                "Each hyperparameter_optimization.search_space entry must be a list; "
                f"got type={type(value).__name__} for {name!r}",
            )
        if not value:
            raise ValueError(f"hyperparameter_optimization.search_space.{name} must be a non-empty list")
        return value

    def _suggest_int(name: str, bounds: Any) -> int:
        if len(bounds) < 2:
            raise ValueError(f"Integer search space for {name!r} must have at least two elements [low, high]")
        low = int(bounds[0])
        high = int(bounds[1])
        return int(trial.suggest_int(name, low, high))

    def _suggest_float(name: str, bounds: Any) -> float:
        if len(bounds) < 2:
            raise ValueError(f"Float search space for {name!r} must have at least two elements [low, high]")
        low = float(bounds[0])
        high = float(bounds[1])
        log_scale = False
        if len(bounds) >= 3 and str(bounds[2]).lower() == "log":
            log_scale = True
        return float(trial.suggest_float(name, low, high, log=log_scale))

    params: Dict[str, Any] = {}

    cnn_filters_1_space = _ensure_list("cnn_filters_1")
    cnn_filters_2_space = _ensure_list("cnn_filters_2")
    lstm_units_space = _ensure_list("lstm_units")
    learning_rate_space = _ensure_list("learning_rate")
    batch_size_space = _ensure_list("batch_size")

    params["cnn_filters_1"] = _suggest_int("cnn_filters_1", cnn_filters_1_space)
    params["cnn_filters_2"] = _suggest_int("cnn_filters_2", cnn_filters_2_space)
    params["lstm_units"] = _suggest_int("lstm_units", lstm_units_space)
    params["learning_rate"] = _suggest_float("learning_rate", learning_rate_space)

    if len(batch_size_space) >= 3:
        choices = sorted({int(v) for v in batch_size_space})
        params["batch_size"] = int(trial.suggest_categorical("batch_size", choices))
    elif len(batch_size_space) == 2:
        params["batch_size"] = _suggest_int("batch_size", batch_size_space)
    else:
        raise ValueError("hyperparameter_optimization.search_space.batch_size must have at least two values")

    return params


def _apply_hyperparameters(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)

    model_cfg = cfg["model"]
    cnn_cfg = model_cfg["cnn"]
    lstm_cfg = model_cfg["lstm"]
    compilation_cfg = model_cfg["compilation"]
    training_cfg = cfg["training"]

    filters = list(cnn_cfg["filters"])
    if len(filters) < 2:
        raise ValueError("model.cnn.filters must have length at least 2 to apply cnn_filters_1 and cnn_filters_2")
    filters[0] = int(params["cnn_filters_1"])
    filters[1] = int(params["cnn_filters_2"])
    cnn_cfg["filters"] = filters

    lstm_cfg["units"] = int(params["lstm_units"])
    compilation_cfg["learning_rate"] = float(params["learning_rate"])
    training_cfg["batch_size"] = int(params["batch_size"])

    model_cfg["cnn"] = cnn_cfg
    model_cfg["lstm"] = lstm_cfg
    model_cfg["compilation"] = compilation_cfg
    cfg["model"] = model_cfg
    cfg["training"] = training_cfg

    return cfg


def run_hyperparameter_search(
    config: Dict[str, Any],
    data_object: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    hpo_cfg = config.get("hyperparameter_optimization", {})
    if not isinstance(hpo_cfg, dict) or not hpo_cfg.get("enabled"):
        logger.info("Hyperparameter optimization disabled in configuration.")
        return None

    framework = str(hpo_cfg.get("framework"))
    if framework != "optuna":
        logger.warning("Only hyperparameter_optimization.framework='optuna' is supported; got %s", framework)
        return None

    try:
        import optuna  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import Optuna for hyperparameter optimization: %s", exc)
        return None

    n_trials = int(hpo_cfg["n_trials"])
    direction = str(hpo_cfg["direction"])
    metric_name = str(hpo_cfg["metric"])

    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "hyperparameter_optimization.direction must be either 'minimize' or 'maximize'; "
            f"got {direction!r}",
        )

    if "search_space" not in hpo_cfg or not isinstance(hpo_cfg["search_space"], dict):
        raise ValueError("hyperparameter_optimization.search_space must be a dict in configuration")

    def objective(trial: Any) -> float:
        params = _sample_hyperparameters(trial, hpo_cfg)
        trial_config = _apply_hyperparameters(config, params)

        from training.pipeline import run_training_pipeline

        run_training_pipeline(trial_config, data_object)

        metadata = data_object.get("metadata", {})
        if "last_hpo_metric" not in metadata:
            raise ValueError(
                "Training pipeline did not populate metadata.last_hpo_metric; "
                "ensure hyperparameter_optimization.metric matches a key in the Keras History.",
            )

        value = float(metadata["last_hpo_metric"])
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("metric_value", value)
        return value

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)

    best_config = _apply_hyperparameters(config, best_params)

    try:
        import mlflow  # type: ignore[import]
    except Exception:  # noqa: BLE001
        pass
    else:
        try:
            mlflow.log_param("hpo_enabled", True)
            mlflow.log_param("hpo_framework", framework)
            mlflow.log_param("hpo_n_trials", n_trials)
            mlflow.log_param("hpo_direction", direction)
            mlflow.log_param("hpo_metric", metric_name)
            for name, value in best_params.items():
                mlflow.log_param(f"hpo_best_{name}", value)
            mlflow.log_metric("hpo_best_value", float(best_trial.value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log hyperparameter optimization results to MLFlow: %s", exc)

    logger.info(
        "Hyperparameter optimization completed: best_value=%s, best_params=%s",
        best_trial.value,
        best_params,
    )

    return best_config


__all__ = ["run_hyperparameter_search"]
