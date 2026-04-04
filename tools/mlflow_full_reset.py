# file: mlflow_full_reset.py
"""Utility to purge MLflow experiments and runs (use with caution)."""

# WARNING: This will delete ALL experiments and ALL runs from the MLflow tracking server
# at MLFLOW_URI. Use DRY_RUN=True first to inspect before actually deleting.

import logging
import os

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DRY_RUN = True  # set to False to actually delete

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiments = client.search_experiments(view_type=ViewType.ALL)
    logger.info("Found %s experiments on %s:", len(experiments), MLFLOW_URI)

    for exp in experiments:
        logger.info(
            "- id=%s, name=%r, lifecycle=%s, artifact_location=%s",
            exp.experiment_id,
            exp.name,
            exp.lifecycle_stage,
            exp.artifact_location,
        )

        # List all runs in this experiment (active + deleted)
        run_infos = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            run_view_type=ViewType.ALL,
        )
        logger.info("  -> %s runs", len(run_infos))

        if DRY_RUN:
            continue

        # Delete all runs
        for run in run_infos:
            run_id = run.info.run_id
            status = run.info.status
            logger.info("    Deleting run %s (status=%s)", run_id, status)
            client.delete_run(run_id)

        # Delete the experiment itself
        logger.info("  Deleting experiment %s (%r)", exp.experiment_id, exp.name)
        client.delete_experiment(exp.experiment_id)

    if DRY_RUN:
        logger.info(
            "DRY_RUN=True: nothing was deleted. If this list looks correct, set DRY_RUN=False and run again."
        )
    else:
        logger.info("All experiments and runs have been marked deleted on the server.")


if __name__ == "__main__":
    main()
