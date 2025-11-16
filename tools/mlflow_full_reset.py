# file: mlflow_full_reset.py
#
# WARNING: This will delete ALL experiments and ALL runs from the MLflow tracking server
# at MLFLOW_URI. Use DRY_RUN=True first to inspect before actually deleting.

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow

MLFLOW_URI = "http://192.168.1.11:5501"  # adjust if needed
DRY_RUN = True  # set to False to actually delete


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiments = client.search_experiments(view_type=ViewType.ALL)
    print(f"Found {len(experiments)} experiments on {MLFLOW_URI}:")

    for exp in experiments:
        print(
            f"- id={exp.experiment_id}, "
            f"name={exp.name!r}, "
            f"lifecycle={exp.lifecycle_stage}, "
            f"artifact_location={exp.artifact_location}"
        )

        # List all runs in this experiment (active + deleted)
        run_infos = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            view_type=ViewType.ALL,
        )
        print(f"  -> {len(run_infos)} runs")

        if DRY_RUN:
            continue

        # Delete all runs
        for ri in run_infos:
            print(f"    Deleting run {ri.run_id} (status={ri.status})")
            client.delete_run(ri.run_id)

        # Delete the experiment itself
        print(f"  Deleting experiment {exp.experiment_id} ({exp.name!r})")
        client.delete_experiment(exp.experiment_id)

    if DRY_RUN:
        print("\nDRY_RUN=True: nothing was deleted. "
              "If this list looks correct, set DRY_RUN=False and run again.")
    else:
        print("\nAll experiments and runs have been marked deleted on the server.")


if __name__ == "__main__":
    main()
