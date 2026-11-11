"""Quality gate: register a newly trained model as @champion only if it beats the current one.

Fair comparison: the new model was evaluated on `valid_month`. The current champion is
evaluated again on the SAME month, so both are judged on the same data. A score from
another month says little, because the late share and the conditions change.

Aliases in the registry:
- @champion: the model the API serves
- @previous: the champion before it, to roll back quickly if something goes wrong

Usage:
    uv run python -m db_delay_mlops.models.register --run-id <run id from train.py> --valid-month 2026-09
"""

import argparse
import logging
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from db_delay_mlops.models.evaluate import evaluate
from db_delay_mlops.models.train import load_months

logger = logging.getLogger(__name__)

MODEL_NAME = "db-delay-lightgbm"
MIN_IMPROVEMENT = 0.0  # PR-AUC points the new model must be better by; 0.0 = any improvement


def champion_pr_auc(client: MlflowClient, valid_month: str) -> tuple[float | None, str | None]:
    """PR-AUC of the current champion on valid_month, or (None, None) if there is no champion yet."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None  # first run ever: nothing to compare with

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    X_valid, y_valid = load_months([valid_month])
    p_valid = model.predict_proba(X_valid)[:, 1]
    return evaluate(y_valid, p_valid)["pr_auc"], champion.version


def model_uri_of(client: MlflowClient, run) -> str:
    """Where the model of this run is stored (models:/m-...).

    Runs from train.py carry it as the tag model_uri. Older runs (logged before the tag
    existed) are looked up: MLflow 3 remembers which run a logged model came from.
    """
    if "model_uri" in run.data.tags:
        return run.data.tags["model_uri"]
    models = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
    )
    if not models:
        raise ValueError(f"run {run.info.run_id} has no logged model")
    return models[0].model_uri


def register_if_better(run_id: str, valid_month: str) -> dict:
    """Compare the model of run_id with the champion on valid_month; register and move the alias if better."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    run = client.get_run(run_id)
    new_pr_auc = run.data.metrics["pr_auc"]
    model_uri = model_uri_of(client, run)
    old_pr_auc, old_version = champion_pr_auc(client, valid_month)
    promoted = old_pr_auc is None or new_pr_auc > old_pr_auc + MIN_IMPROVEMENT

    result = {
        "valid_month": valid_month,
        "new_pr_auc": new_pr_auc,
        "champion_pr_auc": old_pr_auc,
        "champion_version": old_version,
        "promoted": promoted,
    }

    # Write the decision into the new run, so it is visible in MLflow next to the metrics
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("gate_decision", "promoted" if promoted else "rejected")
        if old_pr_auc is not None:
            mlflow.log_metric("champion_pr_auc_same_month", old_pr_auc)

    if promoted:
        version = mlflow.register_model(model_uri, MODEL_NAME).version
        if old_version is not None:
            client.set_registered_model_alias(MODEL_NAME, "previous", old_version)
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        client.set_model_version_tag(MODEL_NAME, version, "valid_month", valid_month)
        client.set_model_version_tag(MODEL_NAME, version, "valid_pr_auc", f"{new_pr_auc:.4f}")
        result["new_version"] = version
        logger.info("promoted: version %s is @champion (pr_auc %.4f vs. %s)", version, new_pr_auc, old_pr_auc)
    else:
        logger.info("rejected: new pr_auc %.4f is not better than champion %.4f, champion stays", new_pr_auc, old_pr_auc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="MLflow run of the new model (printed by models/train.py)")
    parser.add_argument("--valid-month", required=True, help="YYYY-MM, the same month the new model was validated on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    register_if_better(args.run_id, args.valid_month)


if __name__ == "__main__":
    main()
