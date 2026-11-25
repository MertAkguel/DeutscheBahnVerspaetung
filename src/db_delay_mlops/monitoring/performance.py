"""How good is the current champion on the new month, compared with when it was registered?

Once a month is over, we know which trains were late. So the model in production can be
scored on real outcomes, the most direct sign of whether it still works.
"""

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from db_delay_mlops.features.build import OUT_DIR
from db_delay_mlops.models.evaluate import evaluate
from db_delay_mlops.models.register import MODEL_NAME
from db_delay_mlops.models.train import load_months


def champion_performance(month: str, data_dir=OUT_DIR) -> dict:
    """Metrics of @champion on `month`, plus its metrics from training for comparison."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")

    X, y = load_months([month], data_dir)
    p = model.predict_proba(X)[:, 1]
    now = evaluate(y, p)

    # What the champion achieved on its own validation month, logged by models/train.py
    then = client.get_run(version.run_id).data.metrics if version.run_id else {}
    return {
        "champion_version": str(version.version),
        "champion_month": version.tags.get("valid_month"),
        "roc_auc": now["roc_auc"],
        "pr_auc": now["pr_auc"],
        "brier": now["brier"],
        "late_share": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "baseline_roc_auc": then.get("roc_auc"),
        "baseline_pr_auc": then.get("pr_auc"),
        "baseline_late_share": then.get("valid_late_share"),
    }
