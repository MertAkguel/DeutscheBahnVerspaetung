"""Monthly monitoring run: drift + champion performance -> decision, all logged to MLflow.

Prints "retrain" or "skip" as its last line, so the Airflow DAG can choose the next task.

Usage:
    uv run python -m db_delay_mlops.monitoring.monitor --month 2026-10
"""

import argparse
import logging
import os

import mlflow
import polars as pl
from mlflow.exceptions import MlflowException

from db_delay_mlops.monitoring.decide import Decision, decide
from db_delay_mlops.monitoring.drift import drift_table, load_current, load_reference
from db_delay_mlops.monitoring.performance import champion_performance
from db_delay_mlops.monitoring.served import served_performance

logger = logging.getLogger(__name__)

EXPERIMENT = "db-delay-monitoring"


def report(month: str, performance: dict, drift: pl.DataFrame, decision: Decision, served: dict | None = None) -> str:
    """A short Markdown report, saved next to the metrics in MLflow."""
    lines = [
        f"# Monitoring {month}",
        "",
        f"**Decision: {'retrain' if decision.retrain else 'keep the champion'}**",
        *[f"- reason: {r}" for r in decision.reasons],
        *[f"- warning: {w}" for w in decision.warnings],
        "",
        f"Champion v{performance['champion_version']} (validated on {performance['champion_month']})",
        "",
        "| metric | at registration | this month |",
        "| --- | --- | --- |",
    ]
    for name, key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("late share", "late_share")]:
        then = performance[f"baseline_{key}"]
        lines.append(f"| {name} | {then:.4f} | {performance[key]:.4f} |" if then is not None else f"| {name} | - |")
    lines += ["", "## Answers the API actually gave (logged predictions)", ""]
    if served is None:
        lines.append("No prediction was requested for this month.")
    else:
        lines += [f"- {k.removeprefix('served_')}: {v:.4f}" for k, v in served.items()]
    lines += ["", "## Data drift", "", "| feature | PSI | drift |", "| --- | --- | --- |"]
    lines += [f"| {r['feature']} | {r['psi']:.3f} | {r['level']} |" for r in drift.iter_rows(named=True)]
    return "\n".join(lines)


def monitor(month: str) -> Decision:
    drift = drift_table(load_reference(month), load_current(month))
    try:
        performance = champion_performance(month)
    except MlflowException:  # no @champion yet: the very first run must train one
        logger.info("no champion yet: retrain")
        return Decision(retrain=True, reasons=["there is no champion yet"])
    decision = decide(month, performance, drift)
    served = served_performance(month)  # None if nobody asked the API about this month

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor-{month}"):
        mlflow.set_tags(
            {
                "month": month,
                "champion_version": performance["champion_version"],
                "decision": "retrain" if decision.retrain else "skip",
            }
        )
        numbers = {k: v for k, v in performance.items() if isinstance(v, float)}
        mlflow.log_metrics(numbers | {f"psi_{r['feature']}": r["psi"] for r in drift.iter_rows(named=True)})
        mlflow.log_metric("n_strong_drift", int((drift["level"] == "strong").sum()))
        if served:
            mlflow.log_metrics(served)
        mlflow.log_text(report(month, performance, drift, decision, served), "report.md")
        mlflow.log_text(drift.write_csv(), "drift.csv")

    for reason in decision.reasons:
        logger.info("retrain: %s", reason)
    for warning in decision.warnings:
        logger.warning(warning)
    logger.info("%s: %s", month, "retrain" if decision.retrain else "champion stays, no retraining")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", required=True, help="YYYY-MM, a built and validated month")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    decision = monitor(args.month)
    print("retrain" if decision.retrain else "skip")  # last line: read by the Airflow DAG


if __name__ == "__main__":
    main()
