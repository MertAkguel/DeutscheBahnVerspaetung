"""Monthly training: new month of data -> features -> checks -> model -> quality gate.

Runs on the 3rd of each month for the month before (the run on 2026-12-03 handles 2026-11),
because a month is only complete once it is over.

Every task only calls one of our modules as a command, in the pipeline venv ($PIPELINE_PYTHON).
Airflow decides when and in which order; what happens is in src/db_delay_mlops/.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

PY = "$PIPELINE_PYTHON -m db_delay_mlops"


@dag(
    schedule="0 6 3 * *",  # cron: minute 0, hour 6 (UTC), day 3, every month, every weekday
    start_date=datetime(2026, 11, 1),
    catchup=False,  # do not run all past months automatically when the DAG is switched on
    max_active_runs=1,  # one training at a time: two would not fit into memory
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["training"],
)
def monthly_training():
    @task
    def pick_month(logical_date=None) -> str:
        """The month this run is for: the month before the logical date (2026-12-03 -> "2026-11")."""
        last_day_of_previous_month = logical_date.replace(day=1) - timedelta(days=1)
        return last_day_of_previous_month.strftime("%Y-%m")

    @task.bash(retries=3)  # downloads fail now and then (network), so try more often
    def download(month: str) -> str:
        return f"{PY}.data.download --start {month}"

    @task.bash
    def build_features(month: str) -> str:
        return f"{PY}.features.build --month {month}"

    @task.bash
    def validate(month: str) -> str:
        return f"{PY}.data.validate --month {month}"

    # Expensive and deterministic: a retry would only repeat the same hour
    @task.bash(retries=0, execution_timeout=timedelta(hours=3))
    def train(month: str) -> str:
        return f"{PY}.models.train --valid-month {month}"  # prints the run id as its last line

    # Not "run_id": Airflow already uses that name for its own run (scheduled__2026-11-03...)
    @task.bash
    def register(mlflow_run_id: str, month: str) -> str:
        return f"{PY}.models.register --run-id {mlflow_run_id} --valid-month {month}"

    month = pick_month()
    checked = download(month) >> build_features(month) >> validate(month)
    mlflow_run_id = train(month)
    checked >> mlflow_run_id
    register(mlflow_run_id, month)


monthly_training()
