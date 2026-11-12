"""Train the monthly model: LightGBM on a rolling window of months, logged to MLflow.

Three parts, oldest first:
- train months: the `n_train_months` months before the stop month, to fit the trees
- stop month: the month before `valid_month`, only for early stopping (how many trees)
- valid_month: never seen during training, only for the metrics and the comparison
  with the champion. So the new model and the champion are both judged on unseen data.
Example: valid_month = 2026-09, n_train_months = 11
    -> train 2025-09 ... 2026-07, stop 2026-08, valid 2026-09.

The model is saved in MLflow but NOT registered: whether it becomes the new champion
is decided in the next step (models/register.py).

Usage:
    uv run python -m db_delay_mlops.models.train --valid-month 2026-09
"""

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from db_delay_mlops.features.build import OUT_DIR
from db_delay_mlops.features.columns import FEATURES, TARGET, TEXT_FEATURES
from db_delay_mlops.models.evaluate import evaluate
from db_delay_mlops.models.preprocessing import CategoryCaster

logger = logging.getLogger(__name__)

EXPERIMENT = "db-delay-monthly"
N_TRAIN_MONTHS = 11  # + 1 stop month = 12 months before valid_month
EARLY_STOPPING_ROUNDS = 100
SEED = 42

# Settings from the Optuna search in notebooks/06_tuning_and_champion.ipynb (trial 28).
# Tuning is repeated by hand from time to time, not every month.
# Changed for the monthly run: the search used 3 weeks of data (1.1 M rows), the monthly
# window has 12 months (about 17 M rows). With the tuned learning rate of 0.0336 early stopping
# needed thousands of trees and the run took over an hour. A larger learning rate needs far fewer
# trees, and with this much data it costs little quality. The limit of 2,000 trees is a safety net.
LGBM_PARAMS = {
    "n_estimators": 2000,  # upper limit, early stopping decides
    "learning_rate": 0.1,  # tuned value on 3 weeks of data: 0.0336
    "num_leaves": 238,
    "min_child_samples": 56,
    "colsample_bytree": 0.7742,
    "subsample": 0.5511,
    "subsample_freq": 1,
    "reg_lambda": 3.5918,
    # Same data, same seed -> same model: fixed order of the parallel sums (a bit slower)
    "deterministic": True,
    "force_row_wise": True,
}
LOG_EVERY_N_TREES = 100  # progress line in the log, so a long run is visibly alive

# Types skops may load when the model is read again (see docs/baselines-und-vorverarbeitung.md)
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "db_delay_mlops.models.preprocessing.CategoryCaster",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


def previous_months(month: str, n: int) -> list[str]:
    """The n months before `month`, oldest first: previous_months("2026-03", 3) -> ["2025-12", "2026-01", "2026-02"]."""
    year, mon = map(int, month.split("-"))
    months = []
    for _ in range(n):
        year, mon = (year - 1, 12) if mon == 1 else (year, mon - 1)
        months.append(f"{year}-{mon:02d}")
    return months[::-1]


def load_months(months: list[str], data_dir: Path = OUT_DIR):
    """Features and target of the given months, ready for scikit-learn (pandas).

    Text columns become categories already in Polars: in pandas, 17 million strings
    would need several GB, categories only a small number per row.
    """
    paths = [data_dir / f"arrivals-{m}.parquet" for m in months]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"build these months first (features/build.py): {missing}")

    df = (
        pl.scan_parquet(paths)
        .select(*FEATURES, TARGET)  # only what the model needs
        .with_columns(pl.col(TEXT_FEATURES).cast(pl.Categorical))
        .collect()
    )
    return df.select(FEATURES).to_pandas(), df[TARGET].to_numpy()


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("categories", CategoryCaster(columns=TEXT_FEATURES)),  # text -> category (learned on train)
        ("model", LGBMClassifier(random_state=SEED, verbose=-1, **LGBM_PARAMS)),
    ])


def train(valid_month: str, n_train_months: int = N_TRAIN_MONTHS, data_dir: Path = OUT_DIR) -> dict:
    """Train on the months before valid_month, evaluate on valid_month, log one MLflow run.

    Returns run_id and metrics, so the next step knows which model to compare.
    """
    stop_month = previous_months(valid_month, 1)[0]
    train_months = previous_months(stop_month, n_train_months)
    X_train, y_train = load_months(train_months, data_dir)
    X_stop, y_stop = load_months([stop_month], data_dir)
    X_valid, y_valid = load_months([valid_month], data_dir)
    logger.info(
        "train %s .. %s: %d rows | stop %s: %d rows | valid %s: %d rows",
        train_months[0], train_months[-1], len(X_train), stop_month, len(X_stop), valid_month, len(X_valid),
    )

    pipeline = make_pipeline()
    # Early stopping needs the stop month already prepared, with the categories of train
    X_stop_prepared = CategoryCaster(columns=TEXT_FEATURES).fit(X_train).transform(X_stop)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{valid_month}") as run:
        start = time.perf_counter()
        pipeline.fit(
            X_train,
            y_train,
            model__eval_X=X_stop_prepared,  # early stopping looks at the stop month, never at valid
            model__eval_y=y_stop,
            model__callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(LOG_EVERY_N_TREES),  # e.g. "[300] valid_0's binary_logloss: 0.4123"
            ],
        )
        fit_seconds = time.perf_counter() - start

        p_valid = pipeline.predict_proba(X_valid)[:, 1]
        metrics = evaluate(y_valid, p_valid) | {
            "fit_seconds": fit_seconds,
            "best_iteration": pipeline[-1].best_iteration_,
            "valid_late_share": float(y_valid.mean()),
        }

        mlflow.set_tags({"stage": "monthly", "valid_month": valid_month})
        mlflow.log_params({
            "model": "LGBMClassifier",
            "train_months": f"{train_months[0]}..{train_months[-1]}",
            "n_train_months": n_train_months,
            "stop_month": stop_month,
            "valid_month": valid_month,
            "n_train": len(X_train),
            "n_stop": len(X_stop),
            "n_valid": len(X_valid),
            "features": ",".join(FEATURES),
            "trained_on": date.today().isoformat(),
        })
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            # The API will send plain text, not pandas categories: the example (and so the signature) must match that
            input_example=X_valid.head(5).astype({col: object for col in TEXT_FEATURES}),
            skops_trusted_types=TRUSTED_TYPES,
            pyfunc_predict_fn="predict_proba",  # the generic MLflow interface returns probabilities, not 0/1
        )
        # MLflow 3 stores models as their own objects (models:/m-...), not inside the run's folder.
        # The tag tells the registration step where the model of this run is.
        mlflow.set_tag("model_uri", model_info.model_uri)

    logger.info("run %s: %d trees, pr_auc=%.4f, brier=%.4f", run.info.run_id, metrics["best_iteration"], metrics["pr_auc"], metrics["brier"])
    return {"run_id": run.info.run_id, "model_uri": model_info.model_uri, "valid_month": valid_month, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--valid-month", required=True, help="YYYY-MM; the month before it stops training, the ones before that train")
    parser.add_argument("--n-train-months", type=int, default=N_TRAIN_MONTHS)
    parser.add_argument("--data-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train(args.valid_month, args.n_train_months, args.data_dir)
    # Last line on stdout = run id. Airflow's bash task passes the last output line to the next task.
    print(result["run_id"])


if __name__ == "__main__":
    main()
