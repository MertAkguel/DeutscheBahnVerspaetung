"""D6: how good were the answers the API actually gave? (predictions logged by api/prediction_log.py)

performance.py scores the champion on ALL arrivals of a month. This module only looks at
the arrivals someone asked about, with the probability they really got. Once the month is
over, each logged prediction is matched with the real arrival:
same station, same planned arrival time, same final destination.

Few requests -> few matches: the numbers are reported, not used for the retrain decision.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score

from db_delay_mlops.api.prediction_log import log_dir
from db_delay_mlops.features.build import OUT_DIR

KEYS = ["stada_name", "arrival_planned_time", "final_destination_station"]


def load_served(month: str, directory: Path | None = None) -> pl.DataFrame:
    """All logged predictions for arrivals planned in `month` (any request day)."""
    files = sorted((directory or log_dir()).glob("predictions-*.jsonl"))
    if not files:
        return pl.DataFrame()
    return (
        pl.concat([pl.read_ndjson(f) for f in files], how="diagonal_relaxed")
        .filter(pl.col("planned_arrival").str.starts_with(month))
        .select(
            stada_name="station",
            arrival_planned_time=pl.col("planned_arrival").str.to_datetime(time_unit="ns"),
            final_destination_station="final_destination",
            probability_late="probability_late",
            model_version="model_version",
        )
        # Asked twice for the same train (e.g. reloading the demo): count it once, the last answer
        .unique(subset=KEYS, keep="last")
    )


def served_performance(month: str, directory: Path | None = None, data_dir: Path = OUT_DIR) -> dict | None:
    """Metrics of the logged predictions against reality, or None if nothing was asked about this month."""
    served = load_served(month, directory)
    if served.is_empty():
        return None
    actual = pl.read_parquet(data_dir / f"arrivals-{month}.parquet", columns=[*KEYS, "y"]).unique(subset=KEYS)
    matched = served.join(actual, on=KEYS, how="inner")

    result = {"served_n": float(served.height), "served_matched": float(matched.height)}
    if matched.is_empty():
        return result
    p, y = matched["probability_late"].to_numpy(), matched["y"].to_numpy()
    result |= {
        "served_mean_prediction": float(p.mean()),
        "served_late_share": float(y.mean()),
        "served_brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) == 2:  # ROC-AUC needs late AND on-time trains among the matches
        result["served_roc_auc"] = float(roc_auc_score(y, p))
    return result
