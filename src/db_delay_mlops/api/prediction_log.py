"""D6: remember every prediction the API made, to check it against reality later.

One JSON line per prediction, one file per day: data/predictions/predictions-YYYY-MM-DD.jsonl
Once the month is over, monitoring/served.py joins these lines with the real arrivals and
measures how good the answers were that users actually got.

JSON Lines (one JSON object per line) instead of one big JSON file: a new prediction is
simply appended at the end, nothing has to be read or rewritten, and Polars reads it directly.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from db_delay_mlops.api.schemas import ArrivalRequest, Prediction
from db_delay_mlops.paths import DATA_DIR

logger = logging.getLogger(__name__)


def log_dir() -> Path:
    """Where the logs go; the environment variable lets tests and containers choose another folder."""
    return Path(os.environ.get("PREDICTION_LOG_DIR", DATA_DIR / "predictions"))


def log_predictions(arrivals: list[ArrivalRequest], predictions: list[Prediction], model_version: str) -> None:
    """Append one line per prediction. Never fails the request: a missing log is bad, a missing answer worse."""
    now = datetime.now()
    rows = [
        {
            "logged_at": now.isoformat(timespec="seconds"),
            "model_version": model_version,
            **arrival.model_dump(mode="json"),  # mode="json": datetimes become text like "2026-11-25T17:42:00"
            "probability_late": prediction.probability_late,
            "risk": prediction.risk,
            "n_warnings": len(prediction.warnings),
        }
        for arrival, prediction in zip(arrivals, predictions, strict=True)
    ]
    try:
        path = log_dir() / f"predictions-{now:%Y-%m-%d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:  # "a" = append at the end
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    except OSError as error:
        logger.warning("could not log %d predictions: %s", len(rows), error)
