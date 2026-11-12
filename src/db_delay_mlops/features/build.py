"""Build the model table for one month: row rules, target and all features.

The same logic as notebooks/04_build_dataset.ipynb, but for one month at a time,
so Airflow can run it per month and backfill past months.

Usage:
    uv run python -m db_delay_mlops.features.build --month 2026-09
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from db_delay_mlops.features.columns import FEATURES, TARGET
from db_delay_mlops.features.holidays import add_is_holiday
from db_delay_mlops.features.weather import add_weather
from db_delay_mlops.paths import DATA_DIR, STATIONS_FILE, TRAIN_TYPE_GROUPS_FILE

logger = logging.getLogger(__name__)

HF_DIR = DATA_DIR / "raw" / "hf" / "monthly_processed_data"
OUT_DIR = DATA_DIR / "processed" / "arrivals"

RAIL_GROUPS = ["long_distance", "regional", "s_bahn"]
RAW_COLUMNS = [
    "id",
    "eva",
    "train_type",
    "line_number",
    "final_destination_station",
    "train_line_station_num",
    "has_plan_record",
    "arrival_planned_time",
    "arrival_change_time",
    "arrival_is_canceled",
]
METADATA = ["id", "eva", "arrival_planned_time", "ride_start"]  # kept for joins and analysis, not features


def scan_arrivals(month: str) -> pl.LazyFrame:
    """Plan for one monthly file: only needed columns and the row rules. Nothing is read yet."""
    stations = pl.read_csv(STATIONS_FILE, schema_overrides={"eva": pl.String})
    rail_evas = stations.filter(pl.col("collect"))["eva"]
    groups = pl.read_csv(TRAIN_TYPE_GROUPS_FILE).select("train_type", train_group="group")

    return (
        pl.scan_parquet(HF_DIR / f"data-{month}.parquet")
        .select(RAW_COLUMNS)
        .filter(
            pl.col("eva").is_in(rail_evas.implode()),  # only our 134 rail EVA numbers
            pl.col("arrival_planned_time").is_not_null(),  # not the first stop of a ride
            ~pl.col("arrival_is_canceled"),  # cancelled arrivals are excluded
            pl.col("has_plan_record"),  # only stops that are in the timetable
        )
        .join(groups.lazy(), on="train_type", how="left")
        .filter(pl.col("train_group").is_in(RAIL_GROUPS))  # buses and unmapped train types out
        .join(stations.lazy().select("eva", "stada_name"), on="eva", how="left")
    )


def add_timetable_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Target and the features that come from the timetable itself."""
    line_clean = pl.col("line_number").str.replace_all(" ", "")
    return lf.with_columns(
        ride_start=pl.col("id").str.extract(r"-(\d{10})-\d+$").str.to_datetime("%y%m%d%H%M", time_unit="ns"),
    ).with_columns(
        y=((pl.col("arrival_change_time") - pl.col("arrival_planned_time")).dt.total_minutes() >= 6).cast(pl.Int8),
        hour=pl.col("arrival_planned_time").dt.hour(),
        weekday=pl.col("arrival_planned_time").dt.weekday(),  # 1 = Monday, 7 = Sunday (ISO)
        month=pl.col("arrival_planned_time").dt.month(),
        minutes_since_departure=(pl.col("arrival_planned_time") - pl.col("ride_start"))
        .dt.total_minutes()
        .clip(lower_bound=0),
        # "1" (2024) and "S1" (2026) must be the same line: put the train type in front of digit-only values
        line=pl.when(line_clean.str.contains(r"^\d+$")).then(pl.col("train_type") + line_clean).otherwise(line_clean),
    )


def build_month(month: str, out_dir: Path = OUT_DIR) -> Path:
    """Build and save the model table for one month. Running it twice gives the same file (idempotent)."""
    arrivals = add_timetable_features(scan_arrivals(month)).collect()
    arrivals = add_is_holiday(arrivals, time_col="arrival_planned_time", station_col="stada_name")
    arrivals = add_weather(arrivals, time_col="arrival_planned_time", station_col="stada_name")
    table = arrivals.select(*METADATA, *FEATURES, TARGET)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"arrivals-{month}.parquet"
    tmp = path.with_suffix(".tmp")
    table.write_parquet(tmp)
    tmp.replace(path)  # atomic: a half-written file never appears under the real name

    logger.info("%s: %d rows, %.1f %% late -> %s", month, table.height, 100 * table[TARGET].mean(), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_month(args.month, args.out_dir)


if __name__ == "__main__":
    main()
