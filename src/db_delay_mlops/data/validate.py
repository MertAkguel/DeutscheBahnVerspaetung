"""Check a monthly model table before it is used for training.

Two kinds of checks:
- Schema (pandera): every column has the right type, allowed values and no unexpected gaps.
  A failure means the data is broken -> the pipeline stops.
- Plausibility (plain Polars): row count, late share, train types missing in
  config/train_type_groups.csv. Some of these only warn, because the data is usable
  but a human should look at it.

Usage:
    uv run python -m db_delay_mlops.data.validate --month 2026-09
"""

import argparse
import logging
from pathlib import Path

import pandera.polars as pa
import polars as pl

from db_delay_mlops.features.build import HF_DIR, OUT_DIR, RAIL_GROUPS
from db_delay_mlops.paths import STATIONS_FILE, TRAIN_TYPE_GROUPS_FILE

logger = logging.getLogger(__name__)

MIN_ROWS = 500_000  # a normal month has about 1.4 M rows; far fewer means missing data
LATE_SHARE_RANGE = (0.05, 0.60)  # observed 2024-07 to 2026-09: about 0.16 to 0.25
MAX_UNMAPPED_SHARE = 0.005  # train types not in train_type_groups.csv, share of rows at our stations

STATION_NAMES = (
    pl.read_csv(STATIONS_FILE, schema_overrides={"eva": pl.String}).filter(pl.col("collect"))["stada_name"].unique().to_list()
)

ARRIVALS_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(pl.String, unique=True),
        "eva": pa.Column(pl.String),
        "arrival_planned_time": pa.Column(pl.Datetime("ns")),
        "ride_start": pa.Column(pl.Datetime("ns")),
        "stada_name": pa.Column(pl.String, pa.Check.isin(STATION_NAMES)),
        "line": pa.Column(pl.String, nullable=True),  # empty for most long-distance trains
        "final_destination_station": pa.Column(pl.String, nullable=True),
        "train_group": pa.Column(pl.String, pa.Check.isin(RAIL_GROUPS)),
        "hour": pa.Column(pl.Int8, pa.Check.in_range(0, 23)),
        "weekday": pa.Column(pl.Int8, pa.Check.in_range(1, 7)),  # ISO: 1 = Monday
        "month": pa.Column(pl.Int8, pa.Check.in_range(1, 12)),
        "weather_code": pa.Column(pl.Int16, pa.Check.in_range(0, 99)),  # WMO codes
        "train_line_station_num": pa.Column(pl.Int32, pa.Check.ge(1)),
        "minutes_since_departure": pa.Column(pl.Int64, pa.Check.in_range(0, 3000)),
        "temperature_2m": pa.Column(pl.Float64, pa.Check.in_range(-40, 50)),
        "precipitation": pa.Column(pl.Float64, pa.Check.in_range(0, 200)),
        "is_holiday": pa.Column(pl.Boolean),
        "y": pa.Column(pl.Int8, pa.Check.isin([0, 1])),
    },
    strict=True,  # no columns other than these: an outcome column must never slip in
)


def unmapped_share(month: str) -> tuple[float, list[str]]:
    """Share of rows at our stations whose train type is not in train_type_groups.csv, and the top types."""
    stations = pl.read_csv(STATIONS_FILE, schema_overrides={"eva": pl.String})
    known_types = pl.read_csv(TRAIN_TYPE_GROUPS_FILE)["train_type"]
    raw = (
        pl.scan_parquet(HF_DIR / f"data-{month}.parquet")
        .select("eva", "train_type")
        .filter(pl.col("eva").is_in(stations.filter(pl.col("collect"))["eva"].implode()))
        .with_columns(unmapped=~pl.col("train_type").is_in(known_types.implode()))
        .collect()
    )
    top = raw.filter("unmapped").group_by("train_type").len().sort("len", descending=True).head(5)["train_type"]
    return raw["unmapped"].mean(), top.to_list()


def validate_month(month: str, out_dir: Path = OUT_DIR) -> dict[str, float]:
    """Validate the built table of one month. Raises if the data is broken, logs warnings otherwise."""
    table = pl.read_parquet(out_dir / f"arrivals-{month}.parquet")

    # 1. Schema: lazy=True collects all failures instead of stopping at the first one
    ARRIVALS_SCHEMA.validate(table, lazy=True)

    # 2. Plausibility
    stats = {"rows": table.height, "late_share": table["y"].mean()}
    stats["unmapped_share"], top_unmapped = unmapped_share(month)

    if stats["rows"] < MIN_ROWS:
        raise ValueError(f"{month}: only {stats['rows']:,} rows (expected at least {MIN_ROWS:,}), data is incomplete")
    if not LATE_SHARE_RANGE[0] <= stats["late_share"] <= LATE_SHARE_RANGE[1]:
        logger.warning("%s: late share %.1f %% is outside the usual range", month, 100 * stats["late_share"])
    if stats["unmapped_share"] > MAX_UNMAPPED_SHARE:
        logger.warning(
            "%s: %.2f %% of rows have a train type missing in train_type_groups.csv, e.g. %s",
            month, 100 * stats["unmapped_share"], top_unmapped,
        )

    logger.info("%s valid: %s", month, {k: round(v, 4) for k, v in stats.items()})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", required=True, help="YYYY-MM")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_month(args.month, args.out_dir)


if __name__ == "__main__":
    main()
