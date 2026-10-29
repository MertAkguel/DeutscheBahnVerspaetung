"""Collect raw timetable data from the DB Timetables API.

Every call is stored unchanged in the raw layer. Parsing happens later and can
be repeated at any time; a missed fchg snapshot can never be fetched again.

Usage:
    # hourly job: all changes now + the plan of the next hour
    uv run python -m db_delay_mlops.data.collect run
    # repair a gap in the plan data
    uv run python -m db_delay_mlops.data.collect plan --date 2026-10-28 --hours 5,6,7
"""

import argparse
import logging
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from db_delay_mlops.data.db_api import DBApiClient, DBApiSettings, RawResponse

logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
RAW_DIR = Path("data/raw/db_api")
STATIONS_FILE = Path("config/stations.csv")

RAW_SCHEMA = {
    "fetched_at": pl.Datetime("us", "UTC"),
    "endpoint": pl.String,
    "eva": pl.String,
    "url": pl.String,
    "status_code": pl.Int32,
    "body": pl.String,
    "error": pl.String,
    "attempts": pl.Int32,
}


def save_raw(responses: list[RawResponse], endpoint: str, run_started: datetime, raw_dir: Path = RAW_DIR) -> Path:
    """Write one immutable file per run and endpoint, partitioned by fetch date (UTC)."""
    path = raw_dir / f"date={run_started:%Y-%m-%d}" / f"{endpoint}_{run_started:%H%M%S}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([asdict(r) for r in responses], schema=RAW_SCHEMA)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)  # atomic: readers never see a half-written file

    failed = df.filter(pl.col("error").is_not_null()).height
    logger.info("Saved %d %s responses (%d failed) to %s", df.height, endpoint, failed, path)
    return path


def load_evas(stations_file: Path = STATIONS_FILE) -> list[str]:
    """EVA numbers of the stations we collect (see db_delay_mlops.data.stations)."""
    stations = pl.read_csv(stations_file, schema_overrides={"eva": pl.String})
    return stations.filter(pl.col("collect"))["eva"].to_list()


def collect_fchg(client: DBApiClient, evas: list[str], run_started: datetime) -> Path:
    logger.info("Fetching fchg for %d stations", len(evas))
    return save_raw([client.fchg(eva) for eva in evas], "fchg", run_started)


def collect_plan(client: DBApiClient, evas: list[str], hours: list[datetime], run_started: datetime) -> Path:
    logger.info("Fetching plan for %d stations x %d hours", len(evas), len(hours))
    responses = [client.plan(eva, hour) for hour in hours for eva in evas]
    return save_raw(responses, "plan", run_started)


def parse_hours(day: date, hours: str) -> list[datetime]:
    return [datetime(day.year, day.month, day.day, int(h), tzinfo=BERLIN) for h in hours.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations-file", type=Path, default=STATIONS_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fchg for all stations + plan for the coming hour(s)")
    run.add_argument("--plan-hours-ahead", type=int, default=1, help="How many coming hours of plan to fetch")

    sub.add_parser("fchg", help="Only fetch the current changes")

    plan = sub.add_parser("plan", help="Fetch the plan for given hours of one day")
    plan.add_argument("--date", type=date.fromisoformat, required=True, help="YYYY-MM-DD (German local date)")
    plan.add_argument("--hours", required=True, help="Comma-separated hours 0-23, e.g. 5,6,7")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    evas = load_evas(args.stations_file)
    run_started = datetime.now(UTC)

    with DBApiClient(DBApiSettings()) as client:
        if args.command in ("run", "fchg"):
            collect_fchg(client, evas, run_started)
        if args.command == "run":
            next_hour = datetime.now(BERLIN).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            hours = [next_hour + timedelta(hours=i) for i in range(args.plan_hours_ahead)]
            collect_plan(client, evas, hours, run_started)
        if args.command == "plan":
            collect_plan(client, evas, parse_hours(args.date, args.hours), run_started)


if __name__ == "__main__":
    main()
