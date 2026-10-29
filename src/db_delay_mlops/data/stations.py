"""Build config/stations.csv: the stations this project covers, with StaDa master data.

The Hugging Face dataset only covered the ~100 largest stations until 2025-11-02.
We take every EVA number seen before that date, then look each one up in StaDa
for federal state and coordinates (needed for weather and holidays).

Usage:
    uv run python -m db_delay_mlops.data.stations
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from db_delay_mlops.data.collect import STATIONS_FILE, save_raw
from db_delay_mlops.data.db_api import DBApiClient, DBApiSettings, RawResponse

logger = logging.getLogger(__name__)

HF_DIR = Path("data/raw/hf/monthly_processed_data")
FIRST_FULL_COVERAGE_MONTH = "2025-11"


def evas_before_coverage_change(hf_dir: Path = HF_DIR) -> pl.DataFrame:
    """Every EVA number in the monthly files before the switch to all stations."""
    files = sorted(f for f in hf_dir.glob("data-*.parquet") if f.stem.removeprefix("data-") < FIRST_FULL_COVERAGE_MONTH)
    if not files:
        raise FileNotFoundError(f"No monthly files before {FIRST_FULL_COVERAGE_MONTH} in {hf_dir}")
    logger.info("Reading EVA numbers from %d monthly files", len(files))
    return (
        pl.scan_parquet(files)
        .group_by("eva")
        .agg(
            pl.col("xml_station_name").mode().first(),
            pl.len().alias("n_stops"),
            # `time` is clipped to the file's month, planned times can spill into the next one
            pl.col("time").dt.strftime("%Y-%m").n_unique().alias("n_months"),
        )
        .sort("n_stops", descending=True)
        .collect()
    )


def parse_stada(response: RawResponse) -> dict:
    """Pick the station and EVA entry matching the requested EVA number."""
    empty = {"stada_name": None, "category": None, "federal_state": None, "lat": None, "lon": None, "is_main_eva": None}
    if response.status_code != 200 or not response.body:
        return empty
    wanted = int(response.eva)
    for station in json.loads(response.body).get("result", []):
        for eva in station.get("evaNumbers", []):
            if eva.get("number") == wanted:
                lon, lat = eva.get("geographicCoordinates", {}).get("coordinates", [None, None])
                return {
                    "stada_name": station.get("name"),
                    "category": station.get("category"),
                    "federal_state": station.get("federalState"),
                    "lat": lat,
                    "lon": lon,
                    "is_main_eva": eva.get("isMain"),
                }
    return empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-dir", type=Path, default=HF_DIR)
    parser.add_argument("--output", type=Path, default=STATIONS_FILE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    evas = evas_before_coverage_change(args.hf_dir)
    run_started = datetime.now(UTC)
    with DBApiClient(DBApiSettings()) as client:
        responses = [client.station(eva) for eva in evas["eva"]]
    save_raw(responses, "stada", run_started)

    stada = pl.DataFrame([{"eva": r.eva, **parse_stada(r)} for r in responses])
    stations = evas.join(stada, on="eva", how="left").with_columns(
        # Only rail stations known to StaDa; bus stations (ZOB) have no StaDa entry.
        collect=pl.col("federal_state").is_not_null()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stations.write_csv(args.output)
    logger.info(
        "Wrote %d stations to %s (%d to collect, %d without StaDa match)",
        stations.height,
        args.output,
        stations["collect"].sum(),
        stations.filter(~pl.col("collect")).height,
    )
    print(stations.filter(~pl.col("collect")).select("eva", "xml_station_name", "n_stops"))


if __name__ == "__main__":
    main()
