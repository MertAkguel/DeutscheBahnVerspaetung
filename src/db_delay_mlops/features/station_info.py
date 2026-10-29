"""Look up federal state and coordinates of a station from config/stations.csv."""

from functools import cache
from pathlib import Path
from typing import Literal

import polars as pl

from db_delay_mlops.paths import STATIONS_FILE

StationKey = Literal["stada_name", "eva"]


@cache  # read the CSV only once per process
def station_info(stations_file: Path = STATIONS_FILE) -> pl.DataFrame:
    """One row per EVA number of a rail station, with the station's state and coordinates.

    All EVA numbers of one station get the coordinates of its main EVA number,
    so the weather of a station does not depend on which level the train stops at.
    """
    stations = pl.read_csv(stations_file, schema_overrides={"eva": pl.String}).filter(pl.col("collect"))
    main = (
        stations.sort("is_main_eva", descending=True)
        .group_by("stada_name")
        .agg(pl.col("federal_state").first(), pl.col("lat").first(), pl.col("lon").first())
    )
    return stations.select("eva", "stada_name").join(main, on="stada_name")


def lookup(station_col: StationKey = "stada_name") -> pl.DataFrame:
    """Table keyed by `station_col` (station name or EVA number) -> stada_name, federal_state, lat, lon."""
    info = station_info()
    return info.unique(subset=["stada_name"]).drop("eva") if station_col == "stada_name" else info


def check_known(stations: pl.Series, station_col: StationKey) -> None:
    """Fail loudly for stations that are not in config/stations.csv instead of returning empty features."""
    unknown = set(stations.drop_nulls().unique()) - set(lookup(station_col)[station_col])
    if unknown:
        examples = ", ".join(sorted(unknown)[:5])
        raise ValueError(f"{len(unknown)} unknown station(s) in column '{station_col}': {examples}")
