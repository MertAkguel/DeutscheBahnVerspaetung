"""Weather forecasts made the day before, per station and hour (Open-Meteo Previous Runs API).

You pass a time and a station; coordinates are looked up in config/stations.csv and
the needed months are fetched and cached automatically.

`_previous_day1` is what the forecast said 24 hours earlier. That is the information
the model also has in production, so it does not leak the observed weather.

Limits of the free API: 600 calls/min, 5,000/hour, 10,000/day. One location for one
month counts as about 2 calls, so the 110 stations need about 220 calls per month.
Each month is cached as a Parquet file in data/raw/weather/ and only fetched once.

Usage:
    add_weather(arrivals, time_col="arrival_planned_time", station_col="stada_name")
    weather_at(datetime(2024, 7, 15, 17, 42), "Köln Hbf")
"""

import logging
import time
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from db_delay_mlops.features.station_info import StationKey, check_known, lookup
from db_delay_mlops.paths import WEATHER_DIR

logger = logging.getLogger(__name__)

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"  # past day-ahead forecasts (training)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"  # the current forecast (API, trains in the future)
VARIABLES = ["temperature_2m", "precipitation", "weather_code"]

# WMO weather codes -> readable category (https://open-meteo.com/en/docs, section "WMO Weather interpretation codes")
WEATHER_CATEGORIES = {
    "clear": [0],
    "cloudy": [1, 2, 3],
    "fog": [45, 48],
    "drizzle": [51, 53, 55, 56, 57],
    "rain": [61, 63, 66, 80, 81],
    "heavy rain": [65, 67, 82],
    "snow": [71, 73, 75, 77, 85, 86],
    "thunderstorm": [95, 96, 99],
}
_CODE_TO_CATEGORY = {code: name for name, codes in WEATHER_CATEGORIES.items() for code in codes}


def weather_category(code_col: str = "weather_code") -> pl.Expr:
    """Expression that turns a WMO weather code into a readable category, e.g. 95 -> "thunderstorm"."""
    return pl.col(code_col).replace_strict(_CODE_TO_CATEGORY, default="unknown", return_dtype=pl.String)


def fetch_month(month: date, batch_size: int = 25) -> pl.DataFrame:
    """Hourly day-ahead forecasts for all stations for one month."""
    coords = lookup("stada_name").sort("stada_name")
    last_day = monthrange(month.year, month.month)[1]
    frames = []
    for start in range(0, coords.height, batch_size):
        batch = coords.slice(start, batch_size)
        response = httpx.get(
            API_URL,
            params={
                # several locations in one request, comma-separated
                "latitude": ",".join(str(x) for x in batch["lat"]),
                "longitude": ",".join(str(x) for x in batch["lon"]),
                "start_date": f"{month:%Y-%m}-01",
                "end_date": f"{month:%Y-%m}-{last_day}",
                "hourly": ",".join(f"{v}_previous_day1" for v in VARIABLES),
                "timezone": "Europe/Berlin",  # same local time as the timetable data
            },
            timeout=60,
        )
        response.raise_for_status()
        frames.extend(_station_frames(batch["stada_name"], response.json(), suffix="_previous_day1"))
        logger.info(
            "Weather %s: %d/%d stations", f"{month:%Y-%m}", min(start + batch_size, coords.height), coords.height
        )
        time.sleep(1)  # stay far below 600 calls per minute

    return _combine(frames)


def fetch_forecast(stations: list[str], start: date, end: date) -> pl.DataFrame:
    """Hourly current forecast for some stations, from `start` to `end` (at most about 16 days ahead)."""
    coords = lookup("stada_name").filter(pl.col("stada_name").is_in(stations)).sort("stada_name")
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": ",".join(str(x) for x in coords["lat"]),
            "longitude": ",".join(str(x) for x in coords["lon"]),
            "start_date": f"{start:%Y-%m-%d}",
            "end_date": f"{end:%Y-%m-%d}",
            "hourly": ",".join(VARIABLES),  # same variables as in training, without "_previous_day1"
            "timezone": "Europe/Berlin",
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()
    if isinstance(results, dict):  # one location: Open-Meteo returns one object instead of a list
        results = [results]
    return _combine(_station_frames(coords["stada_name"], results, suffix=""))


def _station_frames(names: pl.Series, results: list[dict], suffix: str) -> list[pl.DataFrame]:
    """One table per station from an Open-Meteo answer (one result per location, in request order)."""
    return [
        pl.DataFrame(
            {
                "stada_name": name,
                "hour": result["hourly"]["time"],
                **{v: result["hourly"][f"{v}{suffix}"] for v in VARIABLES},
            }
        )
        for name, result in zip(names, results, strict=True)
    ]


def _combine(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Same column types for past and current forecasts, so training and API see identical data."""
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("hour").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="us"),
            pl.col("temperature_2m").cast(pl.Float64),
            pl.col("precipitation").cast(pl.Float64),
            pl.col("weather_code").cast(pl.Int16),
        )
        # On the night of the autumn clock change, 02:00 local time exists twice
        .unique(subset=["stada_name", "hour"], keep="first", maintain_order=True)
    )


def load_month(month: date, weather_dir: Path = WEATHER_DIR) -> pl.DataFrame:
    """Read the cached month, or fetch and cache it first."""
    path = weather_dir / f"weather-{month:%Y-%m}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        weather = fetch_month(month)
        tmp = path.with_suffix(".tmp")
        weather.write_parquet(tmp)
        tmp.replace(path)
    return pl.read_parquet(path)


def add_weather(
    df: pl.DataFrame,
    time_col: str = "arrival_planned_time",
    station_col: StationKey = "stada_name",
    weather_dir: Path = WEATHER_DIR,
) -> pl.DataFrame:
    """Add the day-ahead forecast for the station in `station_col` and the hour in `time_col`.

    Loads every month that occurs in `time_col`, so arrivals shortly after midnight on
    the 1st of a month also get their weather. Use the planned time, never the actual time.
    """
    check_known(df[station_col], station_col)
    hours = df[time_col].dt.truncate("1h").dt.cast_time_unit("us")  # timetable uses ns, weather us
    months = hours.dt.truncate("1mo").unique().drop_nulls().sort().to_list()
    weather = pl.concat([load_month(m.date(), weather_dir) for m in months])
    return _join_weather(df, weather, hours, station_col)


def add_forecast(
    df: pl.DataFrame, time_col: str = "arrival_planned_time", station_col: StationKey = "stada_name"
) -> pl.DataFrame:
    """Like add_weather, but with the CURRENT forecast: for trains in the next days (the API).

    For a train tomorrow, today's forecast is a day-ahead forecast, like the `_previous_day1`
    values the model was trained with. Further ahead the forecast gets less accurate.
    """
    check_known(df[station_col], station_col)
    hours = df[time_col].dt.truncate("1h").dt.cast_time_unit("us")
    stations = lookup(station_col).filter(pl.col(station_col).is_in(df[station_col].implode()))["stada_name"]
    weather = fetch_forecast(stations.unique().to_list(), hours.min().date(), hours.max().date())
    return _join_weather(df, weather, hours, station_col)


def _join_weather(df: pl.DataFrame, weather: pl.DataFrame, hours: pl.Series, station_col: StationKey) -> pl.DataFrame:
    """Attach the weather of each row's station and hour (shared by add_weather and add_forecast)."""
    weather = weather.with_columns(pl.col("hour").dt.cast_time_unit("us"))
    station_names = lookup(station_col).select(station_col, _station="stada_name")
    return (
        df.with_columns(_hour=hours)
        .join(station_names, on=station_col, how="left")
        .join(weather.rename({"stada_name": "_station", "hour": "_hour"}), on=["_station", "_hour"], how="left")
        .drop("_station", "_hour")
    )


def weather_at(when: datetime, station: str, station_col: StationKey = "stada_name") -> dict:
    """Single lookup, e.g. for the API: day-ahead forecast at `station` for the hour of `when`."""
    row = pl.DataFrame({"time": [when], station_col: [station]})
    return add_weather(row, time_col="time", station_col=station_col).select(VARIABLES).row(0, named=True)
