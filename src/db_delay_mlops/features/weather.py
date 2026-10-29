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

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
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
        # the API returns one result per location, in the order of the request
        for name, result in zip(batch["stada_name"], response.json(), strict=True):
            hourly = result["hourly"]
            frames.append(
                pl.DataFrame(
                    {
                        "stada_name": name,
                        "hour": hourly["time"],
                        **{v: hourly[f"{v}_previous_day1"] for v in VARIABLES},
                    }
                )
            )
        logger.info("Weather %s: %d/%d stations", f"{month:%Y-%m}", min(start + batch_size, coords.height), coords.height)
        time.sleep(1)  # stay far below 600 calls per minute

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
    weather = pl.concat([load_month(m.date(), weather_dir) for m in months]).with_columns(
        pl.col("hour").dt.cast_time_unit("us")
    )
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
