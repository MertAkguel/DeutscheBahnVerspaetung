"""Public holidays per German federal state.

You pass a time and a station; the federal state is looked up in config/stations.csv.
Holidays are known months in advance, so they are safe to use the day before.
Regional holidays below state level (e.g. Augsburger Friedensfest) are not covered.

Usage:
    add_is_holiday(arrivals, time_col="arrival_planned_time", station_col="stada_name")
    is_holiday(datetime(2024, 10, 31, 8, 0), "Hamburg Hbf")  # True: Reformationstag
"""

from datetime import date, datetime

import holidays
import polars as pl

from db_delay_mlops.features.station_info import StationKey, check_known, lookup

# StaDa federal state names -> ISO 3166-2 subdivision codes used by the `holidays` package
FEDERAL_STATE_CODES = {
    "Baden-Württemberg": "BW",
    "Bayern": "BY",
    "Berlin": "BE",
    "Brandenburg": "BB",
    "Bremen": "HB",
    "Hamburg": "HH",
    "Hessen": "HE",
    "Mecklenburg-Vorpommern": "MV",
    "Niedersachsen": "NI",
    "Nordrhein-Westfalen": "NW",
    "Rheinland-Pfalz": "RP",
    "Saarland": "SL",
    "Sachsen": "SN",
    "Sachsen-Anhalt": "ST",
    "Schleswig-Holstein": "SH",
    "Thüringen": "TH",
}


def public_holidays(years: list[int]) -> pl.DataFrame:
    """One row per (federal state, date) that is a public holiday."""
    rows = [
        {"federal_state": state, "date": day, "holiday_name": name}
        for state, code in FEDERAL_STATE_CODES.items()
        for day, name in holidays.country_holidays("DE", subdiv=code, years=years).items()
    ]
    return pl.DataFrame(rows, schema={"federal_state": pl.String, "date": pl.Date, "holiday_name": pl.String})


def add_is_holiday(
    df: pl.DataFrame, time_col: str = "arrival_planned_time", station_col: StationKey = "stada_name"
) -> pl.DataFrame:
    """Add `is_holiday`: is the date in `time_col` a public holiday at the station in `station_col`?

    Use the planned time, never the actual time (`time`, `arrival_change_time`): the
    actual time is not known the day before.
    """
    check_known(df[station_col], station_col)
    states = lookup(station_col).select(station_col, _state="federal_state")
    years = df[time_col].dt.year().unique().drop_nulls().to_list()
    holiday_days = (
        public_holidays(years)
        .select(_state="federal_state", _date="date", is_holiday=pl.lit(True))
        .unique()  # two holidays on the same day (e.g. 1 May and Ascension in 2008) must not duplicate rows
    )
    return (
        df.join(states, on=station_col, how="left")
        .with_columns(_date=pl.col(time_col).dt.date())
        .join(holiday_days, on=["_state", "_date"], how="left")
        .with_columns(pl.col("is_holiday").fill_null(False))
        .drop("_state", "_date")
    )


def is_holiday(when: datetime | date, station: str, station_col: StationKey = "stada_name") -> bool:
    """Single lookup, e.g. for the API: is `when` a public holiday at `station`?"""
    when = when if isinstance(when, datetime) else datetime(when.year, when.month, when.day)
    row = pl.DataFrame({"time": [when], station_col: [station]})
    return add_is_holiday(row, time_col="time", station_col=station_col)["is_holiday"].item()
