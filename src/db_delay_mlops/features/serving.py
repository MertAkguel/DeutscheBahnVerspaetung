"""Features for requests to the API, built with the SAME functions as features/build.py.

Train-serve skew: if the API computed a feature even slightly differently from training
(e.g. weekday 0 = Monday instead of 1 = Monday), the model would not crash, it would
silently give wrong answers. So the API never re-implements a feature. It builds rows that
look like the raw timetable data and sends them through the functions build.py uses.

Only the weather source differs: training used past day-ahead forecasts (Previous Runs API),
a train in the next days gets the current forecast (Forecast API). Both are "the forecast
about a day before", just from two endpoints.

Usage:
    requests = pl.DataFrame({"stada_name": ["Köln Hbf"], "train_type": ["RE"], ...})
    features_for(requests)  # -> one row per request, the 13 columns the model expects
"""

from typing import Literal

import polars as pl

from db_delay_mlops.features.build import RAIL_GROUPS, add_plan_features, add_train_group
from db_delay_mlops.features.columns import FEATURES
from db_delay_mlops.features.holidays import add_is_holiday
from db_delay_mlops.features.station_info import check_known
from db_delay_mlops.features.weather import add_forecast, add_weather

# What a request must contain: the raw timetable columns, with the types of the Hugging Face files
REQUEST_SCHEMA = {
    "stada_name": pl.String,
    "train_type": pl.String,
    "line_number": pl.String,
    "final_destination_station": pl.String,
    "train_line_station_num": pl.Int32,
    "arrival_planned_time": pl.Datetime("ns"),
    "ride_start": pl.Datetime("ns"),
}

WeatherSource = Literal["forecast", "history"]


def features_for(requests: pl.DataFrame, weather: WeatherSource = "forecast") -> pl.DataFrame:
    """One row of model features per request, in the column order of FEATURES.

    weather="forecast": current forecast, for trains in the next days (the API).
    weather="history": the cached day-ahead forecasts used in training (tests, past trains).
    Raises ValueError for unknown stations or train types the model was not trained on.
    """
    # _row remembers the order: joins may reorder rows, and answer i must belong to request i
    requests = requests.select(list(REQUEST_SCHEMA)).cast(REQUEST_SCHEMA).with_row_index("_row")
    check_known(requests["stada_name"], "stada_name")

    table = requests.lazy().pipe(add_train_group).pipe(add_plan_features).collect()
    unsupported = table.filter(~pl.col("train_group").is_in(RAIL_GROUPS).fill_null(False))
    if unsupported.height:
        types = sorted(unsupported["train_type"].drop_nulls().unique())
        raise ValueError(f"train type(s) not supported (rail only, see train_type_groups.csv): {types}")

    table = add_is_holiday(table, time_col="arrival_planned_time", station_col="stada_name")
    add = add_forecast if weather == "forecast" else add_weather
    table = add(table, time_col="arrival_planned_time", station_col="stada_name")
    return table.sort("_row").select(FEATURES)
