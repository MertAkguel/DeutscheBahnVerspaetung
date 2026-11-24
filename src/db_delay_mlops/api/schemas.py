"""What the API accepts and what it answers, as Pydantic models.

Pydantic checks every request before our code sees it: wrong types, missing fields or
values outside the allowed range are answered with HTTP 422 and a message that says
which field is wrong. FastAPI also builds the documentation (/docs) from these classes,
and an LLM can later use the same description to call the API as a tool.
"""

from datetime import datetime, timedelta
from difflib import get_close_matches
from functools import cache
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from db_delay_mlops.features.build import RAIL_GROUPS
from db_delay_mlops.features.station_info import lookup
from db_delay_mlops.paths import TRAIN_TYPE_GROUPS_FILE

MAX_DAYS_AHEAD = 7  # weather forecasts further ahead are too unreliable for our day-ahead model
MAX_BATCH_SIZE = 500

# Risk levels instead of yes/no: the probabilities are not perfectly calibrated (Brier score),
# rough levels are more honest. About 22 % of all arrivals are late (base rate).
LOW_RISK_BELOW = 0.20
MEDIUM_RISK_BELOW = 0.40


@cache
def known_stations() -> list[str]:
    """All station names the model was trained on (the selection list for Streamlit and /stations)."""
    return sorted(lookup("stada_name")["stada_name"].to_list())


@cache
def known_train_types() -> list[str]:
    """Train types of a rail group: the model never saw buses or unmapped types."""
    groups = pl.read_csv(TRAIN_TYPE_GROUPS_FILE)
    return sorted(groups.filter(pl.col("group").is_in(RAIL_GROUPS))["train_type"].to_list())


class ArrivalRequest(BaseModel):
    """One planned arrival of a train at a station. Everything here is in the timetable the day before."""

    station: str = Field(description="Station name, exactly as in GET /stations", examples=["Köln Hbf"])
    train_type: str = Field(description="Train type, e.g. ICE, IC, RE, RB, S", examples=["RE"])
    line_number: str | None = Field(
        default=None, description="Line, e.g. '1' or 'RE1'; empty for most long-distance trains", examples=["1"]
    )
    final_destination: str | None = Field(default=None, description="Last stop of the train", examples=["Aachen Hbf"])
    stop_number: int = Field(ge=1, description="Number of this stop on the ride (1 = first stop)", examples=[12])
    planned_arrival: datetime = Field(description="Planned arrival, local time", examples=["2026-12-04T17:42:00"])
    ride_start: datetime = Field(description="Planned departure at the first stop, local time")

    @field_validator("station")
    @classmethod
    def station_must_be_known(cls, value: str) -> str:
        if value not in known_stations():
            suggestions = get_close_matches(value, known_stations(), n=3)
            raise ValueError(f"unknown station {value!r}, did you mean {suggestions}? See GET /stations")
        return value

    @field_validator("train_type")
    @classmethod
    def train_type_must_be_rail(cls, value: str) -> str:
        if value not in known_train_types():
            raise ValueError(f"train type {value!r} is not supported, use one of {known_train_types()}")
        return value

    @model_validator(mode="after")
    def times_must_make_sense(self) -> "ArrivalRequest":
        # model_validator: checks that need SEVERAL fields at once
        if self.ride_start > self.planned_arrival:
            raise ValueError("ride_start must be before planned_arrival")
        now = datetime.now()
        if not now - timedelta(hours=1) <= self.planned_arrival <= now + timedelta(days=MAX_DAYS_AHEAD):
            raise ValueError(f"planned_arrival must be within the next {MAX_DAYS_AHEAD} days")
        return self


class BatchRequest(BaseModel):
    arrivals: list[ArrivalRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Reason(BaseModel):
    """How much one feature pushed this prediction up (+) or down (-)."""

    feature: str = Field(examples=["hour"])
    contribution: float = Field(description="In log-odds: + makes 'late' more likely, - less likely")


class Prediction(BaseModel):
    probability_late: float = Field(ge=0, le=1, description="Probability of arriving 6 or more minutes late")
    risk: Literal["low", "medium", "high"]
    top_reasons: list[Reason] = Field(description="The features with the largest effect, strongest first")
    warnings: list[str] = Field(
        default=[], description="Inputs the model never saw in training (ignored, so the answer is less specific)"
    )


class PredictionResponse(Prediction):
    model_version: str = Field(description="Version of db-delay-lightgbm that answered", examples=["2"])


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction] = Field(description="Same order as the requested arrivals")


class ModelInfo(BaseModel):
    # "model_" is a reserved prefix in Pydantic; this setting allows our field names anyway
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    valid_month: str | None = Field(description="Month the model was compared with its predecessor on")
    valid_pr_auc: float | None


def risk_level(probability: float) -> Literal["low", "medium", "high"]:
    """0.12 -> "low", 0.31 -> "medium", 0.55 -> "high"."""
    if probability < LOW_RISK_BELOW:
        return "low"
    if probability < MEDIUM_RISK_BELOW:
        return "medium"
    return "high"


def to_frame(arrivals: list[ArrivalRequest]) -> pl.DataFrame:
    """Requests -> the raw timetable columns that features/serving.py expects (API names -> data names)."""
    return pl.DataFrame(
        {
            "stada_name": [a.station for a in arrivals],
            "train_type": [a.train_type for a in arrivals],
            "line_number": [a.line_number for a in arrivals],
            "final_destination_station": [a.final_destination for a in arrivals],
            "train_line_station_num": [a.stop_number for a in arrivals],
            "arrival_planned_time": [a.planned_arrival for a in arrivals],
            "ride_start": [a.ride_start for a in arrivals],
        },
        schema_overrides={"line_number": pl.String, "final_destination_station": pl.String},
    )
