"""Tests for features/build.py: which rows stay, and the target and timetable features."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from db_delay_mlops.features import build

OUR_EVA = "08000085"  # Düsseldorf Hbf, collect = true in config/stations.csv
OTHER_EVA = "99999999"  # not one of our stations
NS = pl.Datetime("ns")  # the Hugging Face files use nanoseconds


def raw_row(**changes) -> dict:
    """One arrival as in the Hugging Face file; by default a row that must be kept."""
    row = {
        "id": "123456789-2610011400-5",  # ride started 2026-10-01 14:00
        "eva": OUR_EVA,
        "train_type": "RE",
        "line_number": "1",
        "final_destination_station": "Aachen Hbf",
        "train_line_station_num": 5,
        "has_plan_record": True,
        "arrival_planned_time": datetime(2026, 10, 1, 14, 30),
        "arrival_change_time": datetime(2026, 10, 1, 14, 30),
        "arrival_is_canceled": False,
    }
    return row | changes


def to_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("arrival_planned_time", "arrival_change_time").cast(NS),
        pl.col("train_line_station_num").cast(pl.Int32),
        # a column with only None has no type in Polars ("null"): say it is text, like in the real file
        pl.col("line_number", "final_destination_station").cast(pl.String),
    )


@pytest.fixture
def raw_month(tmp_path, monkeypatch):
    """Write rows as a fake monthly file and point build.py at it (tmp_path is deleted after the test)."""

    def write(rows: list[dict]) -> str:
        to_frame(rows).write_parquet(tmp_path / "data-2026-10.parquet")
        monkeypatch.setattr(build, "HF_DIR", tmp_path)
        return "2026-10"

    return write


def test_row_rules(raw_month):
    month = raw_month(
        [
            raw_row(id="keep-2610011400-1"),
            raw_row(id="cancelled-2610011400-1", arrival_is_canceled=True),
            raw_row(id="first_stop-2610011400-1", arrival_planned_time=None),
            raw_row(id="not_in_plan-2610011400-1", has_plan_record=False),
            raw_row(id="bus-2610011400-1", train_type="Bus"),
            raw_row(id="unknown_type-2610011400-1", train_type="XYZ"),
            raw_row(id="other_station-2610011400-1", eva=OTHER_EVA),
        ]
    )
    kept = build.scan_arrivals(month).collect()
    assert kept["id"].to_list() == ["keep-2610011400-1"]
    assert kept["train_group"].item() == "regional"
    assert kept["stada_name"].item() == "Düsseldorf Hbf"


@pytest.mark.parametrize(("delay_minutes", "late"), [(0, 0), (5, 0), (6, 1), (30, 1)])
def test_late_means_six_minutes_or_more(delay_minutes, late):
    actual = datetime(2026, 10, 1, 14, 30) + timedelta(minutes=delay_minutes)
    out = build.add_timetable_features(to_frame([raw_row(arrival_change_time=actual)]).lazy()).collect()
    assert out["y"].item() == late


@pytest.mark.parametrize(
    ("train_type", "line_number", "line"),
    [
        ("S", "1", "S1"),  # 2024 format: digits only -> train type in front
        ("S", "S1", "S1"),  # 2026 format: already complete
        ("RE", "RE 1", "RE1"),  # spaces removed
        ("ICE", None, None),  # most long-distance trains have no line
    ],
)
def test_line_is_the_same_in_old_and_new_format(train_type, line_number, line):
    row = raw_row(train_type=train_type, line_number=line_number)
    out = build.add_timetable_features(to_frame([row]).lazy()).collect()
    assert out["line"].item() == line


def test_timetable_features():
    out = build.add_timetable_features(to_frame([raw_row()]).lazy()).collect()
    assert out["ride_start"].item() == datetime(2026, 10, 1, 14, 0)
    assert out["minutes_since_departure"].item() == 30
    assert out["hour"].item() == 14
    assert out["weekday"].item() == 4  # 2026-10-01 is a Thursday, ISO: Monday = 1
    assert out["month"].item() == 10


def test_minutes_since_departure_is_never_negative():
    # Ride ids are sometimes a bit later than the first arrival: clip at 0
    row = raw_row(id="x-2610011440-1", arrival_planned_time=datetime(2026, 10, 1, 14, 30))
    out = build.add_timetable_features(to_frame([row]).lazy()).collect()
    assert out["minutes_since_departure"].item() == 0
