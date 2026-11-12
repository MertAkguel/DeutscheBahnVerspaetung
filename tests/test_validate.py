"""Tests for data/validate.py: broken tables must be rejected, good ones accepted."""

from datetime import datetime

import pandera.errors
import polars as pl
import pytest

from db_delay_mlops.data import validate


def good_table(n: int = 3) -> pl.DataFrame:
    """A small table that matches ARRIVALS_SCHEMA exactly (types included)."""
    planned = datetime(2026, 10, 1, 14, 30)
    return pl.DataFrame(
        {
            "id": [f"ride-{i}" for i in range(n)],
            "eva": ["08000085"] * n,
            "arrival_planned_time": [planned] * n,
            "ride_start": [planned] * n,
            "stada_name": ["Düsseldorf Hbf"] * n,
            "line": ["RE1"] + [None] * (n - 1),
            "final_destination_station": ["Aachen Hbf"] * n,
            "train_group": ["regional"] * n,
            "hour": [14] * n,
            "weekday": [4] * n,
            "month": [10] * n,
            "weather_code": [3] * n,
            "train_line_station_num": [5] * n,
            "minutes_since_departure": [30] * n,
            "temperature_2m": [12.5] * n,
            "precipitation": [0.0] * n,
            "is_holiday": [False] * n,
            "y": [0, 1, 0][:n] + [0] * max(0, n - 3),
        },
        schema_overrides={
            "arrival_planned_time": pl.Datetime("ns"),
            "ride_start": pl.Datetime("ns"),
            "hour": pl.Int8,
            "weekday": pl.Int8,
            "month": pl.Int8,
            "y": pl.Int8,
            "weather_code": pl.Int16,
            "train_line_station_num": pl.Int32,
        },
    )


def test_good_table_passes():
    validate.ARRIVALS_SCHEMA.validate(good_table())


@pytest.mark.parametrize(
    "break_it",
    [
        pytest.param(lambda t: t.with_columns(hour=pl.lit(24, pl.Int8)), id="hour 24"),
        pytest.param(lambda t: t.with_columns(weekday=pl.lit(0, pl.Int8)), id="weekday 0 (ISO starts at 1)"),
        pytest.param(lambda t: t.with_columns(stada_name=pl.lit("Atlantis Hbf")), id="unknown station"),
        pytest.param(lambda t: t.with_columns(train_group=pl.lit("exclude")), id="bus slipped through"),
        pytest.param(lambda t: t.with_columns(y=pl.lit(2, pl.Int8)), id="target not 0/1"),
        pytest.param(lambda t: t.with_columns(id=pl.lit("same")), id="duplicate ids"),
        pytest.param(lambda t: t.with_columns(stada_name=pl.lit(None, pl.String)), id="missing station"),
        pytest.param(lambda t: t.with_columns(delay_in_min=pl.lit(3)), id="extra column (leakage)"),
        pytest.param(lambda t: t.drop("temperature_2m"), id="missing column"),
        pytest.param(lambda t: t.with_columns(pl.col("hour").cast(pl.Int64)), id="wrong type"),
    ],
)
def test_broken_table_is_rejected(break_it):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate.ARRIVALS_SCHEMA.validate(break_it(good_table()), lazy=True)


def test_too_few_rows_stops_the_pipeline(tmp_path, monkeypatch):
    good_table().write_parquet(tmp_path / "arrivals-2026-10.parquet")
    # unmapped_share() reads the raw Hugging Face file, which the test does not have
    monkeypatch.setattr(validate, "unmapped_share", lambda month: (0.0, []))
    with pytest.raises(ValueError, match="rows"):
        validate.validate_month("2026-10", out_dir=tmp_path)


def test_enough_rows_passes(tmp_path, monkeypatch):
    good_table().write_parquet(tmp_path / "arrivals-2026-10.parquet")
    monkeypatch.setattr(validate, "unmapped_share", lambda month: (0.0, []))
    monkeypatch.setattr(validate, "MIN_ROWS", 3)
    stats = validate.validate_month("2026-10", out_dir=tmp_path)
    assert stats["rows"] == 3
