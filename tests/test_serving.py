"""Tests for features/serving.py: the API must compute exactly the features the model was trained on."""

from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from db_delay_mlops.features import serving
from db_delay_mlops.features.build import HF_DIR, OUT_DIR
from db_delay_mlops.features.columns import FEATURES


def fake_weather(df, time_col, station_col):
    """Stands in for the Open-Meteo call: tests must not depend on the internet."""
    return df.with_columns(
        temperature_2m=pl.lit(5.0), precipitation=pl.lit(0.0), weather_code=pl.lit(3, dtype=pl.Int16)
    )


@pytest.fixture(autouse=True)  # autouse: active in every test of this file
def no_internet(monkeypatch):
    monkeypatch.setattr(serving, "add_forecast", fake_weather)


def request(**changes) -> dict:
    row = {
        "stada_name": "Köln Hbf",
        "train_type": "RE",
        "line_number": "1",
        "final_destination_station": "Aachen Hbf",
        "train_line_station_num": 12,
        "arrival_planned_time": datetime(2026, 12, 4, 17, 42),  # a Friday
        "ride_start": datetime(2026, 12, 4, 16, 5),
    }
    return row | changes


def test_one_request_gives_the_model_columns():
    out = serving.features_for(pl.DataFrame([request()]))
    assert out.columns == FEATURES  # same names, same order as in training
    row = out.row(0, named=True)
    assert row["line"] == "RE1"
    assert row["train_group"] == "regional"
    assert (row["hour"], row["weekday"], row["month"]) == (17, 5, 12)  # ISO: Friday = 5
    assert row["minutes_since_departure"] == 97
    assert row["is_holiday"] is False


def test_holiday_depends_on_the_federal_state():
    # 2026-11-01 (All Saints) is a holiday in NRW (Köln), not in Berlin
    day = {"arrival_planned_time": datetime(2026, 11, 1, 9, 0), "ride_start": datetime(2026, 11, 1, 8, 0)}
    cologne = request(**day)
    berlin = request(**day, stada_name="Berlin Hauptbahnhof", final_destination_station="Hamburg Hbf")
    out = serving.features_for(pl.DataFrame([cologne, berlin]))
    assert out["is_holiday"].to_list() == [True, False]


def test_batch_keeps_the_order_of_the_requests():
    hours = [22, 3, 15, 8, 11]
    requests = [
        request(arrival_planned_time=datetime(2026, 12, 4, h, 0), ride_start=datetime(2026, 12, 4, h, 0)) for h in hours
    ]
    assert serving.features_for(pl.DataFrame(requests))["hour"].to_list() == hours


def test_unknown_station_is_refused():
    with pytest.raises(ValueError, match="unknown station"):
        serving.features_for(pl.DataFrame([request(stada_name="Atlantis Hbf")]))


@pytest.mark.parametrize("train_type", ["Bus", "XYZ"])
def test_train_types_the_model_never_saw_are_refused(train_type):
    with pytest.raises(ValueError, match="not supported"):
        serving.features_for(pl.DataFrame([request(train_type=train_type)]))


# The real train-serve skew test. It needs the October data, which only exists on your machine,
# so GitHub Actions skips it (CI has no data/ folder).
ARRIVALS = OUT_DIR / "arrivals-2026-10.parquet"
RAW = HF_DIR / "data-2026-10.parquet"


@pytest.mark.skipif(not (ARRIVALS.exists() and RAW.exists()), reason="needs the October 2026 data")
def test_api_features_equal_training_features():
    trained = pl.read_parquet(ARRIVALS).sample(2000, seed=1)  # rows exactly as the model was trained on
    raw = (  # train_type and line_number are only in the raw file
        pl.scan_parquet(RAW)
        .select("id", "eva", "train_type", "line_number")
        .filter(pl.col("id").is_in(trained["id"].implode()))
        .collect()
    )
    requests = trained.join(raw, on=["id", "eva"], how="left", maintain_order="left")

    # The same trains, sent through the API path, with the same weather source as training
    served = serving.features_for(requests, weather="history")
    assert_frame_equal(served, trained.select(FEATURES))
