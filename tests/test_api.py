"""Tests for the FastAPI app, with a tiny model trained in the test (no MLflow, no internet)."""

from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from db_delay_mlops.api.main import create_app
from db_delay_mlops.api.predictor import Predictor
from db_delay_mlops.api.schemas import ModelInfo, known_stations
from db_delay_mlops.features import serving
from db_delay_mlops.features.build import RAIL_GROUPS
from db_delay_mlops.features.columns import FEATURES
from db_delay_mlops.models.calibration import CalibratedPipeline
from db_delay_mlops.models.train import make_pipeline


def fake_weather(df, time_col, station_col):
    return df.with_columns(
        temperature_2m=pl.lit(5.0), precipitation=pl.lit(0.0), weather_code=pl.lit(3, dtype=pl.Int16)
    )


@pytest.fixture(scope="module")  # module: train the tiny model once for all tests in this file
def tiny_model():
    """The real pipeline (CategoryCaster + LightGBM), trained on 400 random rows: fast, but real."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "stada_name": rng.choice(known_stations()[:20] + ["Köln Hbf"], n),
            "line": rng.choice(["RE1", "S1", "ICE10"], n),
            "final_destination_station": rng.choice(["Aachen Hbf", "Bonn Hbf"], n),
            "train_group": rng.choice(RAIL_GROUPS, n),
            "hour": rng.integers(0, 24, n).astype("int8"),
            "weekday": rng.integers(1, 8, n).astype("int8"),
            "month": rng.integers(1, 13, n).astype("int8"),
            "weather_code": rng.choice([0, 3, 61], n).astype("int16"),
            "train_line_station_num": rng.integers(1, 30, n).astype("int32"),
            "minutes_since_departure": rng.integers(0, 300, n),
            "temperature_2m": rng.normal(10, 5, n),
            "precipitation": rng.exponential(0.5, n),
            "is_holiday": rng.random(n) < 0.05,
        }
    )[FEATURES]
    y = ((X["hour"] >= 16) & (X["hour"] <= 18) | (rng.random(n) < 0.2)).astype(int)  # rush hour is late
    pipeline = make_pipeline().set_params(model__n_estimators=20, model__min_child_samples=5).fit(X, y)
    return CalibratedPipeline(pipeline).fit(X, y)  # like models/train.py: calibration around the pipeline


@pytest.fixture
def client(tiny_model, monkeypatch):
    monkeypatch.setattr(serving, "add_forecast", fake_weather)
    info = ModelInfo(model_name="db-delay-lightgbm", model_version="test", valid_month="2026-10", valid_pr_auc=0.5)
    app = create_app(load_predictor=lambda: Predictor(model=tiny_model, info=info))
    with TestClient(app) as client:  # "with" runs the lifespan: the model is "loaded" here
        yield client


TOMORROW = (datetime.now() + timedelta(days=1)).replace(hour=17, minute=42, second=0, microsecond=0)


def arrival(**changes) -> dict:
    data = {
        "station": "Köln Hbf",
        "train_type": "RE",
        "line_number": "1",
        "final_destination": "Aachen Hbf",
        "stop_number": 12,
        "planned_arrival": TOMORROW.isoformat(),
        "ride_start": (TOMORROW - timedelta(minutes=97)).isoformat(),
    }
    return data | changes


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client):
    assert client.get("/model").json()["model_version"] == "test"


def test_stations_list(client):
    stations = client.get("/stations").json()
    assert "Köln Hbf" in stations
    assert "Berlin Hauptbahnhof" in stations


def test_predict(client):
    response = client.post("/predict", json=arrival())
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_late"] <= 1
    assert body["risk"] in {"low", "medium", "high"}
    assert body["model_version"] == "test"
    assert len(body["top_reasons"]) == 3
    assert {r["feature"] for r in body["top_reasons"]} <= set(FEATURES)
    assert body["warnings"] == []  # Aachen Hbf and RE1 are known to the tiny model


def test_unknown_destination_gives_a_warning_not_an_error(client):
    # Destinations use the timetable's names ("Berlin Hbf"), not our station list ("Berlin Hauptbahnhof")
    response = client.post("/predict", json=arrival(final_destination="Aachen Hauptbahnhof"))
    assert response.status_code == 200  # still answered: new destinations do appear (construction, new lines)
    [warning] = response.json()["warnings"]
    assert "final_destination" in warning
    assert "Aachen Hbf" in warning  # the suggestion


def test_unknown_line_gives_a_warning(client):
    response = client.post("/predict", json=arrival(line_number="99"))  # becomes "RE99"
    assert "RE99" in response.json()["warnings"][0]


def test_unknown_station_gives_422_with_suggestions(client):
    response = client.post("/predict", json=arrival(station="Berlin Hbf"))
    assert response.status_code == 422
    assert "Berlin Hauptbahnhof" in response.text  # the suggestion reaches the caller


def test_batch_answers_in_the_same_order(client):
    morning = arrival(
        planned_arrival=(TOMORROW.replace(hour=9)).isoformat(),
        ride_start=(TOMORROW.replace(hour=8)).isoformat(),
    )
    response = client.post("/predict/batch", json={"arrivals": [arrival(), morning, arrival()]})
    assert response.status_code == 200
    p = [prediction["probability_late"] for prediction in response.json()["predictions"]]
    assert len(p) == 3
    assert p[0] == p[2]  # the same train twice -> the same answer, in its own position


def test_weather_service_down_gives_503(client, monkeypatch):
    def broken_weather(*args, **kwargs):
        raise httpx.ConnectError("Open-Meteo unreachable")

    monkeypatch.setattr(serving, "add_forecast", broken_weather)
    response = client.post("/predict", json=arrival())
    assert response.status_code == 503
