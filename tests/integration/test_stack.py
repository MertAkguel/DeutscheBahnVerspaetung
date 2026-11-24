"""Integration tests: talk to the REAL running stack over HTTP (make up), nothing is faked.

Unit tests check each part alone with fakes. These check that the parts work together:
the API container reaches the MLflow container, loads the champion and gets a weather forecast.
Skipped when the stack is not running (e.g. on GitHub Actions).

Run: make integration
"""

import os
from datetime import datetime, timedelta

import httpx
import pytest

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")


def reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [  # applies to every test in this file
    pytest.mark.integration,
    pytest.mark.skipif(not reachable(API_URL), reason=f"API not running at {API_URL} (make up)"),
]

TOMORROW = (datetime.now() + timedelta(days=1)).replace(hour=17, minute=42, second=0, microsecond=0)
ARRIVAL = {
    "station": "Köln Hbf",
    "train_type": "RE",
    "line_number": "1",
    "final_destination": "Aachen Hbf",
    "stop_number": 5,
    "planned_arrival": TOMORROW.isoformat(),
    "ride_start": (TOMORROW - timedelta(minutes=31)).isoformat(),
}


def test_mlflow_is_up():
    assert reachable(MLFLOW_URL)


def test_api_serves_the_current_champion():
    info = httpx.get(f"{API_URL}/model").json()
    assert info["model_name"] == "db-delay-lightgbm"
    # The API must answer with the version that has the alias @champion right now
    champion = httpx.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias",
        params={"name": "db-delay-lightgbm", "alias": "champion"},
    ).json()["model_version"]["version"]
    assert info["model_version"] == str(champion)


def test_prediction_with_real_weather_forecast():
    response = httpx.post(f"{API_URL}/predict", json=ARRIVAL, timeout=30)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["probability_late"] < 1
    assert len(body["top_reasons"]) == 3


def test_batch_of_one_station():
    hours = [7, 12, 17, 22]
    arrivals = [
        ARRIVAL
        | {
            "planned_arrival": TOMORROW.replace(hour=h).isoformat(),
            "ride_start": (TOMORROW.replace(hour=h) - timedelta(minutes=31)).isoformat(),
        }
        for h in hours
    ]
    response = httpx.post(f"{API_URL}/predict/batch", json={"arrivals": arrivals}, timeout=30)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == len(hours)


def test_bad_input_is_refused():
    response = httpx.post(f"{API_URL}/predict", json=ARRIVAL | {"station": "Atlantis Hbf"})
    assert response.status_code == 422
