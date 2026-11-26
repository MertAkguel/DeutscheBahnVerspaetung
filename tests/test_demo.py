"""Tests for the Streamlit demo, with a fake prediction API (no Docker, no internet).

Streamlit's AppTest runs app.py like a browser would, but without one: it executes the script,
lets us click buttons and choose values, and shows what ended up on the page.
The app talks to the API with httpx.get / httpx.post. We replace those two functions with
fakes that answer like the real API, so each test decides exactly what the API says.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import db_delay_mlops.data.db_api as db_api
import db_delay_mlops.data.timetable as timetable

APP = Path(__file__).parents[1] / "src" / "db_delay_mlops" / "demo" / "app.py"
API = "http://localhost:8000"
TOMORROW = datetime.now().replace(hour=17, minute=42, second=0, microsecond=0) + timedelta(days=1)


def prediction(probability: float, risk: str) -> dict:
    """One answer of POST /predict/batch, as the real API sends it."""
    return {
        "probability_late": probability,
        "risk": risk,
        "top_reasons": [
            {"feature": "hour", "contribution": 0.4},
            {"feature": "weather_code", "contribution": 0.1},
            {"feature": "stada_name", "contribution": -0.2},
        ],
        "warnings": [],
    }


class FakeApi:
    """Answers GET and POST like the real API and remembers what the app sent."""

    def __init__(self, reachable: bool = True, predict_status: int = 200):
        self.reachable = reachable
        self.predict_status = predict_status
        self.sent: list[dict] = []  # bodies of POST /predict/batch

    def get(self, url, **kwargs):
        if not self.reachable:
            raise httpx.ConnectError("connection refused")
        answers = {
            "/model": {"model_name": "db-delay-lightgbm", "model_version": "7", "valid_month": "2026-10"},
            "/stations": ["Bonn Hbf", "Köln Hbf"],
            "/train-types": ["RE", "S"],
        }
        return httpx.Response(200, json=answers[url.removeprefix(API)], request=httpx.Request("GET", url))

    def post(self, url, json, **kwargs):
        self.sent.append(json)
        request = httpx.Request("POST", url)
        if self.predict_status == 422:
            return httpx.Response(422, json={"detail": "Unknown station"}, request=request)
        predictions = [prediction(0.35, "medium") for _ in json["arrivals"]]
        return httpx.Response(200, json={"model_version": "7", "predictions": predictions}, request=request)


@pytest.fixture
def run_app(monkeypatch, tmp_path):
    """Start the app against a FakeApi. Returns a function, so each test can configure the fake."""
    # No DB key: no .env in the working directory and no DB_ variables (the real .env stays untouched)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DB_CLIENT_ID", raising=False)
    monkeypatch.delenv("DB_API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)

    def start(api: FakeApi) -> AppTest:
        monkeypatch.setattr(httpx, "get", api.get)
        monkeypatch.setattr(httpx, "post", api.post)
        st.cache_data.clear()  # the app caches API answers; one test must not see the last test's
        app = AppTest.from_file(str(APP), default_timeout=60)
        return app.run()

    return start


def widget(widgets, label):
    """Find a widget by its label, e.g. widget(app.selectbox, "Train type")."""
    return next(w for w in widgets if w.label == label)


def test_shows_a_hint_when_the_api_is_down(run_app):
    app = run_app(FakeApi(reachable=False))
    assert "not reachable" in app.error[0].value
    assert not app.tabs  # st.stop(): nothing below the error is drawn


def test_page_shows_the_model_version(run_app):
    app = run_app(FakeApi())
    assert not app.exception
    assert "Model v7" in app.caption[0].value
    assert widget(app.sidebar.selectbox, "Station").value == "Köln Hbf"  # preselected


def test_without_db_key_the_timetable_tab_points_to_the_form(run_app):
    app = run_app(FakeApi())
    assert any("No DB API key" in info.value for info in app.info)


def test_single_train_form_predicts(run_app):
    api = FakeApi()
    app = run_app(api)
    widget(app.selectbox, "Train type").select("RE")
    widget(app.button, "Predict").click()
    app.run()

    assert app.metric[0].value == "35%"
    [body] = api.sent
    [arrival] = body["arrivals"]
    assert arrival["station"] == "Köln Hbf"
    assert arrival["train_type"] == "RE"
    assert isinstance(arrival["planned_arrival"], str)  # JSON has no datetime: sent as ISO text


def test_refused_request_is_shown_not_a_crash(run_app):
    app = run_app(FakeApi(predict_status=422))
    widget(app.selectbox, "Train type").select("RE")
    widget(app.button, "Predict").click()
    app.run()

    assert not app.exception
    assert "Unknown station" in app.error[0].value


@pytest.fixture
def db_key(monkeypatch):
    """A fake DB key, and a DBApiClient that never opens a connection."""
    monkeypatch.setenv("DB_CLIENT_ID", "fake-id")
    monkeypatch.setenv("DB_API_KEY", "fake-key")
    monkeypatch.setattr(db_api.DBApiClient, "__enter__", lambda self: self)
    monkeypatch.setattr(db_api.DBApiClient, "__exit__", lambda self, *args: None)


def test_no_published_timetable_is_explained(run_app, monkeypatch, db_key):
    # What the DB does for hours more than ~18 hours ahead: 404, so planned_arrivals finds nothing
    monkeypatch.setattr(timetable, "planned_arrivals", lambda client, station, day, hours: [])
    app = run_app(FakeApi())
    assert any("only about 18 hours ahead" in info.value for info in app.info)


def test_days_after_tomorrow_do_not_ask_the_db(run_app, monkeypatch, db_key):
    calls = []
    monkeypatch.setattr(timetable, "planned_arrivals", lambda *args: calls.append(args) or [])
    app = run_app(FakeApi())
    widget(app.sidebar.date_input, "Day").set_value(date.today() + timedelta(days=3))
    app.run()

    assert any("Choose today or tomorrow" in info.value for info in app.info)
    assert len(calls) == 1  # only the first run (today), not the one for the day in 3 days


def test_timetable_shows_trains_and_skips_buses(run_app, monkeypatch, db_key):
    def fake_arrivals(client, station, day, hours):
        def arrival(train_type, number):
            return {
                "station": station,
                "train_type": train_type,
                "train_number": number,
                "line_number": "1",
                "final_destination": "Aachen Hbf",
                "stop_number": 5,
                "planned_arrival": TOMORROW,
                "ride_start": TOMORROW - timedelta(minutes=31),
            }

        return [arrival("RE", "10123"), arrival("Bus", "42")]

    monkeypatch.setattr(timetable, "planned_arrivals", fake_arrivals)
    api = FakeApi()
    app = run_app(api)

    assert not app.exception
    table = app.dataframe[0].value
    assert list(table["train"]) == ["RE 10123"]  # the bus is not sent to the model
    assert any("1 arrivals not shown" in caption.value for caption in app.caption)
    assert len(api.sent[0]["arrivals"]) == 1
