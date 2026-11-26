"""Tests for D6: predictions are logged, and logged predictions are matched with real arrivals."""

import json
from datetime import datetime

import polars as pl
import pytest

from db_delay_mlops.api.prediction_log import log_predictions
from db_delay_mlops.api.schemas import ArrivalRequest, Prediction
from db_delay_mlops.monitoring.served import served_performance

TOMORROW = datetime.now().replace(hour=17, minute=42, second=0, microsecond=0)


def arrival(hour: int) -> ArrivalRequest:
    planned = TOMORROW.replace(hour=hour)
    return ArrivalRequest.model_construct(  # model_construct: no checks, the time window does not matter here
        station="Köln Hbf",
        train_type="RE",
        line_number="1",
        final_destination="Aachen Hbf",
        stop_number=5,
        planned_arrival=planned,
        ride_start=planned.replace(minute=0),
    )


def prediction(p: float) -> Prediction:
    return Prediction(probability_late=p, risk="low", top_reasons=[])


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
    return tmp_path


def test_every_prediction_becomes_one_line(log_dir):
    log_predictions([arrival(8), arrival(17)], [prediction(0.1), prediction(0.6)], model_version="2")
    [file] = log_dir.glob("predictions-*.jsonl")
    lines = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[1]["station"] == "Köln Hbf"
    assert lines[1]["probability_late"] == 0.6
    assert lines[1]["model_version"] == "2"


def test_logging_appends(log_dir):
    log_predictions([arrival(8)], [prediction(0.1)], model_version="2")
    log_predictions([arrival(9)], [prediction(0.2)], model_version="2")
    [file] = log_dir.glob("predictions-*.jsonl")
    assert len(file.read_text(encoding="utf-8").splitlines()) == 2


def test_a_broken_log_folder_does_not_break_the_answer(monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file where the folder should be")
    monkeypatch.setenv("PREDICTION_LOG_DIR", str(blocker))
    log_predictions([arrival(8)], [prediction(0.1)], model_version="2")  # must not raise


def test_served_predictions_are_matched_with_real_arrivals(log_dir, tmp_path):
    log_predictions(
        [arrival(8), arrival(17), arrival(17), arrival(20)],  # 17:42 asked twice: counted once
        [prediction(0.1), prediction(0.7), prediction(0.7), prediction(0.4)],
        model_version="2",
    )
    month = f"{TOMORROW:%Y-%m}"
    data_dir = tmp_path / "arrivals"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "stada_name": ["Köln Hbf"] * 2,
            "arrival_planned_time": [TOMORROW.replace(hour=8), TOMORROW.replace(hour=17)],
            "final_destination_station": ["Aachen Hbf"] * 2,
            "y": [0, 1],  # the 20:42 train is not in the data (e.g. cancelled): no match
        },
        schema_overrides={"arrival_planned_time": pl.Datetime("ns")},
    ).write_parquet(data_dir / f"arrivals-{month}.parquet")

    result = served_performance(month, directory=log_dir, data_dir=data_dir)
    assert result["served_n"] == 3
    assert result["served_matched"] == 2
    assert result["served_roc_auc"] == 1.0  # 0.1 for the punctual, 0.7 for the late train


def test_no_logs_means_nothing_to_report(tmp_path):
    assert served_performance("2026-11", directory=tmp_path) is None
