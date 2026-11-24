"""Tests for api/schemas.py: which requests Pydantic lets through and which it refuses."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from db_delay_mlops.api.schemas import ArrivalRequest, BatchRequest, risk_level, to_frame
from db_delay_mlops.features.serving import REQUEST_SCHEMA

# Relative to today: a fixed date would become "in the past" and break the test next week
TOMORROW_17_42 = (datetime.now() + timedelta(days=1)).replace(hour=17, minute=42, second=0, microsecond=0)


def body(**changes) -> dict:
    """A valid request as JSON would deliver it (times as text, like over HTTP)."""
    data = {
        "station": "Köln Hbf",
        "train_type": "RE",
        "line_number": "1",
        "final_destination": "Aachen Hbf",
        "stop_number": 12,
        "planned_arrival": TOMORROW_17_42.isoformat(),
        "ride_start": (TOMORROW_17_42 - timedelta(minutes=97)).isoformat(),
    }
    return data | changes


def test_valid_request_is_accepted_and_converted():
    arrival = ArrivalRequest.model_validate(body())
    assert arrival.planned_arrival == TOMORROW_17_42  # the text became a real datetime
    assert arrival.stop_number == 12


def test_missing_line_is_allowed():
    assert ArrivalRequest.model_validate(body(line_number=None)).line_number is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"station": "Berlin Hbf"}, "did you mean"),
        ({"train_type": "Bus"}, "not supported"),
        ({"stop_number": 0}, "greater than or equal to 1"),
        ({"stop_number": "twelve"}, "valid integer"),
        ({"planned_arrival": "tomorrow"}, "datetime"),
        ({"ride_start": (TOMORROW_17_42 + timedelta(hours=1)).isoformat()}, "before planned_arrival"),
        ({"planned_arrival": (TOMORROW_17_42 + timedelta(days=30)).isoformat()}, "next 7 days"),
    ],
)
def test_bad_request_is_refused_with_a_clear_message(changes, message):
    with pytest.raises(ValidationError, match=message):
        ArrivalRequest.model_validate(body(**changes))


def test_missing_field_is_refused():
    data = body()
    del data["station"]
    with pytest.raises(ValidationError, match="station"):
        ArrivalRequest.model_validate(data)


def test_batch_must_not_be_empty():
    with pytest.raises(ValidationError):
        BatchRequest.model_validate({"arrivals": []})


@pytest.mark.parametrize(
    ("p", "level"),
    [(0.05, "low"), (0.19, "low"), (0.20, "medium"), (0.39, "medium"), (0.40, "high"), (1.0, "high")],
)
def test_risk_level(p, level):
    assert risk_level(p) == level


def test_to_frame_gives_the_columns_serving_expects():
    frame = to_frame([ArrivalRequest.model_validate(body()), ArrivalRequest.model_validate(body(line_number=None))])
    assert frame.columns == list(REQUEST_SCHEMA)
    assert frame["line_number"].to_list() == ["1", None]
