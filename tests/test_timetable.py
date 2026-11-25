"""Tests for data/timetable.py: plan XML -> requests for the prediction API."""

from datetime import datetime

from db_delay_mlops.api.schemas import ArrivalRequest
from db_delay_mlops.data.timetable import parse_plan, station_evas

# Shaped like a plan response of the DB Timetables API (shortened)
PLAN_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station="Köln Hbf">
  <s id="-7874571842864554321-2611251605-5">
    <tl f="N" t="p" o="800337" c="RE" n="10123"/>
    <ar pt="2611251742" pp="7" l="1" ppth="Hamm(Westf)Hbf|Dortmund Hbf|Düsseldorf Hbf"/>
    <dp pt="2611251744" pp="7" l="1" ppth="Düren|Aachen Hbf"/>
  </s>
  <s id="1234567890123456789-2611251750-1">
    <tl f="F" t="p" o="80" c="ICE" n="123"/>
    <dp pt="2611251750" pp="4" ppth="Frankfurt(Main)Hbf"/>
  </s>
  <s id="-555-2611251500-9">
    <tl f="F" t="p" o="80" c="ICE" n="456"/>
    <ar pt="2611251755" pp="2" ppth="Berlin Hbf|Hannover Hbf"/>
  </s>
</timetable>"""


def test_parse_plan_reads_arrivals():
    arrivals = parse_plan(PLAN_XML, "Köln Hbf")
    assert len(arrivals) == 2  # the second <s> starts here: no arrival
    re1 = arrivals[0]
    assert re1["train_type"] == "RE"
    assert re1["line_number"] == "1"
    assert re1["final_destination"] == "Aachen Hbf"  # last station after this stop
    assert re1["stop_number"] == 5
    assert re1["planned_arrival"] == datetime(2026, 11, 25, 17, 42)
    assert re1["ride_start"] == datetime(2026, 11, 25, 16, 5)


def test_train_that_ends_here_has_this_station_as_destination():
    ice = parse_plan(PLAN_XML, "Köln Hbf")[1]
    assert ice["final_destination"] == "Köln Hbf"
    assert ice["line_number"] is None


def test_parsed_arrivals_have_exactly_the_api_fields():
    # The demo sends these dicts to the API; train_number is only shown, not sent
    for arrival in parse_plan(PLAN_XML, "Köln Hbf"):
        request = {k: v for k, v in arrival.items() if k != "train_number"}
        assert set(request) == set(ArrivalRequest.model_fields)


def test_station_with_several_levels_has_several_evas():
    assert len(station_evas("Berlin Hauptbahnhof")) == 3
    assert station_evas("Köln Hbf")
