"""Planned arrivals of a station from the DB Timetables API, as requests for our prediction API.

The plan endpoint returns XML for one station and one hour. Each stop is an <s> element:

    <timetable station="Köln Hbf">
      <s id="-7874571842864554321-2611251605-5">      daily trip id - ride start YYMMDDHHmm - stop number
        <tl c="RE" n="10123" .../>                    train type (c) and train number (n)
        <ar pt="2611251742" l="1" ppth="A|B|C"/>      arrival: planned time, line, stations BEFORE this stop
        <dp pt="2611251744" l="1" ppth="D|E|Aachen Hbf"/>  departure: stations AFTER this stop
      </s>
    </timetable>

From that we build exactly the fields of ArrivalRequest. The id carries the ride start and
the stop number, like the "id" column of the Hugging Face data that our features come from.
"""

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime

import polars as pl

from db_delay_mlops.data.db_api import DBApiClient
from db_delay_mlops.features.station_info import station_info

ID_PATTERN = re.compile(r"-(\d{10})-(\d+)$")  # ...-<ride start YYMMDDHHmm>-<stop number>
TIME_FORMAT = "%y%m%d%H%M"


def parse_plan(xml: str, station: str) -> list[dict]:
    """Arrivals in one plan response. Stops without <ar> are first stops: they have no arrival."""
    root = ET.fromstring(xml)
    arrivals = []
    for stop in root.findall("s"):
        arrival, departure, train = stop.find("ar"), stop.find("dp"), stop.find("tl")
        match = ID_PATTERN.search(stop.get("id", ""))
        if arrival is None or train is None or match is None or arrival.get("pt") is None:
            continue
        # The last station after this stop is the destination; no departure = the train ends here
        after = departure.get("ppth") if departure is not None else None
        destination = after.split("|")[-1] if after else root.get("station")
        arrivals.append(
            {
                "station": station,
                "train_type": train.get("c"),
                "train_number": train.get("n"),  # for display only, not a model feature
                "line_number": arrival.get("l") or (departure.get("l") if departure is not None else None),
                "final_destination": destination,
                "stop_number": int(match.group(2)),
                "planned_arrival": datetime.strptime(arrival.get("pt"), TIME_FORMAT),
                "ride_start": datetime.strptime(match.group(1), TIME_FORMAT),
            }
        )
    return arrivals


def station_evas(station: str) -> list[str]:
    """All EVA numbers of a station (e.g. Berlin Hauptbahnhof has three levels)."""
    return station_info().filter(pl.col("stada_name") == station)["eva"].to_list()


def planned_arrivals(client: DBApiClient, station: str, day: date, hours: list[int]) -> list[dict]:
    """All planned arrivals at a station in the given hours, sorted by time.

    One API call per EVA number and hour (free plan: 60 calls per minute).
    """
    arrivals = []
    for eva in station_evas(station):
        for hour in hours:
            response = client.plan(eva, datetime(day.year, day.month, day.day, hour))
            if response.status_code == 200 and response.body:
                arrivals += parse_plan(response.body, station)
    return sorted(arrivals, key=lambda a: a["planned_arrival"])
