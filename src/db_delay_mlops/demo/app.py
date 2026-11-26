"""Streamlit demo: which arrivals at a station are likely to be late in the next hours?

The app never loads the model. It asks the timetable (DB Timetables API) which trains are
planned, and our prediction API how likely each one is to be late. The DB publishes the
timetable only about 18 hours ahead; for later days the tab "Single train" still works. So it shows exactly what
any other client of the API would get, and every answer is logged for monitoring (D6).

Start:  make up  ->  http://localhost:8501
Locally: uv run streamlit run src/db_delay_mlops/demo/app.py  (API_URL defaults to localhost:8000)

Streamlit runs this file from top to bottom on every click. @st.cache_data keeps results
of slow calls (timetable, station list), so a click does not repeat them.
"""

import os
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from db_delay_mlops.data.db_api import DBApiClient, DBApiSettings
from db_delay_mlops.data.timetable import planned_arrivals

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_HOURS = 4  # one timetable call per hour and station level: keep within the free 60 calls per minute
# The DB Timetables API answers 404 for hours further ahead (measured: about 18-20 hours, not documented),
# so the timetable tab offers only today and tomorrow and says so when the DB has no plan yet.
TIMETABLE_DAYS = 1
RISK_ICON = {"low": "🟢 low", "medium": "🟠 medium", "high": "🔴 high"}
FEATURE_NAMES = {
    "stada_name": "station",
    "line": "line",
    "final_destination_station": "destination",
    "train_group": "train group",
    "hour": "time of day",
    "weekday": "weekday",
    "month": "month",
    "weather_code": "weather forecast",
    "train_line_station_num": "stop number on the ride",
    "minutes_since_departure": "minutes since departure",
    "temperature_2m": "temperature forecast",
    "precipitation": "rain forecast",
    "is_holiday": "public holiday",
}


@st.cache_data(ttl=3600)
def api_get(path: str):
    response = httpx.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def predict(arrivals: list[dict]) -> dict:
    """Send arrivals to POST /predict/batch (datetimes as ISO text, like any JSON client)."""
    body = {
        "arrivals": [
            {k: v.isoformat() if isinstance(v, datetime) else v for k, v in a.items() if k != "train_number"}
            for a in arrivals
        ]
    }
    response = httpx.post(f"{API_URL}/predict/batch", json=body, timeout=120)
    if response.status_code == 422:
        st.error(f"The API refused the request: {response.json()['detail']}")
        st.stop()
    response.raise_for_status()
    return response.json()


def timetable_available() -> bool:
    """Is a DB API key configured (DB_CLIENT_ID and DB_API_KEY)? Never shows the key itself."""
    try:
        DBApiSettings()
        return True
    except ValidationError:
        return False


@st.cache_data(ttl=3600, show_spinner="Loading the timetable ...")
def load_timetable(station: str, day: date, hours: tuple[int, ...]) -> list[dict]:
    with DBApiClient(DBApiSettings(), max_retries=2) as client:
        return planned_arrivals(client, station, day, list(hours))


def show_reasons(prediction: dict) -> None:
    """Bar chart of the strongest reasons: + makes a delay more likely, - less likely."""
    reasons = pd.DataFrame(prediction["top_reasons"])
    reasons["feature"] = reasons["feature"].map(FEATURE_NAMES)
    st.bar_chart(reasons, x="feature", y="contribution", horizontal=True)
    st.caption("Positive: makes a delay more likely. Negative: less likely (log-odds, from LightGBM).")
    for warning in prediction["warnings"]:
        st.warning(warning)


# ---------------------------------------------------------------- page
st.set_page_config(page_title="DB delay risk", page_icon="🚆", layout="wide")
st.title("🚆 Will my train be late?")

try:
    model = api_get("/model")
except httpx.HTTPError:
    st.error(f"The prediction API at {API_URL} is not reachable. Start the stack with: make up")
    st.stop()

st.caption(
    f"Probability of arriving **6 or more minutes late** (DB's punctuality limit), predicted the day before "
    f"from the timetable and the weather forecast. Model v{model['model_version']}, "
    f"validated on {model['valid_month']}."
)

with st.sidebar:
    stations = api_get("/stations")
    station = st.selectbox("Station", stations, index=stations.index("Köln Hbf") if "Köln Hbf" in stations else 0)
    today = date.today()
    day = st.date_input("Day", value=today, min_value=today, max_value=today + timedelta(days=6))
    now_hour = datetime.now().hour  # default: the next hours, the timetable is always published for them
    first, last = st.slider("Arrivals between", 0, 23, (min(now_hour, 21), min(now_hour + 2, 23)), format="%d:00")
    st.caption("Risk: 🟢 below 20 %, 🟠 20–40 %, 🔴 40 % or more. On average about 22 % of arrivals are late.")

timetable_tab, manual_tab = st.tabs(["Timetable", "Single train"])

with timetable_tab:
    if not timetable_available():
        st.info("No DB API key configured (DB_CLIENT_ID, DB_API_KEY in .env). Use the tab 'Single train'.")
    elif day > today + timedelta(days=TIMETABLE_DAYS):
        st.info(
            "The DB publishes the timetable only about 18 hours ahead. Choose today or tomorrow, "
            "or use the tab 'Single train' for a later day."
        )
    elif last - first + 1 > MAX_HOURS:
        st.info(f"Please choose at most {MAX_HOURS} hours.")
    else:
        arrivals = load_timetable(station, day, tuple(range(first, last + 1)))
        known_types = set(api_get("/train-types"))
        earliest, latest = datetime.now() - timedelta(hours=1), datetime.now() + timedelta(days=7)  # API's window
        usable = [a for a in arrivals if a["train_type"] in known_types and earliest <= a["planned_arrival"] <= latest]
        if not arrivals:
            st.info(
                "The DB has no timetable for this time yet: it is published only about 18 hours ahead. "
                "Choose an earlier time, or use the tab 'Single train'."
            )
        elif not usable:
            st.info("No arrivals of trains the model knows in this time window.")
        else:
            result = predict(usable)
            table = pd.DataFrame(
                {
                    "arrival": [a["planned_arrival"].strftime("%H:%M") for a in usable],
                    "train": [f"{a['train_type']} {a['train_number'] or ''}".strip() for a in usable],
                    "line": [a["line_number"] or "" for a in usable],
                    "to": [a["final_destination"] for a in usable],
                    "late probability": [p["probability_late"] for p in result["predictions"]],
                    "risk": [RISK_ICON[p["risk"]] for p in result["predictions"]],
                }
            )
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "late probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
                },
            )
            skipped = len(arrivals) - len(usable)
            if skipped:
                st.caption(f"{skipped} arrivals not shown (buses or other train types the model was not trained on).")

            labels = [f"{row.arrival}  {row.train}  to {row.to}" for row in table.itertuples()]
            choice = st.selectbox("Why? Choose a train", range(len(labels)), format_func=lambda i: labels[i])
            show_reasons(result["predictions"][choice])

with manual_tab:
    with st.form("single"):
        left, right = st.columns(2)
        train_type = left.selectbox("Train type", api_get("/train-types"), index=None, placeholder="e.g. RE")
        line = left.text_input("Line", "1")
        destination = left.text_input("Final destination", "Aachen Hbf")
        stop_number = right.number_input("Stop number on the ride", min_value=1, value=5)
        # Default: in one hour (the day defaults to today, and the API refuses arrivals in the past)
        in_an_hour = min(datetime.now() + timedelta(hours=1), datetime.combine(today, datetime.max.time()))
        arrival_time = right.time_input("Planned arrival", value=in_an_hour.time().replace(second=0, microsecond=0))
        minutes = right.number_input("Minutes since the train started", min_value=0, value=31)
        submitted = st.form_submit_button("Predict")
    if submitted and train_type:
        planned = datetime.combine(day, arrival_time)
        single = {
            "station": station,
            "train_type": train_type,
            "line_number": line or None,
            "final_destination": destination or None,
            "stop_number": int(stop_number),
            "planned_arrival": planned,
            "ride_start": planned - timedelta(minutes=int(minutes)),
        }
        [prediction] = predict([single])["predictions"]
        st.metric("Probability of arriving 6+ minutes late", f"{prediction['probability_late']:.0%}")
        st.write(RISK_ICON[prediction["risk"]])
        show_reasons(prediction)
