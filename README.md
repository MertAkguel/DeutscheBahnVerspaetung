# Deutsche Bahn Delay Prediction (db-delay-mlops)

An end-to-end MLOps project that predicts whether a Deutsche Bahn train will arrive at a station **at least 6 minutes late**. It is built as the capstone project of the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by DataTalks.Club.

## Project idea

- **Question:** Will ICE 123 arrive on time at Köln Hbf today at 17:42?
- **ML task:** Binary classification with a probability. One row is one stop of one train at one station.
- **Target:** Arrival delay ≥ 6 minutes. The threshold follows the official DB punctuality definition: a stop counts as on time if it is less than 6 minutes late. The delay is computed from the arrival columns (`arrival_change_time − arrival_planned_time`), not from `delay_in_min`, which mostly describes the departure. Cancelled arrivals and the first stop of a ride (no arrival) are excluded. Details in [Target and features](#target-and-features).
- **Prediction time:** the day before. The model only uses information that is already known at that point, for example the weather forecast instead of the observed weather.
- **Why this fits MLOps:** New data arrives every month, and the true delay becomes known afterwards. Strikes, winter and construction work cause real drift. Deployment, monitoring and retraining follow naturally from that.

## Current status

As of 2026-10-29: data download, station list, own raw data collector, target definition and the first features (time, public holidays, weather forecast) are in place; data analysis is in progress.

| Component | Tool (planned) | Status |
| --- | --- | --- |
| Problem description | this README | 🔄 in progress |
| Data download and own collector | huggingface_hub, httpx, Polars | ✅ first version |
| Data analysis and baseline | Jupyter, Polars, Plotly | 🔄 in progress |
| Target and feature definition | this README, `src/db_delay_mlops/features/` | ✅ first version |
| Experiment tracking and model registry | MLflow | ⬜ open |
| Workflow orchestration | Apache Airflow 3 | ⬜ open |
| Model deployment | FastAPI, Docker | ⬜ open |
| Monitoring with retraining on drift | Evidently, Grafana | ⬜ open |
| Cloud and infrastructure as code | AWS, Terraform | ⬜ open |
| Best practices | pytest, ruff, pre-commit, Makefile, GitHub Actions | ⬜ open |

## Prerequisites

**Software**

| Tool | Version | Purpose |
| --- | --- | --- |
| Linux, macOS or Windows 11 with WSL2 (Ubuntu 24.04) | – | Development environment |
| Python | 3.12 | Project language |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.12 | Packages and virtual environment |
| Git | – | Version control |
| Docker with Docker Compose | – | Containers for the API, MLflow, Airflow and monitoring |

**Accounts and access**

- The main data needs **no account**. The dataset, weather and holidays are freely available.
- Optional: a free API key from the [DB API Marketplace](https://developers.deutschebahn.com/) for the Timetables and StaDa APIs. It is only needed to collect data yourself (Plan B, see below) or to rebuild `config/stations.csv`, which is already in the repository.
- A cloud account is only needed in the cloud phase.

**Disk space:** One monthly file is about 60 MB up to October 2025 and about 340 MB from November 2025 on. Plan for a few GB of free space for several months.

## Data

| Source | Content | Access | License |
| --- | --- | --- | --- |
| [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (Hugging Face) | Planned and actual times of all stops, one Parquet file per month (`monthly_processed_data/data-YYYY-MM.parquet`), from July 2024 | free, no sign-up | Data CC BY 4.0 (source: Deutsche Bahn), code MIT |
| [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | Weather forecasts made the day before, per station and hour, from January 2024. Cached per month in `data/raw/weather/`. | free, no API key, non-commercial | CC BY 4.0 |
| [`holidays`](https://pypi.org/project/holidays/) (Python package) | Public holidays per federal state | offline | MIT |
| [DB StaDa API](https://developers.deutschebahn.com/) | Station name, category, federal state and coordinates per EVA number, stored in `config/stations.csv` | free API key | see DB API Marketplace |
| [OpenHolidays API](https://www.openholidaysapi.org/en/) | School holidays per federal state (planned, not used yet) | free | see website |

**How the rail dataset is built:** According to its [GitHub README](https://github.com/piebro/deutsche-bahn-data), a GitHub Actions job queries the DB Timetables API four times a day (endpoints `plan` and `fchg`) and turns the raw responses into tables once a month.

**Coverage:** Until 2025-11-02 only the largest stations were collected, after that all of them. This project is limited to those 110 large stations (134 EVA numbers), so the history from July 2024 on is consistent.

**Weather without leakage:** For training we use the forecast that was made the day before (Open-Meteo `_previous_day1`), not the observed weather. This way the model sees the same information in training as it will in production.

**Plan B:** If the dataset is discontinued, the project collects the data itself through the [DB Timetables API](https://developers.deutschebahn.com/) with a free key. Scraping bahn.de is not an option.

## Stations

`config/stations.csv` lists every EVA number that appears in the dataset before the coverage change (146 in total). Each one was looked up in the DB StaDa API for name, category, federal state and coordinates.

- **134 EVA numbers** are rail stations known to StaDa (`collect = true`). They belong to **110 stations**: many large stations have a separate EVA number for their S-Bahn level, Berlin Hbf even has three.
- **12 EVA numbers** are bus stations or tram stops without a StaDa entry (`collect = false`).

| Column | Source | Meaning | Example |
| --- | --- | --- | --- |
| `eva` | dataset | EVA number: DB's ID of a stopping point, with a leading `0` as in the dataset. One station can have several. | `08000207` |
| `xml_station_name` | dataset | Most common name the Timetables API uses for this EVA number. Differs per level of a station. | `Berlin Hbf (S-Bahn)` |
| `n_stops` | dataset | Number of stop records (rows) for this EVA number in the 16 months before 2025-11 | `583068` |
| `n_months` | dataset | In how many of these 16 months the EVA number appears | `16` |
| `stada_name` | StaDa | Official station name, the same for all EVA numbers of one station | `Berlin Hauptbahnhof` |
| `category` | StaDa | DB station category from 1 (largest stations) to 7 (smallest) | `1` |
| `federal_state` | StaDa | Federal state, used for public holidays | `Nordrhein-Westfalen` |
| `lat`, `lon` | StaDa | Coordinates of this EVA number, used for weather and the map | `50.94303`, `6.958729` |
| `is_main_eva` | StaDa | Whether this is the main EVA number of the station | `true` |
| `collect` | derived | `true` if StaDa knows the EVA number: it is a rail station that we collect and model | `true` |

`notebooks/02_station_map.ipynb` draws the stations on an interactive map, coloured by their share of late arrivals. You can filter by train type group and by single train types.

## Train type groups

`train_type` in the data is often an operator brand (`NX` = National Express, `VIA`, `HLB` = Hessische Landesbahn), not a product. Brands also change over time: from 2025-11 on, some lines that were `RE` before appear as `NX`. The data contains 209 different values at our stations, most of them typos and replacement buses with only a few stops.

`config/train_type_groups.csv` maps each train type to one group:

| Group | Train types | Share of arrivals (2024-07 to 2026-09) |
| --- | --- | --- |
| `s_bahn` | 9, e.g. S, SBH (S-Bahn Hannover), AVG (Karlsruhe) | 46.3 % |
| `regional` | 43, e.g. RE, RB, IRE, NX, VIA, HLB, MEX | 39.6 % |
| `long_distance` | 18, e.g. ICE, IC, EC, NJ, FLX, TGV | 12.3 % |
| `exclude` | Bus (no real-time data: 86 % of arrivals exactly on time), CBG | 1.8 % |

- Only train types with at least 1,000 arrivals are mapped. They cover 99.987 % of all arrivals. The remaining 137 types (0.013 %) are treated as unmapped and left out.
- Operators were assigned by their most common line (`line_number`), e.g. `SWE` → `s_bahn` because its most common line is S2. Operators that run both S-Bahn and regional lines are therefore simplified. The `note` column explains every non-obvious choice.
- **Why groups instead of raw train types:** the group stays stable when an operator changes its brand, and it allows evaluating the model per group. The S-Bahn makes up almost half of all arrivals, so an overall metric alone would mostly describe the S-Bahn.
- **Planned check:** the feature pipeline will warn when a new month contains a train type that is not in the file, so brand changes are noticed instead of silently dropping trains.

## Target and features

### Prediction setting

The prediction is made **the day before**, e.g. at 18:00 for all stops of the next day. **Leakage rule:** a feature is only allowed if it is already known at that moment. Every feature below was checked against this rule.

### Rows: which stops are used

A row is one arrival of one train at one station. A row is kept if all of the following hold:

| Rule | Why |
| --- | --- |
| The EVA number is one of our 134 rail EVA numbers (`collect = true`, 110 stations) | Consistent history from 2024-07 on |
| `train_group` is `long_distance`, `regional` or `s_bahn` | Buses have no real-time data and would look punctual |
| The stop has a planned arrival (`arrival_planned_time` is set) | The first stop of a ride has no arrival |
| The arrival is not cancelled (`arrival_is_canceled = false`) | Follows the DB definition, see limitations |
| The stop is in the timetable (`has_plan_record = true`) | The day before, only planned stops are known. Unplanned extra stops and short-notice replacement trains do not exist yet at prediction time, so the model should not learn from them. |

Effect of the rules in July 2024 (`notebooks/03_column_selection.ipynb`):

| Step | Rows left |
| --- | --- |
| All rows at the 146 EVA numbers | 2,029,509 |
| + rail EVA number | 1,981,930 |
| + rail train group | 1,908,261 |
| + planned arrival (not the first stop) | 1,489,182 |
| + not cancelled | 1,418,566 |
| + in the timetable | **1,405,544** (20.7 % late) |

The first stop of a ride removes the most rows: many trains start at the large stations we cover.

### Target y

```
arrival_delay_min = arrival_change_time − arrival_planned_time
y = 1 if arrival_delay_min ≥ 6 else 0
```

- Binary classification; the model outputs the probability that y = 1.
- The **arrival** is used, because the question is whether the train *arrives* on time. `delay_in_min` from the dataset mostly describes the departure and is not used.
- Early arrivals (negative delay) count as on time.

### Features X

| Feature | Source | How it is computed | Why it may matter |
| --- | --- | --- | --- |
| `stada_name` | `config/stations.csv` | Official name of the station where this arrival happens, joined via `eva`. All EVA numbers of a station (e.g. its S-Bahn level) share it. | Some stations are bottlenecks (e.g. Hamm, Bielefeld) |
| `line_number` | dataset | Line of the train, e.g. `RE6`, `S1` | Lines share routes and their problems; more stable than the train number |
| `final_destination_station` | dataset | Last station of the ride | Direction and length of the route |
| `train_group` | `config/train_type_groups.csv` | `long_distance`, `regional` or `s_bahn`, mapped from `train_type` | Groups behave very differently (S-Bahn ~11 % late, ICE ~50 %). The group stays stable when an operator changes its brand (e.g. `RE` → `NX`), the raw train type does not. |
| `train_line_station_num` | dataset | Position of this stop within the ride (1 = first stop) | Delays build up along the ride |
| `minutes_since_departure` | dataset | `arrival_planned_time` minus the planned start time of the ride, which is part of the stop `id` (`YYMMDDHHmm`) | How long the train has been on its way |
| `hour` | `arrival_planned_time` | Hour of the planned arrival, 0–23, local German time | Rush hours: the late share rises from 9 % at 4:00 to 28 % at 17:00 (July 2024) |
| `weekday` | `arrival_planned_time` | Day of the week of the planned arrival, ISO numbering: **1 = Monday … 7 = Sunday** (Polars `.dt.weekday()`; Python's `datetime.weekday()` would give 0 = Monday) | Commuter days vs. weekend |
| `month` | `arrival_planned_time` | Month of the planned arrival, 1–12 | Seasons: winter, autumn leaves, summer heat |
| `is_holiday` | [`holidays`](https://pypi.org/project/holidays/) | Whether the date of the planned arrival is a public holiday in the federal state of the station | Different traffic on holidays |
| `temperature_2m` | Open-Meteo Previous Runs API | Temperature 2 m above ground, **forecast made the day before** (`_previous_day1`) for the station's coordinates and the hour of the planned arrival | Heat and frost cause failures of switches and air conditioning |
| `precipitation` | Open-Meteo Previous Runs API | Precipitation in mm, forecast made the day before, same station and hour | Heavy rain |
| `weather_code` | Open-Meteo Previous Runs API | WMO weather code, forecast made the day before, same station and hour. Covers e.g. thunderstorms, snowfall and heavy rain. Heat has no code; it is covered by `temperature_2m`. | Storms and snow disrupt traffic |

**Why the forecast and not the measured weather:** the measured weather of the travel day is not known the day before. A model trained on it would look good in tests and fail in production, because that value does not exist yet when the prediction is made.

**Why the planned and not the actual time:** all time-based features (`hour`, `weekday`, `month`, `is_holiday`, weather) are computed from `arrival_planned_time`. The dataset column `time` and `arrival_change_time` hold the actual time, which already contains the delay.

### How the features are computed

Holidays and weather only need **a time column and a station column**. Federal state and coordinates are looked up in `config/stations.csv` in the background, so the same functions work for training, the API and the demo app.

```python
from db_delay_mlops.features.holidays import add_is_holiday, is_holiday
from db_delay_mlops.features.weather import add_weather, weather_at

# whole DataFrame
arrivals = add_is_holiday(arrivals, time_col="arrival_planned_time", station_col="stada_name")
arrivals = add_weather(arrivals, time_col="arrival_planned_time", station_col="stada_name")

# single stop
is_holiday(datetime(2024, 10, 31, 8, 0), "Hamburg Hbf")  # True: Reformationstag
weather_at(datetime(2024, 7, 15, 17, 42), "Köln Hbf")   # {'temperature_2m': 27.9, 'precipitation': 0.1, 'weather_code': 3}
```

| Module | What it does |
| --- | --- |
| `features/station_info.py` | Reads `config/stations.csv` once (cached) and maps a station name or EVA number to federal state and coordinates. All EVA numbers of a station use the coordinates of its main EVA number. Unknown stations raise an error instead of silently returning empty features. |
| `features/holidays.py` | Builds a table of all public holidays per federal state with the `holidays` package and joins it on (federal state, date of the planned arrival). Duplicates are removed, so two holidays on one day cannot duplicate rows. |
| `features/weather.py` | Finds every month that occurs in the time column, loads it from `data/raw/weather/weather-YYYY-MM.parquet` or fetches it from Open-Meteo first (25 stations per request, 1 s pause), and joins on (station, hour of the planned arrival). `weather_category()` turns WMO codes into readable categories (clear, cloudy, fog, drizzle, rain, heavy rain, snow, thunderstorm). |
| `paths.py` | Absolute project paths, so the code works from the repository root, from `notebooks/` and later from Airflow |

Technical details that matter for correctness:

- **Time zone:** the timetable data and the weather are both in local German time (`Europe/Berlin`). On the night of the autumn clock change, 02:00 exists twice; the weather keeps the first value.
- **Time precision:** the timetable uses nanoseconds, the weather microseconds. Join keys must have the same type, so both are cast to microseconds before the join.
- **Month boundaries:** a monthly file also contains arrivals planned just before midnight of the previous month. `add_weather` loads every month that occurs, so these rows get their weather too.

### Columns that are not used

| Column | Reason |
| --- | --- |
| `delay_in_min`, `arrival_change_time`, `departure_change_time`, `time` | The outcome itself, only known after the ride |
| `arrival_is_canceled`, `departure_is_canceled` | Only known on the travel day; cancelled arrivals are removed |
| `is_replacement_train`, `is_additional_stop`, `replaced_train_type`, `replaced_train_number` | Mostly decided at short notice, not known the day before |
| `train_type` | Replaced by `train_group`, because brands change over time |
| `train_number` | About 15,000 different values that are partly reassigned at the December timetable change; `line_number` carries most of the information |
| `xml_station_name` | Replaced by `stada_name`, which is the same for all levels of a station |
| `eva`, `id`, `train_line_ride_id` | IDs, used for joins and grouping, not as features |

**Considered for later:** school holidays and bridge days, wind speed and gusts, station size (`category`), delay rate of a line in the past 30 days (only computed from data up to the end of the previous month, so training and production see the same information).

## First findings from the data analysis

All numbers refer to arrivals that pass the row rules above.

**Late share by weather forecast, July 2024** (1,405,544 arrivals, 20.7 % late):

| Forecast | Arrivals | Late share |
| --- | --- | --- |
| heavy rain | 55 | 25.5 % |
| rain | 179,637 | 23.0 % |
| thunderstorm | 15,775 | 22.9 % |
| cloudy | 1,057,607 | 20.5 % |
| clear | 147,193 | 19.1 % |
| drizzle | 628 | 14.5 % |
| fog | 4,649 | 13.7 % |

| Forecast temperature | Arrivals | Late share |
| --- | --- | --- |
| 10–20 °C | 644,619 | 18.1 % |
| 20–25 °C | 491,485 | 21.4 % |
| 25–30 °C | 222,428 | 24.4 % |
| ≥ 30 °C | 46,794 | 30.2 % |

**The time of day is a confounder.** It drives both the weather and the delays: fog is almost only forecast between 4:00 and 7:00, when only about 10 % of arrivals are late; heat above 30 °C only between 11:00 and 19:00, when up to 28 % are late. Comparing within the same hours gives a fairer picture:

| Comparison | Without | With | Effect |
| --- | --- | --- | --- |
| Fog, only 4:00–8:59 | 13.0 % | 12.7 % | about 0 |
| Heat ≥ 30 °C, only 15:00–18:59 | 25.6 % | 31.8 % | +6 percentage points |
| Heat ≥ 30 °C, all hours | 18.1 % | 30.2 % | +12 percentage points |

So the "fog makes trains punctual" effect disappears completely, and about half of the heat effect is due to the time of day. A model that gets `hour` and the weather at the same time can separate these effects. Feature importances and SHAP values still only show what **helps the prediction**, not what **causes** delays. Public holidays have a similar confounder: on holidays a Sunday timetable with fewer trains and fewer commuters runs, so `is_holiday` must be read together with `weekday`.

**Other findings:**

- **Train groups differ strongly:** in September 2026 about 12 % of S-Bahn arrivals but 55 % of ICE arrivals were late.
- **Local hot spots:** in September 2026, 76 % of ICE arrivals in Hamm and 83 % in Bielefeld were late, compared with 36 % in Hamm across all trains in July 2024. This is the kind of real drift the monitoring has to detect.
- **Small samples mislead:** a station with a single diverted ICE that was late shows "100 % late". The station map therefore hides stations with fewer than 30 arrivals of the selected train types.
- **Buses have no real-time data:** 86 % of bus arrivals are exactly on time, so buses are excluded.

## Project structure

```
config/
  stations.csv              # 146 EVA numbers with StaDa master data
  train_type_groups.csv     # train type -> long_distance / regional / s_bahn / exclude
notebooks/
  01_explore_hf_data.ipynb  # first look at the data
  02_station_map.ipynb      # interactive map, filter by train group and train type
  03_column_selection.ipynb # row rules, target, time, holiday and weather features
src/db_delay_mlops/
  paths.py                  # absolute project paths
  data/
    download.py             # monthly files from Hugging Face, pinned to a commit
    stations.py             # builds config/stations.csv with the StaDa API
    db_api.py               # DB API client: rate limits, retries, raw responses
    collect.py              # own collection of raw plan and fchg responses
  features/
    station_info.py         # station -> federal state, coordinates
    holidays.py             # is_holiday
    weather.py              # day-ahead weather forecast, weather categories
data/                       # not in git: downloaded and collected data
```

## Getting the data

All commands run from the repository root.

```bash
# 1. Monthly files from Hugging Face (about 5 GB for 2024-07 to 2026-09)
uv run python -m db_delay_mlops.data.download --start 2024-07 --end 2026-09

# 2. Station list (needs a DB API key in .env and the months before 2025-11 from step 1)
uv run python -m db_delay_mlops.data.stations

# 3. Own collection of raw data: current changes + plan of the next hour (needs a DB API key)
uv run python -m db_delay_mlops.data.collect run
```

The DB API key goes into a `.env` file that is never committed:

```
DB_CLIENT_ID=...
DB_API_KEY=...
```

The download pins the dataset to a fixed Hugging Face commit, so a repeated download returns the same files.

Weather data needs no command: `add_weather` fetches every missing month from Open-Meteo on first use and caches it in `data/raw/weather/`. The free API allows 600 calls per minute, 5,000 per hour and 10,000 per day; one month for all 110 stations costs about 220 calls.

## Coverage and limitations

- **Only 110 large stations.** Predictions only cover arrivals at these stations. Smaller stations have been in the data only since November 2025 and have few stops, which is not enough for a fair test.
- **Not representative of DB as a whole.** Large hubs collect delays from many directions. At our stations 20–24 % of arrivals are at least 6 minutes late (2024-07 to 2026-09), while across all stations in September 2026 it was about 16 %.
- **Cancellations are excluded from the target.** About 4 % of planned arrivals at our stations are cancelled (3.9–4.6 % in 2024-07, 2025-01 and 2026-09). Cancellations cluster on chaotic days (storms, strikes), so the remaining data looks better than reality. The cancellation share is therefore tracked separately in monitoring. A three-class target (on time / late / cancelled) is possible future work.
- **"Actual" times are the last reported times**, not measured arrivals. If a delay was never captured by a snapshot of the `fchg` endpoint, the stop keeps its planned time and looks on time.
- **Minute resolution.** Delays are only known in whole minutes, so stops near the 6-minute threshold are uncertain.
- **Weather only at the station of arrival.** A storm 200 km earlier on the route of an ICE is not seen by the model.
- **Weather is a forecast.** It can be wrong; that is intended, because production has the same uncertainty.
- **Regional holidays below state level are not covered**, e.g. the Augsburger Friedensfest or Assumption Day only in parts of Bavaria.
- **Train type groups are simplified.** Operators that run both S-Bahn and regional lines were assigned by their most common line.
- **Correlation is not causation.** The findings above describe associations in the data; confounders such as the time of day can create or hide effects.

## Attribution

Rail data: Deutsche Bahn, prepared by [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data), CC BY 4.0. Weather data: [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0.

The own collector in `src/db_delay_mlops/data/` (`db_api.py`, `collect.py`) is a rewrite inspired by the collection approach of [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data) (MIT License, © Piet Brömmel & Contributors): store raw API responses unchanged, then parse them in a separate step.
