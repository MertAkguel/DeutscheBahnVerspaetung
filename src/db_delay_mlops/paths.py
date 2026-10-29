"""Absolute project paths, so code works no matter where it is started (repo root, notebooks/, Airflow)."""

from pathlib import Path

# src/db_delay_mlops/paths.py -> parents[2] is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

STATIONS_FILE = CONFIG_DIR / "stations.csv"
TRAIN_TYPE_GROUPS_FILE = CONFIG_DIR / "train_type_groups.csv"
WEATHER_DIR = DATA_DIR / "raw" / "weather"
