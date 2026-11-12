"""Feature lists and target, shared by all notebooks and scripts.

Defined once, so training, the API and monitoring always use the same columns.
"""

# Text columns: categories for tree models, one-hot encoded for linear models
TEXT_FEATURES = ["stada_name", "line", "final_destination_station", "train_group"]
# Numbers that are really categories: one-hot encoded for linear models, plain numbers for trees
CODE_FEATURES = ["hour", "weekday", "month", "weather_code"]
NUMERIC_FEATURES = [
    "train_line_station_num",
    "minutes_since_departure",
    "temperature_2m",
    "precipitation",
    "is_holiday",
]

FEATURES = TEXT_FEATURES + CODE_FEATURES + NUMERIC_FEATURES
TARGET = "y"
