"""Tests for monitoring: the PSI must see drift where there is some, and the rules must decide as documented."""

import numpy as np
import polars as pl
import pytest

from db_delay_mlops.monitoring.decide import decide, months_between
from db_delay_mlops.monitoring.drift import categorical_psi, drift_table, numeric_psi, psi

rng = np.random.default_rng(0)


def test_psi_is_zero_for_identical_shares():
    shares = np.array([0.2, 0.3, 0.5])
    assert psi(shares, shares) == pytest.approx(0.0)


def test_numeric_psi_stable_vs_shifted():
    reference = pl.Series("temperature_2m", rng.normal(10, 5, 50_000))
    same = pl.Series("temperature_2m", rng.normal(10, 5, 50_000))
    winter = pl.Series("temperature_2m", rng.normal(0, 4, 50_000))
    assert numeric_psi(reference, same) < 0.01
    assert numeric_psi(reference, winter) > 0.25


def test_numeric_psi_with_many_equal_values():
    # precipitation is 0.0 most of the time: equal quantiles must not break the bins
    reference = pl.Series("precipitation", np.where(rng.random(10_000) < 0.8, 0.0, rng.exponential(1, 10_000)))
    assert numeric_psi(reference, reference) == pytest.approx(0.0)


def test_categorical_psi_sees_a_shift_between_groups():
    reference = pl.Series("train_group", ["s_bahn"] * 50 + ["regional"] * 40 + ["long_distance"] * 10)
    more_ice = pl.Series("train_group", ["s_bahn"] * 30 + ["regional"] * 30 + ["long_distance"] * 40)
    assert categorical_psi(reference, reference) == pytest.approx(0.0)
    assert categorical_psi(reference, more_ice) > 0.25


def test_categorical_psi_handles_new_values_and_nulls():
    reference = pl.Series("line", ["RE1"] * 50 + ["S1"] * 50)
    current = pl.Series("line", ["RE1"] * 40 + ["NEW9"] * 30 + [None] * 30)  # never seen + missing
    assert categorical_psi(reference, current) > 0.25


def test_drift_table_has_one_row_per_feature_strongest_first():
    n = 5_000
    table = pl.DataFrame(
        {
            "train_line_station_num": rng.integers(1, 30, n),
            "minutes_since_departure": rng.integers(0, 300, n),
            "temperature_2m": rng.normal(10, 5, n),
            "precipitation": rng.exponential(0.5, n),
            "stada_name": rng.choice(["Köln Hbf", "Bonn Hbf"], n),
            "line": rng.choice(["RE1", "S1"], n),
            "final_destination_station": rng.choice(["Aachen Hbf", "Bonn Hbf"], n),
            "train_group": rng.choice(["s_bahn", "regional"], n),
            "hour": rng.integers(0, 24, n),
            "weekday": rng.integers(1, 8, n),
            "weather_code": rng.choice([0, 3, 61], n),
            "is_holiday": rng.random(n) < 0.05,
        }
    )
    colder = table.with_columns(pl.col("temperature_2m") - 10)
    result = drift_table(table, colder)
    assert result.height == 12  # every feature except "month"
    assert result.row(0, named=True)["feature"] == "temperature_2m"
    assert result.row(0, named=True)["level"] == "strong"


def performance(**changes) -> dict:
    """The champion doing exactly as well as at registration."""
    base = {
        "champion_version": "2",
        "champion_month": "2026-10",
        "roc_auc": 0.80,
        "pr_auc": 0.54,
        "brier": 0.14,
        "late_share": 0.22,
        "mean_prediction": 0.23,
        "baseline_roc_auc": 0.80,
        "baseline_pr_auc": 0.54,
        "baseline_late_share": 0.22,
    }
    return base | changes


NO_DRIFT = pl.DataFrame({"feature": ["hour"], "psi": [0.01], "level": ["stable"]})
WINTER = pl.DataFrame({"feature": ["temperature_2m"], "psi": [0.9], "level": ["strong"]})


def test_months_between():
    assert months_between("2026-10", "2026-11") == 1
    assert months_between("2026-10", "2027-01") == 3


def test_all_fine_means_no_retraining():
    decision = decide("2026-11", performance(), NO_DRIFT)
    assert not decision.retrain
    assert decision.warnings == []


def test_drift_alone_only_warns():
    # A cold month is different, but the model has seen last winter: not a reason to retrain
    decision = decide("2026-11", performance(), WINTER)
    assert not decision.retrain
    assert "temperature_2m" in decision.warnings[0]


def test_worse_model_means_retraining():
    decision = decide("2026-11", performance(roc_auc=0.77), NO_DRIFT)
    assert decision.retrain
    assert "ROC-AUC" in decision.reasons[0]


def test_small_dip_is_tolerated():
    assert not decide("2026-11", performance(roc_auc=0.79), NO_DRIFT).retrain


def test_old_champion_is_retrained_anyway():
    decision = decide("2027-01", performance(), NO_DRIFT)
    assert decision.retrain
    assert "3 months old" in decision.reasons[0]


def test_calibration_gap_and_late_share_only_warn():
    decision = decide("2026-11", performance(late_share=0.15, mean_prediction=0.23), NO_DRIFT)
    assert not decision.retrain
    assert len(decision.warnings) == 2  # late share moved + predictions too high
