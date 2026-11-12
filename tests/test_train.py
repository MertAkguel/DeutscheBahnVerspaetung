"""Tests for models/train.py: which months a run trains on."""

import pytest

from db_delay_mlops.models.train import previous_months


def test_previous_months_within_a_year():
    assert previous_months("2026-09", 3) == ["2026-06", "2026-07", "2026-08"]


def test_previous_months_crosses_the_year():
    # The bug that is easy to make: January minus one month is December of the YEAR BEFORE
    assert previous_months("2026-02", 3) == ["2025-11", "2025-12", "2026-01"]


@pytest.mark.parametrize("month", ["2026-01", "2026-06", "2026-12"])
def test_previous_months_twelve_is_one_full_year(month):
    months = previous_months(month, 12)
    assert len(months) == 12
    assert months == sorted(months)  # oldest first
    assert month not in months  # the month itself is never part of its own training
