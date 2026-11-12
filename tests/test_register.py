"""Tests for the quality gate rules in models/register.py."""

import pytest

from db_delay_mlops.models.register import beats_champion, is_older_than_champion


@pytest.mark.parametrize(
    ("new", "champion", "promote"),
    [
        (0.50, None, True),  # first model ever: nothing to beat
        (0.54, 0.47, True),  # better
        (0.5373, 0.5373, False),  # exactly as good (our first Airflow run): champion stays
        (0.45, 0.47, False),  # worse
    ],
)
def test_beats_champion(new, champion, promote):
    assert beats_champion(new, champion) is promote


@pytest.mark.parametrize(
    ("valid_month", "champion_month", "older"),
    [
        ("2026-09", "2026-10", True),  # backfill: must not replace the champion
        ("2025-12", "2026-01", True),  # across the year
        ("2026-10", "2026-10", False),  # same month: normal comparison
        ("2026-11", "2026-10", False),  # the normal monthly run
        ("2026-09", None, False),  # champion without month tag (v1): no guard
    ],
)
def test_is_older_than_champion(valid_month, champion_month, older):
    assert is_older_than_champion(valid_month, champion_month) is older
