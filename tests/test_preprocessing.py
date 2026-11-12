"""Tests for models/preprocessing.py: text columns -> categories that match the training data."""

import pandas as pd
import pytest

from db_delay_mlops.models.preprocessing import CategoryCaster


@pytest.fixture
def train():
    """A tiny training table: two stations, one train line missing."""
    return pd.DataFrame(
        {
            "stada_name": ["Köln Hbf", "Berlin Hbf", "Köln Hbf"],
            "line": ["ICE10", None, "RE1"],
            "hour": [7, 8, 9],  # not a text column: must stay untouched
        }
    )


@pytest.fixture
def caster(train):
    return CategoryCaster(columns=["stada_name", "line"]).fit(train)


def test_categories_are_learned_from_train_only(caster):
    assert caster.categories_["stada_name"] == ["Berlin Hbf", "Köln Hbf"]
    assert caster.categories_["line"] == ["ICE10", "RE1"]  # None is not a category


def test_same_station_gets_the_same_code_in_any_table(caster, train):
    # valid has the stations in another order: the numbers inside must still match train
    valid = pd.DataFrame({"stada_name": ["Köln Hbf", "Berlin Hbf"], "line": ["RE1", "ICE10"], "hour": [1, 2]})
    # strict=True: zip raises if the two lists differ in length instead of silently cutting one
    train_codes = dict(zip(train["stada_name"], caster.transform(train)["stada_name"].cat.codes, strict=True))
    valid_codes = dict(zip(valid["stada_name"], caster.transform(valid)["stada_name"].cat.codes, strict=True))
    assert train_codes == valid_codes


def test_unknown_station_becomes_missing_instead_of_crashing(caster):
    new = pd.DataFrame({"stada_name": ["Hamburg Hbf"], "line": ["ICE10"], "hour": [7]})
    out = caster.transform(new)
    assert out["stada_name"].isna().all()
    assert out["line"].iloc[0] == "ICE10"


def test_input_that_is_already_categorical_is_renumbered(caster):
    # load_months() hands over pandas categories with their OWN numbering (here Köln = 0).
    # The caster must switch to the numbering learned in fit() (Köln = 1).
    valid = pd.DataFrame(
        {
            "stada_name": pd.Categorical(["Köln Hbf"], categories=["Köln Hbf", "Berlin Hbf"]),
            "line": pd.Categorical(["RE1"]),
            "hour": [7],
        }
    )
    out = caster.transform(valid)
    assert list(out["stada_name"].cat.categories) == ["Berlin Hbf", "Köln Hbf"]
    assert out["stada_name"].cat.codes.iloc[0] == 1


def test_transform_does_not_change_the_callers_table(caster, train):
    before = train.copy()
    caster.transform(train)
    pd.testing.assert_frame_equal(train, before)


def test_other_columns_stay_untouched(caster, train):
    assert caster.transform(train)["hour"].tolist() == [7, 8, 9]
