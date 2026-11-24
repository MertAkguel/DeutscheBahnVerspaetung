"""Tests for models/calibration.py: probabilities that are too high must come down, the order must stay."""

import numpy as np
import pandas as pd
import pytest

from db_delay_mlops.models.calibration import CalibratedPipeline, logit, sigmoid


class OverconfidentModel:
    """Stands in for the trained pipeline: knows the true probability, but reports it too high."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        too_high = sigmoid(logit(X["p_true"].to_numpy()) + 0.5)  # +0.5 log-odds: 22 % becomes about 32 %
        return np.column_stack([1 - too_high, too_high])


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    p_true = rng.beta(2, 7, 50_000)  # mostly low probabilities, like our trains (mean about 22 %)
    y = (rng.random(50_000) < p_true).astype(int)
    return pd.DataFrame({"p_true": p_true}), y


def test_logit_and_sigmoid_undo_each_other():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert np.allclose(sigmoid(logit(p)), p)


def test_calibration_brings_the_mean_to_the_real_late_share(data):
    X, y = data
    raw = OverconfidentModel().predict_proba(X)[:, 1]
    calibrated = CalibratedPipeline(OverconfidentModel()).fit(X, y).predict_proba(X)[:, 1]
    assert raw.mean() - y.mean() > 0.05  # before: clearly too high
    assert abs(calibrated.mean() - y.mean()) < 0.005  # after: matches reality


def test_calibration_learns_the_shift(data):
    X, y = data
    model = CalibratedPipeline(OverconfidentModel()).fit(X, y)
    assert model.slope_ == pytest.approx(1.0, abs=0.1)
    assert model.intercept_ == pytest.approx(-0.5, abs=0.1)  # undoes the +0.5


def test_calibration_never_changes_the_order(data):
    # same order -> same PR-AUC and ROC-AUC: the quality gate is not affected
    X, y = data
    raw = OverconfidentModel().predict_proba(X)[:, 1]
    calibrated = CalibratedPipeline(OverconfidentModel()).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(np.argsort(raw), np.argsort(calibrated))
