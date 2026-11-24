"""Calibration: make the model's probabilities match how often trains are really late.

The trees rank trains well (PR-AUC), but their probabilities were too high: of the arrivals
with "45 %", only 39 % were late (October 2026). A calibration step learns a correction on
data the trees did not train on (the stop month) and applies it to every prediction.

Sigmoid ("Platt scaling"), not isotonic: it bends the probabilities with two numbers,
    calibrated = sigmoid(slope * logit(p) + intercept)
which never changes the ORDER of the trains. So PR-AUC and ROC-AUC stay exactly the same
(the quality gate is not affected), only the height of the probabilities is corrected.
Isotonic regression fixed the height as well, but merged close values into steps and
lowered PR-AUC a little (0.5373 -> 0.5339 on October).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EPS = 1e-6  # keeps logit finite for probabilities of exactly 0 or 1


def logit(p: np.ndarray) -> np.ndarray:
    """Probability -> log-odds: 0.5 -> 0, 0.9 -> 2.2, 0.1 -> -2.2 (the scale LightGBM adds trees on)."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Log-odds -> probability, the inverse of logit."""
    return 1 / (1 + np.exp(-z))


class CalibratedPipeline(ClassifierMixin, BaseEstimator):
    """A trained pipeline plus the two calibration numbers. Behaves like a normal classifier."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @property
    def classes_(self):
        return self.pipeline.classes_

    def fit(self, X, y):
        """Learn ONLY slope and intercept; the pipeline is already trained and stays as it is.

        Use data the trees were not fitted on (the stop month), else the correction learns
        the trees' over-confidence on their own training data.
        """
        z = logit(self.pipeline.predict_proba(X)[:, 1]).reshape(-1, 1)
        fitted = LogisticRegression(C=1e6).fit(z, y)  # large C: practically no regularisation
        self.slope_ = float(fitted.coef_[0][0])
        self.intercept_ = float(fitted.intercept_[0])
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        p = sigmoid(self.slope_ * logit(raw) + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
