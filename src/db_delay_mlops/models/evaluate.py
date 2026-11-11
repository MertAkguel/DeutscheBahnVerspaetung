"""Metrics used to compare all models."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """All metrics we compare models on. p = predicted probability of being late."""
    return {
        "pr_auc": average_precision_score(y_true, p),
        "roc_auc": roc_auc_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "log_loss": log_loss(y_true, p),
    }
