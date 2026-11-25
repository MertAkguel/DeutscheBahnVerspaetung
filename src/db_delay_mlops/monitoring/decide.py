"""The retraining decision: plain rules, no MLflow, so they are easy to read and to test.

Retrain only for a reason that is about the MODEL:
- its ROC-AUC on the new month dropped clearly below the value at registration, or
- it is too old (safety net, even if nothing looks wrong).
Drift, a changed late share or a calibration gap only WARN: they are hints for a human.
The model has seen a whole year, so a winter month differs from the average without the
model being worse.

ROC-AUC, not PR-AUC, for the comparison across months: PR-AUC depends on the late share of
the month (a month with more delays gets a higher PR-AUC for the same model), ROC-AUC does not.
"""

from dataclasses import dataclass, field

import polars as pl

ROC_AUC_DROP = 0.02  # retrain if the champion lost more than this
MAX_CHAMPION_AGE_MONTHS = 3  # retrain at least every 3 months
LATE_SHARE_CHANGE = 0.05  # warn if the late share moved by more than 5 percentage points
CALIBRATION_GAP = 0.03  # warn if the mean prediction is off by more than 3 percentage points


@dataclass
class Decision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)  # why retrain (empty = no reason)
    warnings: list[str] = field(default_factory=list)  # for a human to look at


def months_between(earlier: str, later: str) -> int:
    """months_between("2026-10", "2027-01") -> 3."""
    (y1, m1), (y2, m2) = (map(int, earlier.split("-")), map(int, later.split("-")))
    return (y2 * 12 + m2) - (y1 * 12 + m1)


def decide(month: str, performance: dict, drift: pl.DataFrame) -> Decision:
    reasons, warnings = [], []

    champion_month = performance["champion_month"]
    if champion_month is None:
        reasons.append("the champion has no validation month, its age is unknown")
    elif months_between(champion_month, month) >= MAX_CHAMPION_AGE_MONTHS:
        reasons.append(f"the champion is {months_between(champion_month, month)} months old ({champion_month})")

    baseline = performance["baseline_roc_auc"]
    if baseline is not None and baseline - performance["roc_auc"] > ROC_AUC_DROP:
        reasons.append(f"ROC-AUC dropped from {baseline:.4f} to {performance['roc_auc']:.4f}")

    strong = drift.filter(pl.col("level") == "strong")["feature"].to_list()
    if strong:
        warnings.append(f"strong drift in {strong}")

    baseline_late = performance["baseline_late_share"]
    if baseline_late is not None and abs(performance["late_share"] - baseline_late) > LATE_SHARE_CHANGE:
        warnings.append(f"late share moved from {baseline_late:.1%} to {performance['late_share']:.1%}")

    gap = performance["mean_prediction"] - performance["late_share"]
    if abs(gap) > CALIBRATION_GAP:
        direction = "too high" if gap > 0 else "too low"
        warnings.append(f"predictions are {abs(gap):.1%} {direction} on average (calibration)")

    return Decision(retrain=bool(reasons), reasons=reasons, warnings=warnings)
