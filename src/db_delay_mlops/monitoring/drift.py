"""Data drift: how different is a new month from the year before, feature by feature?

Measured with the PSI (Population Stability Index). Each feature is cut into bins, and the
share of rows per bin is compared between the reference (the 12 months before) and the
new month:

    PSI = sum over bins of (new share - old share) * ln(new share / old share)

Identical distributions give 0; the more rows move to other bins, the larger it gets.
Common rule of thumb: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 strong drift.

Bins: numbers get 10 bins at the reference quantiles (each bin holds 10 % of the reference),
text and codes one bin per value (the 20 most common, the rest as "other").
Drift is only a hint, not a reason to retrain: one winter month always differs from a
whole year, and the model has seen last winter. See monitoring/decide.py.
"""

import numpy as np
import polars as pl

from db_delay_mlops.features.build import OUT_DIR
from db_delay_mlops.features.columns import CODE_FEATURES, NUMERIC_FEATURES, TARGET, TEXT_FEATURES
from db_delay_mlops.models.train import previous_months

# "month" is left out: one month against twelve always "drifts", by construction
NUMERIC = [c for c in NUMERIC_FEATURES if c != "is_holiday"]
CATEGORICAL = TEXT_FEATURES + [c for c in CODE_FEATURES if c != "month"] + ["is_holiday"]

N_BINS = 10
TOP_VALUES = 20
EPS = 1e-4  # an empty bin would give ln(0); treat it as 0.01 %
MODERATE_PSI = 0.10
STRONG_PSI = 0.25
REFERENCE_ROWS_PER_MONTH = 50_000  # 12 x 50,000 rows: enough for stable shares, quick to compute


def psi(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    """PSI of two share vectors over the same bins (each sums to 1)."""
    r = np.maximum(reference_share, EPS)
    c = np.maximum(current_share, EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_psi(reference: pl.Series, current: pl.Series) -> float:
    """Bins at the reference deciles; missing values get a bin of their own."""
    ref = reference.drop_nulls().to_numpy()
    cur = current.drop_nulls().to_numpy()
    # np.unique: many equal values (e.g. precipitation 0.0) would give the same edge several times
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, N_BINS + 1)[1:-1]))

    def shares(values: np.ndarray, series: pl.Series) -> np.ndarray:
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        return np.append(counts, series.null_count()) / len(series)

    return psi(shares(ref, reference), shares(cur, current))


def categorical_psi(reference: pl.Series, current: pl.Series) -> float:
    """One bin per common value, "other" for the rest, "missing" for nulls."""
    top = reference.drop_nulls().value_counts(sort=True).head(TOP_VALUES)[reference.name].cast(pl.String).to_list()

    def shares(series: pl.Series) -> pl.DataFrame:
        binned = (
            series.cast(pl.String)
            .fill_null("<missing>")
            .replace_strict({v: v for v in top} | {"<missing>": "<missing>"}, default="<other>")
        )
        return binned.value_counts(normalize=True, name="share").rename({binned.name: "bin"})

    both = shares(reference).join(shares(current), on="bin", how="full", coalesce=True, suffix="_current")
    both = both.fill_null(0.0)
    return psi(both["share"].to_numpy(), both["share_current"].to_numpy())


def drift_table(reference: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    """PSI and drift level per feature, strongest first."""
    rows = [{"feature": c, "psi": numeric_psi(reference[c], current[c])} for c in NUMERIC]
    rows += [{"feature": c, "psi": categorical_psi(reference[c], current[c])} for c in CATEGORICAL]
    return (
        pl.DataFrame(rows)
        .with_columns(
            level=pl.when(pl.col("psi") > STRONG_PSI)
            .then(pl.lit("strong"))
            .when(pl.col("psi") > MODERATE_PSI)
            .then(pl.lit("moderate"))
            .otherwise(pl.lit("stable"))
        )
        .sort("psi", descending=True)
    )


def load_reference(month: str, n_months: int = 12, data_dir=OUT_DIR) -> pl.DataFrame:
    """A fixed-seed sample of the 12 months before `month`: the year the model has learned from."""
    frames = []
    for m in previous_months(month, n_months):
        table = pl.read_parquet(data_dir / f"arrivals-{m}.parquet", columns=[*NUMERIC, *CATEGORICAL, TARGET])
        frames.append(table.sample(min(REFERENCE_ROWS_PER_MONTH, table.height), seed=0))
    return pl.concat(frames)


def load_current(month: str, data_dir=OUT_DIR) -> pl.DataFrame:
    return pl.read_parquet(data_dir / f"arrivals-{month}.parquet", columns=[*NUMERIC, *CATEGORICAL, TARGET])
