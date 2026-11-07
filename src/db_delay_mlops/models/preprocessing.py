"""Preprocessing steps that are saved together with the model."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Turn text columns into pandas categories, with the categories learned in fit().

    Tree models (LightGBM, XGBoost) handle categories natively, but every table must use
    the same categories as the training data: internally "Köln Hbf" becomes a number,
    and a different numbering would silently mix up stations. Values not seen in fit()
    become missing (NaN), which the trees can handle.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        # learned on train only: the sorted list of known values per column
        self.categories_ = {col: sorted(X[col].dropna().unique()) for col in self.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # never change the caller's DataFrame
        for col, categories in self.categories_.items():
            X[col] = pd.Categorical(X[col], categories=categories)
        return X
    
    
    
class FillMissingText(BaseEstimator, TransformerMixin):
    """Replace missing values in text columns with "missing" (CatBoost cannot handle None in categories).

    "missing" becomes a category of its own, e.g. "long-distance train without line number".
    Nothing is learned from the data, so fit() only checks the columns.
    """

    def __init__(self, columns: list[str], fill_value: str = "missing"):
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"columns not found: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.fill_value).astype(str)
        return X