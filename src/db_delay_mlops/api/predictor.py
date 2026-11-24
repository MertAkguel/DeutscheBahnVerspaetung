"""The model behind the API: load the champion once, then answer requests.

Kept apart from the web code (api/main.py), so it can be tested without HTTP and
the web code stays thin: it only translates between HTTP and this class.
"""

import os
from dataclasses import dataclass, field
from difflib import get_close_matches

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from db_delay_mlops.api.schemas import ArrivalRequest, ModelInfo, Prediction, Reason, risk_level, to_frame
from db_delay_mlops.features.columns import FEATURES
from db_delay_mlops.features.serving import features_for
from db_delay_mlops.models.calibration import CalibratedPipeline
from db_delay_mlops.models.register import MODEL_NAME

N_REASONS = 3

# Text features the model may not know, with the name the API caller used for them
CHECKED_FEATURES = {"line": "line_number", "final_destination_station": "final_destination"}


@dataclass
class Predictor:
    # CalibratedPipeline from models/train.py, or a plain Pipeline (models trained before calibration, e.g. v2)
    model: CalibratedPipeline | Pipeline
    info: ModelInfo
    known: dict[str, set[str]] = field(init=False)  # filled below, not passed in

    def __post_init__(self):
        # dataclass calls this right after __init__: remember the categories the model learned in training
        caster = self.pipeline.named_steps["categories"]
        self.known = {col: set(caster.categories_[col]) for col in CHECKED_FEATURES}

    @property
    def pipeline(self) -> Pipeline:
        """The CategoryCaster + LightGBM inside, with or without calibration around it."""
        return self.model.pipeline if isinstance(self.model, CalibratedPipeline) else self.model

    def predict(self, arrivals: list[ArrivalRequest]) -> list[Prediction]:
        """One prediction per arrival, in the same order."""
        X = features_for(to_frame(arrivals)).to_pandas()
        probabilities = self.model.predict_proba(X)[:, 1]
        contributions = self.contributions(X)
        warnings = self.unknown_value_warnings(X)
        return [
            Prediction(probability_late=round(float(p), 4), risk=risk_level(p), top_reasons=top_reasons(c), warnings=w)
            for p, c, w in zip(probabilities, contributions, warnings, strict=True)
        ]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per row and feature: how much it pushed the prediction, in log-odds (the idea behind SHAP).

        LightGBM computes this itself with pred_contrib=True. It needs the prepared table,
        so the CategoryCaster (all steps but the last, pipeline[:-1]) runs first.
        The sigmoid calibration multiplies all log-odds by its slope, so the contributions too.
        """
        prepared = self.pipeline[:-1].transform(X)
        raw = self.pipeline[-1].predict(prepared, pred_contrib=True)
        slope = self.model.slope_ if isinstance(self.model, CalibratedPipeline) else 1.0
        return slope * raw[:, :-1]  # the last column is the base value (average prediction), not a feature

    def unknown_value_warnings(self, X: pd.DataFrame) -> list[list[str]]:
        """Per row: values the model never saw in training. They are treated as missing, so say so."""
        warnings = [[] for _ in range(len(X))]
        for col, api_name in CHECKED_FEATURES.items():
            for i, value in enumerate(X[col]):
                if value is not None and not pd.isna(value) and value not in self.known[col]:
                    suggestions = get_close_matches(value, sorted(self.known[col]), n=3)
                    warnings[i].append(
                        f"{api_name} {value!r} was not seen in training and is ignored; did you mean {suggestions}?"
                    )
        return warnings


def top_reasons(contribution: np.ndarray, n: int = N_REASONS) -> list[Reason]:
    """The n features with the largest effect, up or down, strongest first."""
    strongest = np.argsort(-np.abs(contribution))[:n]
    return [Reason(feature=FEATURES[i], contribution=round(float(contribution[i]), 3)) for i in strongest]


def load_champion() -> Predictor:
    """Load the model with the alias @champion from MLflow (once, when the API starts)."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    # Load by version number, not by alias: if the alias moves while we load, version and model still match
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
    pr_auc = version.tags.get("valid_pr_auc")
    info = ModelInfo(
        model_name=MODEL_NAME,
        model_version=str(version.version),  # MLflow gives a number or a text, depending on the store
        valid_month=version.tags.get("valid_month"),
        valid_pr_auc=float(pr_auc) if pr_auc else None,
    )
    return Predictor(model=model, info=info)
