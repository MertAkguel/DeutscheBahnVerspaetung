"""FastAPI app: delay predictions over HTTP.

Start locally (MLflow must be running):
    uv run uvicorn db_delay_mlops.api.main:app --reload --port 8000
Then open http://localhost:8000/docs to try every endpoint in the browser.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request

from db_delay_mlops.api.predictor import Predictor, load_champion
from db_delay_mlops.api.schemas import (
    ArrivalRequest,
    BatchRequest,
    BatchResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
    known_stations,
)


def create_app(load_predictor: Callable[[], Predictor] = load_champion) -> FastAPI:
    """Build the app. Tests pass their own load_predictor, so they need neither MLflow nor the real model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs ONCE at start-up: loading the model takes seconds, a prediction milliseconds
        app.state.predictor = load_predictor()
        yield  # the API serves requests here; code after yield would run at shutdown

    app = FastAPI(
        title="DB delay prediction",
        description="Probability that a Deutsche Bahn train arrives at a station 6 or more minutes late.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> Predictor:
        return request.app.state.predictor

    # "Give this endpoint the predictor": FastAPI calls get_predictor for every request (dependency injection)
    PredictorDep = Annotated[Predictor, Depends(get_predictor)]

    def run(predictor: Predictor, arrivals: list[ArrivalRequest]) -> list[Prediction]:
        try:
            return predictor.predict(arrivals)
        except httpx.HTTPError as error:  # Open-Meteo down or slow: our fault would be 500, theirs is 503
            raise HTTPException(status_code=503, detail="weather forecast unavailable, try again later") from error

    @app.get("/health")
    def health() -> dict[str, str]:
        """Is the API up? Docker and monitoring call this."""
        return {"status": "ok"}

    @app.get("/model")
    def model_info(predictor: PredictorDep) -> ModelInfo:
        """Which model answers: version, validation month and PR-AUC."""
        return predictor.info

    @app.get("/stations")
    def stations() -> list[str]:
        """All station names the model knows; use them exactly as written."""
        return known_stations()

    @app.post("/predict")
    def predict(arrival: ArrivalRequest, predictor: PredictorDep) -> PredictionResponse:
        """Delay risk for one arrival."""
        [prediction] = run(predictor, [arrival])
        return PredictionResponse(**prediction.model_dump(), model_version=predictor.info.model_version)

    @app.post("/predict/batch")
    def predict_batch(batch: BatchRequest, predictor: PredictorDep) -> BatchResponse:
        """Delay risk for many arrivals at once (answers in the same order)."""
        return BatchResponse(model_version=predictor.info.model_version, predictions=run(predictor, batch.arrivals))

    return app


app = create_app()  # what uvicorn starts: "db_delay_mlops.api.main:app"
