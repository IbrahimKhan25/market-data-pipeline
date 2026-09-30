"""Online serving for the promoted volatility model."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, cast

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from market_pipeline import __version__
from market_pipeline.config import Settings, get_settings
from market_pipeline.ml.models import ALL_FEATURES
from market_pipeline.ml.predict import latest_features, score
from market_pipeline.ml.train import load_model, predict_vol


class Forecast(BaseModel):
    ticker: str
    as_of_date: date
    horizon_days: int
    forecast_vol: float = Field(
        description="Annualised volatility forecast for the next horizon_days sessions"
    )
    current_rv_21d: float
    is_stale: bool
    model_version: str
    model_name: str


class FeatureVector(BaseModel):
    """Raw feature values for one (ticker, date), as in marts.features_volatility."""

    rv_5d: float = Field(gt=0)
    rv_21d: float = Field(gt=0)
    rv_63d: float = Field(gt=0)
    parkinson_5d: float = Field(ge=0)
    parkinson_21d: float = Field(ge=0)
    market_rv_21d: float = Field(gt=0)
    relative_rv_21d: float = Field(gt=0)
    abs_return_1d: float = Field(ge=0)
    log_return: float
    overnight_gap: float
    momentum_21d: float
    volume_z_21d: float
    day_of_week: int = Field(ge=1, le=7)


class PredictResponse(BaseModel):
    forecast_vol: float
    horizon_days: int
    model_version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings or get_settings()
        try:
            app.state.pipeline, app.state.card = load_model(app.state.settings)
        except FileNotFoundError:
            app.state.pipeline, app.state.card = None, None
        yield

    app = FastAPI(
        title="Market Volatility Forecaster",
        version=__version__,
        description="Serves next-week realised-volatility forecasts from the promoted model.",
        lifespan=lifespan,
    )

    def _model(request: Request) -> tuple[Any, dict[str, Any]]:
        if request.app.state.pipeline is None:
            raise HTTPException(503, "no promoted model; run `market-pipeline train`")
        return request.app.state.pipeline, request.app.state.card

    def _features(request: Request, ticker: str | None = None) -> pd.DataFrame:
        try:
            return latest_features(request.app.state.settings, ticker)
        except duckdb.Error as exc:  # warehouse missing or locked by a running transform
            raise HTTPException(503, f"warehouse unavailable: {exc}") from exc

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        card = request.app.state.card
        return {
            "status": "ok",
            "model_loaded": card is not None,
            "model_version": card["version"] if card else None,
        }

    @app.get("/model")
    def model_card(request: Request) -> dict[str, Any]:
        _, card = _model(request)
        return {k: v for k, v in card.items() if k != "reference_profile"}

    @app.get("/forecasts", response_model=list[Forecast])
    def forecasts(request: Request) -> list[dict[str, Any]]:
        pipeline, card = _model(request)
        records = score(_features(request), pipeline, card).to_dict(orient="records")
        return cast(list[dict[str, Any]], records)

    @app.get("/forecasts/{ticker}", response_model=Forecast)
    def forecast_ticker(ticker: str, request: Request) -> dict[str, Any]:
        pipeline, card = _model(request)
        feats = _features(request, ticker)
        if feats.empty:
            raise HTTPException(404, f"unknown ticker {ticker.upper()!r}")
        return cast(dict[str, Any], score(feats, pipeline, card).iloc[0].to_dict())

    @app.post("/predict", response_model=PredictResponse)
    def predict(features: FeatureVector, request: Request) -> PredictResponse:
        pipeline, card = _model(request)
        frame = pd.DataFrame([features.model_dump()])[ALL_FEATURES]
        return PredictResponse(
            forecast_vol=float(predict_vol(pipeline, frame)[0]),
            horizon_days=card["horizon_days"],
            model_version=card["version"],
        )

    return app


app = create_app()
