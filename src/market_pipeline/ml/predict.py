"""Batch scoring: forecast next-week volatility for every ticker's latest session."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from market_pipeline.config import Settings
from market_pipeline.ml.train import load_model, predict_vol
from market_pipeline.warehouse import connect

log = logging.getLogger(__name__)

# A ticker whose last bar lags the freshest ticker by more than this is flagged stale.
STALE_AFTER_DAYS = 4

FORECAST_TABLE = "marts.volatility_forecasts"
_DDL = f"""
create table if not exists {FORECAST_TABLE} (
    ticker          varchar,
    as_of_date      date,
    horizon_days    integer,
    forecast_vol    double,
    current_rv_21d  double,
    is_stale        boolean,
    model_version   varchar,
    model_name      varchar,
    scored_at       timestamp,
    primary key (ticker, as_of_date, model_version)
)
"""


def latest_features(settings: Settings, ticker: str | None = None) -> pd.DataFrame:
    params: list[Any] = []
    where = ""
    if ticker:
        where, params = "where ticker = ?", [ticker.upper()]
    sql = f"""
        select * from marts.features_volatility {where}
        qualify row_number() over (partition by ticker order by date desc) = 1
        order by ticker
    """
    with connect(settings) as con:
        return con.execute(sql, params).df()


def score(
    features: pd.DataFrame,
    pipeline: Any,
    card: dict[str, Any],
    stale_after_days: int = STALE_AFTER_DAYS,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(features["date"])
    return pd.DataFrame(
        {
            "ticker": features["ticker"],
            "as_of_date": dates.dt.date,
            "horizon_days": card["horizon_days"],
            "forecast_vol": predict_vol(pipeline, features),
            "current_rv_21d": features["rv_21d"],
            "is_stale": (dates.max() - dates).dt.days > stale_after_days,
            "model_version": card["version"],
            "model_name": card["champion"],
            "scored_at": datetime.now(UTC).replace(tzinfo=None),
        }
    ).reset_index(drop=True)


def run_predict(settings: Settings) -> pd.DataFrame:
    pipeline, card = load_model(settings)
    forecasts = score(latest_features(settings), pipeline, card)
    if forecasts.empty:
        log.warning("no feature rows to score")
        return forecasts

    # Idempotent upsert: re-scoring the same day with the same model replaces its rows.
    with connect(settings, read_only=False) as con:
        con.execute(_DDL)
        con.register("new_forecasts", forecasts)
        con.execute(f"insert or replace into {FORECAST_TABLE} select * from new_forecasts")

    out_dir = settings.paths.gold_dir / "forecasts" / f"as_of_date={forecasts['as_of_date'].max()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / f"forecasts-{card['version']}.parquet", index=False)
    stale = forecasts.loc[forecasts["is_stale"], "ticker"].tolist()
    if stale:
        log.warning("stale inputs for: %s", stale)
    log.info("scored %d tickers with %s/%s", len(forecasts), card["champion"], card["version"])
    return forecasts
