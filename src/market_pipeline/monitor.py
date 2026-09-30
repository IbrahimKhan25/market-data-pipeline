"""Monitoring: feature drift against the training reference, plus live accuracy of past forecasts.

Volatility-*level* features (rv_*, parkinson_*, market_rv) move between regimes all the
time; tracking those regimes is what the model is for. PSI on them is reported for
information, but it does not affect health. Health depends on:

* drift in the scale-free features (returns, gaps, volume z-scores, relative vol), which
  should be stationary, so drift there usually points to a data problem; and
* live accuracy: QLIKE of matured forecasts compared with the cross-validated QLIKE.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from market_pipeline.config import Settings
from market_pipeline.ml.drift import drift_report
from market_pipeline.ml.evaluation import regression_metrics
from market_pipeline.ml.models import VOL_FEATURES, log_target
from market_pipeline.ml.predict import FORECAST_TABLE
from market_pipeline.ml.train import load_model
from market_pipeline.warehouse import connect

log = logging.getLogger(__name__)

RECENT_SESSIONS = 63
DEGRADATION_TOLERANCE = 1.5  # alert if live QLIKE > 1.5x the cross-validated QLIKE


def _recent_features(settings: Settings, sessions: int) -> pd.DataFrame:
    with connect(settings) as con:
        return con.execute(
            """
            select * from marts.features_volatility
            where date in (
                select distinct date from marts.features_volatility order by date desc limit ?
            )
            """,
            [sessions],
        ).df()


def _matured_forecasts(settings: Settings) -> pd.DataFrame:
    """Past forecasts whose realised target is now observable."""
    with connect(settings) as con:
        exists = con.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = 'marts' and table_name = 'volatility_forecasts'"
        ).fetchone()
        if not exists or exists[0] == 0:
            return pd.DataFrame()
        return con.sql(
            f"""
            select f.ticker, f.as_of_date, f.forecast_vol, f.model_version, v.target_rv_fwd
            from {FORECAST_TABLE} f
            join marts.features_volatility v on v.ticker = f.ticker and v.date = f.as_of_date
            where v.target_rv_fwd is not null
            """
        ).df()


def run_monitor(settings: Settings) -> dict[str, Any]:
    _, card = load_model(settings)
    recent = _recent_features(settings, RECENT_SESSIONS)
    drift = drift_report(card["reference_profile"], recent)
    drift["gates_health"] = ~drift["feature"].isin(VOL_FEATURES)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": card["version"],
        "window_sessions": RECENT_SESSIONS,
        "drift": drift.to_dict(orient="records"),
        "drifted_features": drift.loc[
            (drift["status"] == "alert") & drift["gates_health"], "feature"
        ].tolist(),
        "regime_shift_features": drift.loc[
            (drift["status"] == "alert") & ~drift["gates_health"], "feature"
        ].tolist(),
        "live": None,
    }

    matured = _matured_forecasts(settings)
    if not matured.empty:
        m = regression_metrics(
            log_target(matured["target_rv_fwd"]), np.log(matured["forecast_vol"].to_numpy())
        )
        cv_qlike = card["cv"][card["champion"]]["qlike_mean"]
        report["live"] = {
            "n_forecasts": len(matured),
            **{k: round(v, 5) for k, v in m.items()},
            "cv_qlike": cv_qlike,
            "degraded": m["qlike"] > DEGRADATION_TOLERANCE * cv_qlike,
        }

    report["healthy"] = not report["drifted_features"] and not (
        report["live"] and report["live"]["degraded"]
    )

    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.paths.reports_dir / f"monitoring_{datetime.now(UTC):%Y%m%d}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("monitoring report -> %s (healthy=%s)", path, report["healthy"])
    return report
