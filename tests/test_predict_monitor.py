import pandas as pd

from market_pipeline.ml.predict import FORECAST_TABLE, run_predict, score
from market_pipeline.ml.train import load_model
from market_pipeline.monitor import run_monitor
from market_pipeline.warehouse import connect


def test_predict_is_idempotent_upsert(built) -> None:  # type: ignore[no-untyped-def]
    first = run_predict(built.settings)
    run_predict(built.settings)
    assert len(first) == 4
    assert (first["forecast_vol"] > 0).all() and first["forecast_vol"].lt(5).all()
    with connect(built.settings) as con:
        (n,) = con.sql(f"select count(*) from {FORECAST_TABLE}").fetchone()  # type: ignore[misc]
    assert n == 4


def test_stale_ticker_is_flagged(built) -> None:  # type: ignore[no-untyped-def]
    pipeline, card = load_model(built.settings)
    with connect(built.settings) as con:
        feats = con.sql(
            """select * from marts.features_volatility
               qualify row_number() over (partition by ticker order by date desc) = 1"""
        ).df()
    feats.loc[0, "date"] = pd.Timestamp(feats["date"].max()) - pd.Timedelta(days=10)
    out = score(feats, pipeline, card)
    assert out["is_stale"].tolist() == [True, False, False, False]


def test_monitor_measures_live_accuracy_of_matured_forecasts(built) -> None:  # type: ignore[no-untyped-def]
    pipeline, card = load_model(built.settings)
    with connect(built.settings) as con:
        past = con.sql(
            """select * from marts.features_volatility
               where is_labeled
                 and date >= (select max(date) - interval 60 day from marts.features_volatility)"""
        ).df()
    backdated = score(past, pipeline, card)
    with connect(built.settings, read_only=False) as con:
        con.register("bd", backdated)
        con.execute(f"insert or replace into {FORECAST_TABLE} select * from bd")

    report = run_monitor(built.settings)
    live = report["live"]
    assert live["n_forecasts"] == len(backdated)
    assert live["qlike"] < 1.5 * card["cv"][card["champion"]]["qlike_mean"]
    assert not live["degraded"]
    assert report["drifted_features"] == []  # synthetic data is stationary
    assert (built.settings.paths.reports_dir).glob("monitoring_*.json")
