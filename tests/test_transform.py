"""Checks on the dbt-built warehouse, recomputed independently in pandas."""

import numpy as np
import pandas as pd

from market_pipeline.warehouse import connect


def _table(built, sql: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    with connect(built.settings) as con:
        return con.sql(sql).df()


def test_features_match_independent_pandas_computation(built) -> None:  # type: ignore[no-untyped-def]
    prices = _table(built, "select * from staging.stg_prices where ticker = 'BBB' order by date")
    feats = _table(
        built, "select * from marts.features_volatility where ticker = 'BBB' order by date"
    ).set_index("date")

    r = np.log(prices["adj_close"] / prices["adj_close"].shift(1))
    r.index = prices["date"]
    r = r.dropna()
    rv_21 = np.sqrt(252 * (r**2).rolling(21).mean())
    fwd_5 = np.sqrt(252 * (r**2).rolling(5).mean()).shift(-5)  # mean of r_{t+1..t+5}

    sample = feats.index[100:110]
    assert np.allclose(feats.loc[sample, "rv_21d"], rv_21.loc[sample])
    assert np.allclose(feats.loc[sample, "target_rv_fwd"], fwd_5.loc[sample])


def test_latest_rows_are_unlabeled_scoring_set(built) -> None:  # type: ignore[no-untyped-def]
    tail = _table(
        built,
        """select ticker, count(*) filter (where not is_labeled) as unlabeled
           from marts.features_volatility group by 1""",
    )
    assert (tail["unlabeled"] == 5).all()  # horizon_days rows per ticker await their future


def test_monthly_aggregate_covers_all_tickers(built) -> None:  # type: ignore[no-untyped-def]
    agg = _table(built, "select * from marts.agg_ticker_monthly")
    assert set(agg["ticker"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert agg["pct_up_days"].between(0, 1).all()
