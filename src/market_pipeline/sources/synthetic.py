"""Deterministic GARCH(1,1) price simulator.

Used for offline development, CI and tests: it needs no network, and because returns
follow a GARCH process, volatility is genuinely forecastable — so the ML stage has a
real signal to find. Each ticker's path is simulated from a fixed origin, so any
``[start, end)`` window returns the same bars no matter how the range is split, which
matters when testing incremental ingestion.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from market_pipeline.sources.base import PRICE_COLUMNS, empty_prices

ORIGIN = date(2010, 1, 1)
HORIZON = date(2040, 1, 1)  # simulate a fixed span so every window sees the same path


class SyntheticSource:
    name = "synthetic"

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def _simulate(self, ticker: str) -> pd.DataFrame:
        return _simulate_path(ticker, self.seed)

    def fetch(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if end <= start:
            return empty_prices()
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        frames = []
        for ticker in tickers:
            df = self._simulate(ticker)
            frames.append(df[(df["date"] >= lo) & (df["date"] < hi)])
        return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]


@lru_cache(maxsize=256)
def _simulate_path(ticker: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + zlib.crc32(ticker.encode()))
    days = pd.bdate_range(ORIGIN, HORIZON, inclusive="left")
    n = len(days)

    # Per-ticker GARCH parameters so tickers differ in level and persistence.
    long_run_vol = rng.uniform(0.15, 0.55) / np.sqrt(252)
    # alpha + beta <= 0.97 keeps the process covariance-stationary (variance can't explode).
    alpha, beta = rng.uniform(0.05, 0.12), rng.uniform(0.80, 0.85)
    omega = long_run_vol**2 * (1 - alpha - beta)

    z = rng.standard_t(df=6, size=n) / np.sqrt(6 / 4)
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = long_run_vol**2
    for t in range(n):
        if t > 0:
            var[t] = omega + alpha * ret[t - 1] ** 2 + beta * var[t - 1]
        ret[t] = 0.0002 + np.sqrt(var[t]) * z[t]

    close = rng.uniform(20, 400) * np.exp(np.cumsum(ret))
    prev_close = np.concatenate([[close[0]], close[:-1]])
    sigma = np.sqrt(var)
    open_ = prev_close * np.exp(rng.normal(0, 0.3, n) * sigma)
    wick_hi = np.abs(rng.normal(0, 0.5, n)) * sigma
    wick_lo = np.abs(rng.normal(0, 0.5, n)) * sigma
    high = np.maximum(open_, close) * np.exp(wick_hi)
    low = np.minimum(open_, close) * np.exp(-wick_lo)
    volume = np.round(rng.lognormal(15, 0.3, n) * (1 + 20 * np.abs(ret - 0.0002)))

    return pd.DataFrame(
        {
            "date": days.astype("datetime64[ns]"),
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )
