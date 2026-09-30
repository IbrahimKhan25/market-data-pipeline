from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class MarketDataSource(Protocol):
    """A provider of daily OHLCV bars.

    ``fetch`` returns a tidy (long) frame with exactly ``PRICE_COLUMNS``, one row per
    (date, ticker), for trading days in ``[start, end)``.
    """

    name: str

    def fetch(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame: ...


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}).astype(
        {"date": "datetime64[ns]", "ticker": "object"}
    )
