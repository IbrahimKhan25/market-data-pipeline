"""Yahoo Finance source via ``yfinance``, with batching and retries."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from market_pipeline.sources.base import PRICE_COLUMNS, empty_prices

log = logging.getLogger(__name__)

_RENAME = {
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class EmptyDownloadError(RuntimeError):
    """yfinance returned nothing — usually rate limiting or a transient outage."""


def normalise_download(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn yfinance's wide (Ticker, Price) MultiIndex frame into tidy rows.

    The original version of this project wrote the wide frame straight to CSV, so every
    ticker after the first landed in suffixed columns (``close.1``...) and was silently
    dropped downstream. Stacking the ticker level fixes that.
    """
    if raw.empty:
        return empty_prices()
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("expected (Ticker, Price) MultiIndex columns from yfinance")

    tidy = raw.stack(level="Ticker", future_stack=True).reset_index()
    tidy = tidy.rename(columns=_RENAME)
    if "adj_close" not in tidy.columns:
        tidy["adj_close"] = tidy["close"]
    tidy = tidy.dropna(subset=["open", "high", "low", "close"], how="all")
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    return tidy[PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


class YahooSource:
    name = "yahoo"

    def __init__(self, batch_size: int = 20) -> None:
        self.batch_size = batch_size

    @retry(
        retry=retry_if_exception_type(EmptyDownloadError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _download(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            multi_level_index=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise EmptyDownloadError(f"no data for {tickers} in [{start}, {end})")
        return raw

    def fetch(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        tickers = list(tickers)
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            try:
                raw = self._download(batch, start, end)
            except EmptyDownloadError:
                log.warning("yahoo returned no rows for %s in [%s, %s)", batch, start, end)
                continue
            frames.append(normalise_download(raw))

        if not frames:
            return empty_prices()
        out = pd.concat(frames, ignore_index=True)
        missing = sorted(set(tickers) - set(out["ticker"].unique()))
        if missing:
            log.warning("no rows returned for tickers: %s", missing)
        return out
