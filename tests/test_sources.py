from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_pipeline.sources import PRICE_COLUMNS, SyntheticSource, YahooSource, get_source
from market_pipeline.sources import yahoo as yahoo_mod
from market_pipeline.sources.yahoo import normalise_download


def _fake_yf_frame(tickers: list[str], days: int = 3) -> pd.DataFrame:
    """Shape yfinance returns for group_by='ticker': (Ticker, Price) MultiIndex columns."""
    idx = pd.DatetimeIndex(pd.bdate_range("2026-09-01", periods=days), name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Price"])
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 101, size=(days, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_normalise_keeps_every_ticker() -> None:
    """Regression test: the original pipeline silently dropped all but the first ticker."""
    tidy = normalise_download(_fake_yf_frame(["AAPL", "MSFT", "TSLA"]))
    assert list(tidy.columns) == PRICE_COLUMNS
    assert sorted(tidy["ticker"].unique()) == ["AAPL", "MSFT", "TSLA"]
    assert len(tidy) == 9
    assert not tidy.duplicated(["date", "ticker"]).any()


def test_normalise_drops_rows_missing_all_prices() -> None:
    raw = _fake_yf_frame(["AAPL", "NEWCO"])
    raw.loc[raw.index[0], ("NEWCO", slice(None))] = np.nan  # not listed yet on day 1
    tidy = normalise_download(raw)
    assert len(tidy[tidy["ticker"] == "NEWCO"]) == 2


def test_yahoo_fetch_batches_and_tolerates_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_download(tickers: list[str], **_: object) -> pd.DataFrame:
        calls.append(tickers)
        return pd.DataFrame() if "BAD" in tickers else _fake_yf_frame(tickers)

    monkeypatch.setattr(yahoo_mod.yf, "download", fake_download)
    monkeypatch.setattr(YahooSource._download.retry, "wait", lambda *_: 0)  # type: ignore[attr-defined]
    src = YahooSource(batch_size=2)
    out = src.fetch(["A", "B", "BAD"], date(2026, 9, 1), date(2026, 9, 5))
    assert sorted(out["ticker"].unique()) == ["A", "B"]
    assert calls.count(["BAD"]) == 4  # retried with backoff, then skipped


def test_synthetic_is_deterministic_and_window_consistent() -> None:
    src = SyntheticSource()
    full = src.fetch(["AAA"], date(2024, 1, 1), date(2024, 3, 1))
    part = src.fetch(["AAA"], date(2024, 2, 1), date(2024, 2, 15))
    merged = part.merge(full, on=["date", "ticker"], suffixes=("", "_full"))
    assert len(merged) == len(part) > 0
    assert np.allclose(merged["close"], merged["close_full"])
    assert (full["high"] >= full[["open", "close"]].max(axis=1)).all()
    assert (full["low"] <= full[["open", "close"]].min(axis=1)).all()


def test_synthetic_paths_are_finite_for_many_tickers() -> None:
    """Every simulated path must stay finite over the full horizon (GARCH stationarity)."""
    tickers = [f"T{i}" for i in range(60)] + ["NVDA", "TSLA"]
    df = SyntheticSource().fetch(tickers, date(2010, 1, 1), date(2027, 1, 1))
    prices = df[["open", "high", "low", "close", "volume"]].to_numpy()
    assert np.isfinite(prices).all() and (prices > 0).all()


def test_get_source_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        get_source("bloomberg")
