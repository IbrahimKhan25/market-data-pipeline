"""Extract step: incremental pull from a source, validate, then land in bronze."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from market_pipeline.config import Settings
from market_pipeline.lake import Lake
from market_pipeline.schemas import split_valid
from market_pipeline.sources import PRICE_COLUMNS, MarketDataSource, get_source

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    run_id: str
    source: str
    rows_fetched: int = 0
    rows_landed: int = 0
    rows_quarantined: int = 0
    duplicates_dropped: int = 0
    windows: dict[str, list[str]] = field(default_factory=dict)
    bronze_path: str | None = None
    quarantine_path: str | None = None
    duration_s: float = 0.0


def plan_windows(
    tickers: list[str], watermarks: dict[str, date], start_date: date, lookback_days: int, end: date
) -> dict[date, list[str]]:
    """Group tickers by the date their fetch should start from.

    New tickers backfill from ``start_date``. Known tickers re-fetch from
    ``watermark - lookback_days``, so late vendor corrections are picked up.
    Grouping keeps the number of vendor calls small.
    """
    windows: dict[date, list[str]] = defaultdict(list)
    for ticker in tickers:
        wm = watermarks.get(ticker)
        start = start_date if wm is None else max(start_date, wm - timedelta(days=lookback_days))
        if start < end:
            windows[start].append(ticker)
    return dict(windows)


def run_ingest(
    settings: Settings,
    source: MarketDataSource | None = None,
    as_of: date | None = None,
) -> IngestResult:
    t0 = time.perf_counter()
    source = source or get_source(settings.source)
    lake = Lake(settings.paths)
    today = as_of or datetime.now(UTC).date()
    end = today + timedelta(days=1)  # sources treat end as exclusive
    run_id = f"{today:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    result = IngestResult(run_id=run_id, source=source.name)

    existing = lake.sources() - {source.name}
    if existing:
        # Mixing vendors in one lake would let the dedup step silently swap one for the other.
        raise ValueError(
            f"lake at {settings.paths.data_dir} already holds {sorted(existing)} data; "
            f"use a separate data_dir for source {source.name!r}"
        )

    windows = plan_windows(
        settings.tickers, lake.watermarks(), settings.start_date, settings.lookback_days, end
    )
    result.windows = {str(k): v for k, v in windows.items()}
    log.info("ingest %s: %d fetch window(s) %s", run_id, len(windows), result.windows)

    frames = [source.fetch(tickers, start, end) for start, tickers in windows.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.warning("ingest %s: source returned no rows", run_id)
        result.duration_s = round(time.perf_counter() - t0, 2)
        lake.write_manifest(run_id, asdict(result))
        return result

    raw = pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
    result.rows_fetched = len(raw)
    deduped = raw.drop_duplicates(subset=["date", "ticker"], keep="last")
    result.duplicates_dropped = len(raw) - len(deduped)

    valid, rejected = split_valid(deduped)
    ingested_at = datetime.now(UTC).replace(tzinfo=None)
    meta: dict[str, Any] = {"ingested_at": ingested_at, "source": source.name, "run_id": run_id}

    if not valid.empty:
        result.bronze_path = str(lake.write_bronze(valid.assign(**meta), today, run_id))
        result.rows_landed = len(valid)
    if not rejected.empty:
        result.quarantine_path = str(lake.write_quarantine(rejected.assign(**meta), today, run_id))
        result.rows_quarantined = len(rejected)
        log.warning("ingest %s: quarantined %d row(s)", run_id, len(rejected))

    result.duration_s = round(time.perf_counter() - t0, 2)
    lake.write_manifest(run_id, asdict(result))
    log.info(
        "ingest %s: fetched=%d landed=%d quarantined=%d in %.1fs",
        run_id,
        result.rows_fetched,
        result.rows_landed,
        result.rows_quarantined,
        result.duration_s,
    )
    return result
