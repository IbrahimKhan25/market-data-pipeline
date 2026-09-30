from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from market_pipeline.config import Settings
from market_pipeline.ingest import plan_windows, run_ingest
from market_pipeline.lake import Lake
from market_pipeline.sources import SyntheticSource


def test_plan_windows_backfills_new_and_increments_known() -> None:
    wm = {"AAA": date(2026, 9, 20), "BBB": date(2026, 9, 20), "CCC": date(2026, 9, 1)}
    windows = plan_windows(
        ["AAA", "BBB", "CCC", "NEW"], wm, date(2020, 1, 1), lookback_days=5, end=date(2026, 9, 26)
    )
    assert windows == {
        date(2026, 9, 15): ["AAA", "BBB"],
        date(2026, 8, 27): ["CCC"],
        date(2020, 1, 1): ["NEW"],
    }


def test_incremental_runs_append_and_only_fetch_the_tail(settings: Settings) -> None:
    src = SyntheticSource()
    first = run_ingest(settings, source=src, as_of=date(2026, 9, 18))
    second = run_ingest(settings, source=src, as_of=date(2026, 9, 25))

    assert first.rows_landed > 10_000
    # Second run re-fetches only watermark - lookback .. as_of: ~2 weeks x 4 tickers.
    assert 0 < second.rows_landed < 60
    assert list(second.windows) == [str(date(2026, 9, 18) - timedelta(days=5))]

    lake = Lake(settings.paths)
    files = sorted(settings.paths.bronze_dir.glob("*/*.parquet"))
    assert len(files) == 2  # append-only: one immutable file per run
    assert all(d == date(2026, 9, 25) for d in lake.watermarks().values())
    assert (settings.paths.runs_dir / f"{second.run_id}.json").exists()


def test_rerunning_is_idempotent_after_dedup(settings: Settings) -> None:
    src = SyntheticSource()
    for _ in range(3):
        run_ingest(settings, source=src, as_of=date(2026, 9, 25))
    glob = Lake(settings.paths).bronze_glob
    raw, distinct = duckdb.sql(
        f"select count(*), count(distinct (ticker, date)) from read_parquet('{glob}')"
    ).fetchone()  # type: ignore[misc]
    assert raw > distinct  # bronze keeps every observation...
    # ...and staging's dedup (latest ingested_at wins) recovers exactly one row per key.
    deduped = duckdb.sql(
        f"""select count(*) from (
                select * from read_parquet('{glob}')
                qualify row_number() over (partition by ticker, date order by ingested_at desc) = 1
            )"""
    ).fetchone()
    assert deduped == (distinct,)


def test_refuses_to_mix_sources_in_one_lake(settings: Settings) -> None:
    run_ingest(settings, source=SyntheticSource(), as_of=date(2026, 9, 25))

    class OtherVendor(SyntheticSource):
        name = "other"

    with pytest.raises(ValueError, match="already holds"):
        run_ingest(settings, source=OtherVendor(), as_of=date(2026, 9, 26))


class _FlakySource(SyntheticSource):
    """Vendor that returns one in-progress bar (no close yet) and a duplicate row."""

    def fetch(self, tickers, start, end):  # type: ignore[no-untyped-def]
        df = super().fetch(tickers, start, end)
        df.loc[df.index[-1], ["close", "adj_close"]] = None
        return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def test_bad_rows_go_to_quarantine_not_bronze(settings: Settings) -> None:
    result = run_ingest(settings, source=_FlakySource(), as_of=date(2026, 9, 25))
    assert result.duplicates_dropped == 1
    assert result.rows_quarantined == 1
    quarantined = pd.read_parquet(result.quarantine_path)
    assert "close:not_nullable" in quarantined["_reject_reason"].iloc[0]
    bronze = pd.read_parquet(result.bronze_path)
    assert bronze["close"].notna().all()
