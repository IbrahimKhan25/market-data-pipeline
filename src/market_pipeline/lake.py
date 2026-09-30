"""Local lake layout (bronze / quarantine / gold) on Parquet.

Bronze is an append-only log: every ingest run writes a new file under
``ingest_date=YYYY-MM-DD/`` and never rewrites existing files. That keeps it an exact
audit trail of what each vendor call returned. The dbt staging layer handles
deduplication, keeping the latest observation per (ticker, date), so re-running any
ingest is idempotent downstream.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from market_pipeline.config import PathsConfig


class Lake:
    def __init__(self, paths: PathsConfig) -> None:
        self.paths = paths

    @property
    def bronze_glob(self) -> str:
        return str(self.paths.bronze_dir / "*" / "*.parquet")

    def has_bronze(self) -> bool:
        return any(self.paths.bronze_dir.glob("*/*.parquet"))

    def sources(self) -> set[str]:
        """Distinct sources already landed in bronze."""
        if not self.has_bronze():
            return set()
        rows = duckdb.sql(f"select distinct source from read_parquet('{self.bronze_glob}')")
        return {r[0] for r in rows.fetchall()}

    def watermarks(self) -> dict[str, date]:
        """Latest trading date already landed in bronze, per ticker."""
        if not self.has_bronze():
            return {}
        rows = duckdb.sql(
            f"select ticker, max(date)::date from read_parquet('{self.bronze_glob}') group by 1"
        ).fetchall()
        return {ticker: d for ticker, d in rows}

    @staticmethod
    def _write(df: pd.DataFrame, directory: Path, run_id: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{run_id}.parquet"
        df.to_parquet(path, index=False)
        return path

    def write_bronze(self, df: pd.DataFrame, ingest_date: date, run_id: str) -> Path:
        return self._write(df, self.paths.bronze_dir / f"ingest_date={ingest_date}", run_id)

    def write_quarantine(self, df: pd.DataFrame, ingest_date: date, run_id: str) -> Path:
        return self._write(df, self.paths.quarantine_dir / f"ingest_date={ingest_date}", run_id)

    def write_manifest(self, run_id: str, manifest: dict[str, Any]) -> Path:
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.runs_dir / f"{run_id}.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path
