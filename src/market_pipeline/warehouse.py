"""Transform step: runs the dbt project against the DuckDB warehouse, and provides read helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from market_pipeline.config import Settings

log = logging.getLogger(__name__)

DBT_DIR = Path(os.environ.get("MP_DBT_DIR", Path(__file__).resolve().parents[2] / "dbt"))


class TransformError(RuntimeError):
    pass


def dbt_env(settings: Settings) -> dict[str, str]:
    return {
        "MP_WAREHOUSE_PATH": str(settings.paths.warehouse.resolve()),
        "MP_DATA_DIR": str(settings.paths.data_dir.resolve()),
    }


def run_dbt(settings: Settings, args: list[str] | None = None) -> None:
    """Run dbt (``dbt build`` by default: models + tests in DAG order).

    dbt runs in a subprocess so its DuckDB handle is released before we read the warehouse.
    """
    args = args or ["build"]
    settings.paths.warehouse.parent.mkdir(parents=True, exist_ok=True)
    dbt_bin = Path(sys.executable).with_name("dbt")
    cmd = [
        str(dbt_bin),
        "--no-use-colors",
        *args,
        "--project-dir",
        str(DBT_DIR),
        "--profiles-dir",
        str(DBT_DIR),
    ]
    log.info("dbt %s", " ".join(args))
    proc = subprocess.run(
        cmd, env={**os.environ, **dbt_env(settings)}, capture_output=True, text=True, check=False
    )
    for line in proc.stdout.splitlines():
        if any(k in line for k in ("ERROR", "FAIL", "WARN", "Done.", "Finished")):
            log.info("dbt | %s", line.strip())
    if proc.returncode != 0:
        raise TransformError(
            f"dbt {' '.join(args)} failed:\n{proc.stdout[-3000:]}{proc.stderr[-2000:]}"
        )


@contextmanager
def connect(settings: Settings, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(settings.paths.warehouse), read_only=read_only)
    try:
        yield con
    finally:
        con.close()
