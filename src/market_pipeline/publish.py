"""Publish step: export gold tables to Parquet and upload them to Azure Blob Storage.

Every publish writes a dated snapshot plus a ``latest/`` copy, so downstream readers
(Synapse, Fabric, Power BI) can either pin a snapshot or always read current data.
Works against real Azure or the Azurite emulator (see docker-compose.yml).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

from market_pipeline.config import Settings
from market_pipeline.ml.train import CARD_FILE
from market_pipeline.warehouse import connect

log = logging.getLogger(__name__)

GOLD_TABLES = [
    "fct_daily_returns",
    "features_volatility",
    "agg_ticker_monthly",
    "volatility_forecasts",
]


def export_gold(settings: Settings) -> dict[str, Path]:
    out = settings.paths.gold_dir / "exports"
    out.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    with connect(settings) as con:
        existing = {
            r[0]
            for r in con.sql(
                "select table_name from information_schema.tables where table_schema = 'marts'"
            ).fetchall()
        }
        for table in GOLD_TABLES:
            if table not in existing:
                continue
            path = out / f"{table}.parquet"
            con.execute(f"copy marts.{table} to '{path}' (format parquet, compression zstd)")
            exported[table] = path
    return exported


def run_publish(settings: Settings, client: BlobServiceClient | None = None) -> list[str]:
    exported = export_gold(settings)
    card = settings.paths.models_dir / CARD_FILE
    if card.exists():
        exported["model_card"] = card

    if client is None:
        if settings.azure_connection_string is None:
            log.warning("AZURE_STORAGE_CONNECTION_STRING not set; exported locally only")
            return []
        client = BlobServiceClient.from_connection_string(
            settings.azure_connection_string.get_secret_value()
        )

    container = client.get_container_client(settings.azure.container)
    with suppress(ResourceExistsError):
        container.create_container()

    snapshot = datetime.now(UTC).strftime("%Y-%m-%d")
    uploaded = []
    for name, path in exported.items():
        is_json = path.suffix == ".json"
        content = ContentSettings(
            content_type="application/json" if is_json else "application/vnd.apache.parquet"
        )
        for key in (f"snapshot_date={snapshot}", "latest"):
            blob = f"{settings.azure.prefix}/gold/{name}/{key}/{path.name}"
            with path.open("rb") as fh:
                container.upload_blob(blob, fh, overwrite=True, content_settings=content)
            uploaded.append(blob)
    log.info("published %d blobs to container '%s'", len(uploaded), settings.azure.container)
    return uploaded
