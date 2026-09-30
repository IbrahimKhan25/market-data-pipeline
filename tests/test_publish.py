import os
from typing import Any

import pytest
from azure.core.exceptions import ResourceExistsError

from market_pipeline.publish import run_publish


class _FakeContainer:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.created = False

    def create_container(self) -> None:
        if self.created:
            raise ResourceExistsError("exists")
        self.created = True

    def upload_blob(self, name: str, data: Any, overwrite: bool, **_: Any) -> None:
        assert overwrite
        self.blobs[name] = data.read()


class _FakeService:
    def __init__(self) -> None:
        self.container = _FakeContainer()

    def get_container_client(self, name: str) -> _FakeContainer:
        return self.container


def test_publish_uploads_snapshot_and_latest(built) -> None:  # type: ignore[no-untyped-def]
    svc = _FakeService()
    first = run_publish(built.settings, client=svc)  # type: ignore[arg-type]
    run_publish(built.settings, client=svc)  # container already exists: must not fail

    names = set(svc.container.blobs)
    prefix = built.settings.azure.prefix
    assert f"{prefix}/gold/features_volatility/latest/features_volatility.parquet" in names
    assert f"{prefix}/gold/model_card/latest/model_card.json" in names
    assert any("snapshot_date=" in n for n in names)
    assert all(svc.container.blobs[n][:4] == b"PAR1" for n in names if n.endswith(".parquet"))
    assert len(first) == len(names)


def test_publish_without_credentials_exports_locally(built) -> None:  # type: ignore[no-untyped-def]
    assert run_publish(built.settings) == []
    assert (built.settings.paths.gold_dir / "exports" / "fct_daily_returns.parquet").exists()


@pytest.mark.integration
@pytest.mark.skipif("AZURITE_CONNECTION_STRING" not in os.environ, reason="Azurite not running")
def test_publish_to_azurite(built) -> None:  # type: ignore[no-untyped-def]
    from azure.storage.blob import BlobServiceClient

    svc = BlobServiceClient.from_connection_string(os.environ["AZURITE_CONNECTION_STRING"])
    uploaded = run_publish(built.settings, client=svc)
    container = svc.get_container_client(built.settings.azure.container)
    assert set(uploaded) <= {b.name for b in container.list_blobs()}
