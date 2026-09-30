from pathlib import Path

import pytest

from market_pipeline.config import Settings


def test_yaml_then_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    cfg = tmp_path / "p.yaml"
    cfg.write_text("tickers: [msft, aapl, msft]\nml:\n  horizon_days: 5\n  n_splits: 4\n")
    monkeypatch.setenv("MP_CONFIG", str(cfg))
    monkeypatch.setenv("MP_ML__HORIZON_DAYS", "10")

    s = Settings()
    assert s.tickers == ["AAPL", "MSFT"]  # upper-cased, de-duplicated, sorted
    assert s.ml.n_splits == 4  # from YAML
    assert s.ml.horizon_days == 10  # env wins over YAML


def test_azure_connection_string_is_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountKey=supersecret")
    s = Settings()
    assert s.azure_connection_string is not None
    assert "supersecret" not in repr(s)
