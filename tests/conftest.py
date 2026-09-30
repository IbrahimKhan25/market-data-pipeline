"""Shared fixtures. Everything runs offline against the deterministic synthetic source."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from market_pipeline.config import MLConfig, PathsConfig, Settings
from market_pipeline.ingest import run_ingest
from market_pipeline.ml.train import TrainResult, train
from market_pipeline.sources import SyntheticSource
from market_pipeline.warehouse import run_dbt

AS_OF = date(2026, 9, 25)
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def make_settings(root: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "source": "synthetic",
        "tickers": TICKERS,
        "start_date": date(2016, 1, 1),
        "lookback_days": 5,
        "azure_connection_string": None,  # never pick up real credentials from the shell
        "paths": PathsConfig(
            data_dir=root / "data",
            warehouse=root / "data" / "warehouse.duckdb",
            models_dir=root / "models",
            reports_dir=root / "reports",
        ),
        "ml": MLConfig(
            n_splits=3,
            test_days=126,
            min_train_days=504,
            mlflow_tracking_uri=f"sqlite:///{root / 'mlflow.db'}",
            lgbm_params={"n_estimators": 120, "learning_rate": 0.05, "min_child_samples": 50},
        ),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@dataclass
class Built:
    settings: Settings
    trained: TrainResult


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    """A fully built lakehouse: ingest -> dbt build -> train (with MLflow)."""
    settings = make_settings(tmp_path_factory.mktemp("lakehouse"))
    run_ingest(settings, source=SyntheticSource(), as_of=AS_OF)
    run_dbt(settings, ["build"])
    return Built(settings, train(settings, log_mlflow=True))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)
