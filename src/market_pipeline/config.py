"""Typed pipeline configuration.

Precedence (highest first): explicit kwargs > MP_* environment variables > YAML file.
The YAML path defaults to ``config/pipeline.yaml`` and can be changed with ``MP_CONFIG``.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("config/pipeline.yaml")


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    warehouse: Path = Path("data/warehouse.duckdb")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze" / "prices"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine" / "prices"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "_runs"


class MLConfig(BaseModel):
    horizon_days: int = Field(5, ge=1)
    n_splits: int = Field(5, ge=2)
    test_days: int = Field(252, ge=20)
    min_train_days: int = Field(504, ge=60)
    min_improvement_vs_naive: float = 0.05
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "volatility-forecasting"
    lgbm_params: dict[str, Any] = Field(default_factory=dict)


class AzureConfig(BaseModel):
    container: str = "market-data"
    prefix: str = "market-pipeline"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MP_", env_nested_delimiter="__", extra="ignore", populate_by_name=True
    )

    source: Literal["yahoo", "synthetic"] = "yahoo"
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    start_date: date = date(2015, 1, 1)
    lookback_days: int = Field(5, ge=0)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    azure_connection_string: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("azure_connection_string", "AZURE_STORAGE_CONNECTION_STRING"),
    )

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, v: list[str]) -> list[str]:
        cleaned = sorted({t.strip().upper() for t in v if t.strip()})
        if not cleaned:
            raise ValueError("at least one ticker is required")
        return cleaned

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("MP_CONFIG", DEFAULT_CONFIG_PATH))
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if yaml_path.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
