"""Command-line entry point: ``market-pipeline <step>``."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Annotated

import typer

from market_pipeline.config import Settings, get_settings

os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

app = typer.Typer(add_completion=False, help="Market data lakehouse + volatility forecasting.")

SourceOpt = Annotated[
    str | None, typer.Option("--source", "-s", help="Override the data source (yahoo | synthetic).")
]


def _settings(source: str | None = None) -> Settings:
    settings = get_settings()
    return settings.model_copy(update={"source": source}) if source else settings


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("mlflow", "alembic", "urllib3", "yfinance", "azure", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def ingest(source: SourceOpt = None) -> None:
    """Extract bars incrementally, validate against the contract, and land them in bronze."""
    from market_pipeline.ingest import run_ingest

    result = run_ingest(_settings(source))
    typer.echo(json.dumps(asdict(result), indent=2, default=str))


@app.command()
def transform(
    select: Annotated[str | None, typer.Option(help="dbt node selector")] = None,
) -> None:
    """Build and test the dbt models (staging -> marts)."""
    from market_pipeline.warehouse import run_dbt

    settings = _settings()
    run_dbt(settings, ["source", "freshness"])
    run_dbt(settings, ["build", *(["--select", select] if select else [])])


@app.command()
def train(no_mlflow: Annotated[bool, typer.Option("--no-mlflow")] = False) -> None:
    """Walk-forward CV all candidates, gate the champion and promote it."""
    from market_pipeline.ml.train import train as run_train

    result = run_train(_settings(), log_mlflow=not no_mlflow)
    cols = ["qlike_mean", "rmse_log_mean", "r2_log_mean", "qlike_improvement_vs_naive"]
    typer.echo(result.summary[cols].round(4).to_string())
    typer.echo(f"\nchampion={result.champion} promoted={result.promoted} version={result.version}")


@app.command()
def predict() -> None:
    """Score the latest session for every ticker and persist forecasts."""
    from market_pipeline.ml.predict import run_predict

    typer.echo(run_predict(_settings()).round(4).to_string(index=False))


@app.command()
def monitor(
    fail_on_alert: Annotated[bool, typer.Option(help="Exit 1 if drift or degradation")] = False,
) -> None:
    """Check feature drift (PSI) and live forecast accuracy."""
    from market_pipeline.monitor import run_monitor

    report = run_monitor(_settings())
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "drift"}, indent=2))
    if fail_on_alert and not report["healthy"]:
        raise typer.Exit(1)


@app.command()
def publish() -> None:
    """Export gold tables to Parquet and upload them to Azure Blob Storage."""
    from market_pipeline.publish import run_publish

    for blob in run_publish(_settings()):
        typer.echo(blob)


@app.command()
def run(
    source: SourceOpt = None,
    retrain: Annotated[bool, typer.Option(help="Retrain before scoring")] = True,
    publish_: Annotated[bool, typer.Option("--publish/--no-publish")] = False,
) -> None:
    """Run the whole pipeline end to end."""
    from market_pipeline.orchestration import run_pipeline

    run_pipeline(_settings(source), retrain=retrain, publish=publish_)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the forecasting API."""
    import uvicorn

    uvicorn.run("market_pipeline.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
