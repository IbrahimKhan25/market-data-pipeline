"""Pipeline orchestration.

``run_pipeline`` is plain Python, and the CLI and CI use it directly. If Prefect is
installed (``uv sync --extra orchestration``), ``daily_flow`` wraps the same steps as
Prefect tasks with retries and a weekday schedule:

    python -m market_pipeline.orchestration          # serve on a cron schedule
"""

from __future__ import annotations

import logging
from typing import Any

from market_pipeline.config import Settings, get_settings
from market_pipeline.ingest import run_ingest
from market_pipeline.ml.predict import run_predict
from market_pipeline.ml.train import load_model, train
from market_pipeline.monitor import run_monitor
from market_pipeline.publish import run_publish
from market_pipeline.warehouse import run_dbt

log = logging.getLogger(__name__)


def _needs_training(settings: Settings, retrain: bool) -> bool:
    if retrain:
        return True
    try:
        load_model(settings)
        return False
    except FileNotFoundError:
        return True


def run_pipeline(settings: Settings, retrain: bool = True, publish: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["ingest"] = run_ingest(settings).rows_landed
    run_dbt(settings, ["build"])
    if _needs_training(settings, retrain):
        result = train(settings)
        summary["train"] = {"champion": result.champion, "promoted": result.promoted}
    summary["predict"] = len(run_predict(settings))
    summary["healthy"] = run_monitor(settings)["healthy"]
    if publish:
        summary["published"] = len(run_publish(settings))
    log.info("pipeline complete: %s", summary)
    return summary


try:
    from prefect import flow, task

    @task(retries=3, retry_delay_seconds=60)
    def ingest_task(settings: Settings) -> int:
        return run_ingest(settings).rows_landed

    @task(retries=1)
    def transform_task(settings: Settings) -> None:
        run_dbt(settings, ["build"])

    @task
    def train_task(settings: Settings) -> dict[str, Any]:
        r = train(settings)
        return {"champion": r.champion, "promoted": r.promoted, "version": r.version}

    @task
    def predict_task(settings: Settings) -> int:
        return len(run_predict(settings))

    @task
    def monitor_task(settings: Settings) -> dict[str, Any]:
        return run_monitor(settings)

    @task(retries=3, retry_delay_seconds=30)
    def publish_task(settings: Settings) -> int:
        return len(run_publish(settings))

    @flow(name="market-data-daily", log_prints=True)
    def daily_flow(retrain: bool = False, publish: bool = True) -> dict[str, Any]:
        settings = get_settings()
        rows = ingest_task(settings)
        transform_task(settings, wait_for=[rows])
        trained = train_task(settings) if _needs_training(settings, retrain) else None
        n = predict_task(settings, wait_for=[trained])
        report = monitor_task(settings, wait_for=[n])
        if not report["healthy"]:
            # Drift or degraded live accuracy: retrain now, not at the weekly job.
            train_task(settings)
            predict_task(settings)
        published = publish_task(settings) if publish else 0
        return {"rows": rows, "trained": trained, "forecasts": n, "published": published}

    @flow(name="market-data-weekly-retrain")
    def weekly_retrain_flow() -> dict[str, Any]:
        return daily_flow(retrain=True, publish=True)

except ImportError:  # pragma: no cover - prefect is an optional extra
    daily_flow = None  # type: ignore[assignment]
    weekly_retrain_flow = None  # type: ignore[assignment]


if __name__ == "__main__":
    if daily_flow is None:
        raise SystemExit("prefect not installed: uv sync --extra orchestration")
    from prefect import serve

    # Daily run after the US close (22:30 UTC); full retrain Saturday morning.
    serve(
        daily_flow.to_deployment("daily", cron="30 22 * * 1-5"),  # type: ignore[arg-type]
        weekly_retrain_flow.to_deployment("weekly-retrain", cron="0 6 * * 6"),  # type: ignore[arg-type]
    )
