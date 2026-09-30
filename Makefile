.DEFAULT_GOAL := help
UV := uv run

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies (incl. dev + Prefect)
	uv sync --all-extras
	uv run pre-commit install || true

lint:  ## Ruff lint + format check + mypy
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) mypy src

format:  ## Auto-format
	$(UV) ruff format src tests
	$(UV) ruff check --fix src tests

test:  ## Run the test suite with coverage
	$(UV) pytest --cov=market_pipeline --cov-report=term-missing

pipeline:  ## Run the full pipeline against Yahoo Finance
	$(UV) market-pipeline run

DEMO_ENV := MP_PATHS__DATA_DIR=demo/data MP_PATHS__WAREHOUSE=demo/data/warehouse.duckdb \
	MP_PATHS__MODELS_DIR=demo/models MP_PATHS__REPORTS_DIR=demo/reports \
	MP_ML__MLFLOW_TRACKING_URI=sqlite:///demo/mlflow.db

demo:  ## Run the full pipeline offline on simulated data (isolated in demo/)
	mkdir -p demo && $(DEMO_ENV) $(UV) market-pipeline run --source synthetic

serve:  ## Start the forecasting API on :8000
	$(UV) market-pipeline serve

mlflow:  ## Open the MLflow UI on :5000
	$(UV) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

dbt-docs:  ## Generate and serve dbt docs (lineage graph)
	cd dbt && MP_WAREHOUSE_PATH=../data/warehouse.duckdb MP_DATA_DIR=../data ../.venv/bin/dbt docs generate --profiles-dir . && ../.venv/bin/dbt docs serve --profiles-dir . --port 8081

up:  ## Full stack in Docker (pipeline, API, MLflow, Azurite)
	docker compose up --build

clean:  ## Remove generated data, models and caches
	rm -rf data demo models reports mlruns mlflow.db dbt/target dbt/logs .pytest_cache .mypy_cache .ruff_cache .coverage

.PHONY: help install lint format test pipeline demo serve mlflow dbt-docs up clean
