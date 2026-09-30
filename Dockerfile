# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MP_DBT_DIR=/app/dbt \
    MLFLOW_DISABLE_AGENT_HINT=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer stays cached across source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra orchestration --no-install-project

COPY README.md ./
COPY src ./src
COPY dbt ./dbt
COPY config ./config
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra orchestration

RUN useradd --create-home --uid 1000 app \
    && mkdir -p data models reports state mlruns \
    && chown -R app:app /app
USER app

EXPOSE 8000
ENTRYPOINT ["market-pipeline"]
CMD ["serve"]
