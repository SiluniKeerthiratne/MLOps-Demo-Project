# Multi-stage build. Poetry lives only in the builder stages; final images contain just the
# installed packages. Targets: trainer (pipeline + DVC), dev (trainer + api + test tools), api.
ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=2.2.1

FROM python:${PYTHON_VERSION}-slim AS builder-base
ARG POETRY_VERSION
ENV PIP_NO_CACHE_DIR=1 POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1
RUN python -m venv /opt/poetry && /opt/poetry/bin/pip install "poetry==${POETRY_VERSION}"
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml poetry.lock ./

FROM builder-base AS deps-trainer
RUN /opt/poetry/bin/poetry install --no-root --only main,train,data

FROM builder-base AS deps-dev
RUN /opt/poetry/bin/poetry install --no-root --only main,train,data,api,dev

FROM builder-base AS deps-api
RUN /opt/poetry/bin/poetry install --no-root --only main,api

# ---- trainer: runs DVC, training, evaluation, promotion, export ----
FROM python:${PYTHON_VERSION}-slim AS trainer
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    MLFLOW_DISABLE_AGENT_HINT=1 DVC_NO_ANALYTICS=1
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory /app
COPY --from=deps-trainer /opt/venv /opt/venv
WORKDIR /app
# The repo is bind-mounted at /app by docker compose.

# ---- dev: trainer + API deps + pytest/ruff, used by `make test` / `make lint` ----
FROM trainer AS dev
COPY --from=deps-dev /opt/venv /opt/venv

# ---- api: minimal runtime, non-root ----
FROM python:${PYTHON_VERSION}-slim AS api
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    MODELS_DIR=/app/models PREDICTION_LOG=/app/logs/predictions.jsonl
RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs /app/models && chown -R app /app
COPY --from=deps-api /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app api/ ./api/
COPY --chown=app config.yaml ./config.yaml
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
