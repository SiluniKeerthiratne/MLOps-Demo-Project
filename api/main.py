"""FastAPI service for the exported production model."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.model_loader import LoadedModel, load_model
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)


def _default_log_path() -> str:
    cfg = Path("config.yaml")
    if cfg.exists():
        return yaml.safe_load(cfg.read_text())["paths"]["prediction_log"]
    return "logs/predictions.jsonl"


class PredictionLog:
    """Append-only JSONL log; one line per /predict request."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as exc:  # logging must never take the service down
            print(f"WARNING: could not write prediction log: {exc}", flush=True)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self._lock, open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def compute_stats(records: list[dict]) -> dict:
    latencies = sorted(r["latency_ms"] for r in records if "latency_ms" in r)
    p95 = None
    if latencies:
        p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
    classes: dict[str, int] = {}
    for r in records:
        if r.get("status") == "ok" and r.get("prediction") is not None:
            key = str(r["prediction"])
            classes[key] = classes.get(key, 0) + 1
    return {
        "request_count": len(records),
        "error_count": sum(1 for r in records if r.get("status") != "ok"),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_ms": p95,
        "class_distribution": classes,
        "last_request_time": records[-1]["ts"] if records else None,
    }


class InputError(Exception):
    def __init__(self, detail, status_code: int = 422):
        self.detail = detail
        self.status_code = status_code


def validate_features(body, model: LoadedModel) -> tuple[dict, list[str]]:
    """Return ({feature: float}, warnings) or raise InputError."""
    if not isinstance(body, dict) or not isinstance(body.get("features"), dict):
        raise InputError('Body must be a JSON object like {"features": {"name": number, ...}}')
    given = body["features"]
    expected = model.feature_names
    missing = [f for f in expected if f not in given]
    unknown = [f for f in given if f not in expected]
    if missing or unknown:
        raise InputError({"missing_features": missing, "unknown_features": unknown})
    problems = {}
    values: dict[str, float] = {}
    for name in expected:
        v = given[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            problems[name] = f"expected a number, got {type(v).__name__}"
        elif not math.isfinite(v):
            problems[name] = "value must be finite (no NaN/inf)"
        else:
            values[name] = float(v)
    if problems:
        raise InputError({"invalid_values": problems})
    warnings = []
    for spec in model.schema["features"]:
        v = values[spec["name"]]
        if v < spec["min"] or v > spec["max"]:
            warnings.append(
                f"{spec['name']}={v} is outside the training range [{spec['min']}, {spec['max']}]"
            )
    return values, warnings


def create_app(models_dir: Path | None = None, log_path: Path | None = None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("MODELS_DIR", "models"))
    log = PredictionLog(Path(log_path or os.environ.get("PREDICTION_LOG", _default_log_path())))
    started = time.monotonic()
    state = {"model": LoadedModel(error="not loaded yet")}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["model"] = load_model(models_dir)
        m = state["model"]
        print(
            f"Model loaded: v{m.version} (dataset {m.dataset_version})"
            if m.loaded
            else f"MODEL LOAD FAILED: {m.error}",
            flush=True,
        )
        yield

    app = FastAPI(title="Wine classifier", version="1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health():
        m = state["model"]
        body = HealthResponse(
            status="ok" if m.loaded else "unhealthy",
            model_loaded=m.loaded,
            model_version=m.version,
            dataset_version=m.dataset_version,
            mlflow_run_id=m.metadata.get("run_id"),
            uptime_seconds=round(time.monotonic() - started, 1),
            error=m.error,
        )
        return JSONResponse(body.model_dump(), status_code=200 if m.loaded else 503)

    @app.get("/stats", response_model=StatsResponse)
    def stats():
        return compute_stats(log.read())

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": PredictRequest.model_json_schema()}},
            }
        },
    )
    async def predict(request: Request):
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex
        m = state["model"]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "model_version": m.version,
            "dataset_version": m.dataset_version,
            "input": None,
            "prediction": None,
            "confidence": None,
            "warnings": [],
        }

        def finish(status_code: int, payload: dict, error=None):
            record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            record["status"] = "ok" if status_code == 200 else "error"
            record["http_status"] = status_code
            if error is not None:
                record["error"] = error
            log.append(record)
            return JSONResponse(payload, status_code=status_code)

        try:
            raw = await request.body()
            if not raw.strip():
                raise InputError("Empty request body", 422)
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise InputError(f"Malformed JSON: {exc}", 400) from exc
            record["input"] = body.get("features") if isinstance(body, dict) else body
            if not m.loaded:
                return finish(
                    503,
                    {"detail": f"Model not loaded: {m.error}", "request_id": request_id},
                    m.error,
                )
            values, warnings = validate_features(body, m)
            record["warnings"] = warnings
            X = pd.DataFrame([values], columns=m.feature_names)
            proba = m.pipeline.predict_proba(X)[0]
            classes = [int(c) for c in m.pipeline.classes_]
            best = int(proba.argmax())
            record["prediction"] = classes[best]
            record["confidence"] = round(float(proba[best]), 6)
            payload = {
                "prediction": classes[best],
                "probabilities": {
                    str(c): round(float(p), 6) for c, p in zip(classes, proba, strict=True)
                },
                "model_version": m.version,
                "dataset_version": m.dataset_version,
                "request_id": request_id,
                "warnings": warnings,
            }
            return finish(200, payload)
        except InputError as exc:
            return finish(
                exc.status_code,
                {"detail": exc.detail, "request_id": request_id},
                exc.detail,
            )
        except Exception as exc:  # noqa: BLE001 - never leak a 500 traceback to clients
            return finish(
                500,
                {"detail": "Internal error while predicting", "request_id": request_id},
                f"{type(exc).__name__}: {exc}",
            )

    return app


app = create_app()
