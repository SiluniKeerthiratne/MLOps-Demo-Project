"""Pydantic models for API responses (request bodies are validated by hand against schema.json)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Documentation model: the handler validates against the exported schema.json itself."""

    features: dict[str, float] = Field(description="Feature name -> numeric value")


class PredictResponse(BaseModel):
    prediction: int
    probabilities: dict[str, float]
    model_version: int
    dataset_version: str
    request_id: str
    warnings: list[str] = []


class ErrorResponse(BaseModel):
    detail: Any
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: int | None = None
    dataset_version: str | None = None
    mlflow_run_id: str | None = None
    uptime_seconds: float
    error: str | None = None


class StatsResponse(BaseModel):
    request_count: int
    error_count: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    class_distribution: dict[str, int]
    last_request_time: str | None
