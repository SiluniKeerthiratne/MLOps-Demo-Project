"""Loads the exported production model (models/current.json) without needing MLflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass
class LoadedModel:
    loaded: bool = False
    error: str | None = None
    pipeline: Any = None
    schema: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def version(self) -> int | None:
        return self.metadata.get("model_version")

    @property
    def dataset_version(self) -> str | None:
        return self.metadata.get("dataset_version")

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.schema.get("features", [])]


def load_model(models_dir: Path) -> LoadedModel:
    """Never raises: a failure yields LoadedModel(loaded=False, error=...)."""
    try:
        current = json.loads((models_dir / "current.json").read_text())
        schema = json.loads((models_dir / current["schema"]).read_text())
        metadata = json.loads((models_dir / current["metadata"]).read_text())
        pipeline = joblib.load(models_dir / current["model"])
        return LoadedModel(True, None, pipeline, schema, metadata)
    except Exception as exc:  # noqa: BLE001 - service must start even if the model is broken
        return LoadedModel(loaded=False, error=f"{type(exc).__name__}: {exc}")
