"""Loads config.yaml / params.yaml and resolves paths relative to the repo root."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(root: Path = ROOT) -> dict:
    return load_yaml(root / "config.yaml")


def load_params(root: Path = ROOT) -> dict:
    return load_yaml(root / "params.yaml")


def path_for(cfg: dict, key: str, root: Path = ROOT) -> Path:
    return root / cfg["paths"][key]


def tracking_uri() -> str:
    """Compose sets MLFLOW_TRACKING_URI=http://mlflow:5000; on the host we default to localhost."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
