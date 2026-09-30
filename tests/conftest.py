import hashlib
import json

import joblib
import pandas as pd
import pytest
import yaml

from src import preprocess
from src.config import load_config, load_params
from src.make_dataset import csv_bytes, write_dataset
from src.train import build_pipeline


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def params():
    p = load_params()
    p["model"]["random_forest"]["n_estimators"] = 10  # keep tests fast
    return p


@pytest.fixture
def project(tmp_path, cfg, params):
    """A throwaway project root with dataset v1, its DVC pointer, and preprocessed outputs."""
    raw = tmp_path / cfg["paths"]["raw_data"]
    write_dataset("v1", raw)
    md5 = hashlib.md5(csv_bytes("v1")).hexdigest()
    pointer = tmp_path / cfg["paths"]["raw_dvc"]
    pointer.write_text(
        yaml.safe_dump(
            {"outs": [{"md5": md5, "size": raw.stat().st_size, "hash": "md5", "path": "wine.csv"}]}
        )
    )
    preprocess.run(cfg, params, root=tmp_path)
    return tmp_path


@pytest.fixture
def mlflow_uri(tmp_path, monkeypatch):
    """File-based MLflow store (SQLite + local artifacts); no server, no network."""
    monkeypatch.chdir(tmp_path)  # default artifact root is ./mlruns
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def exported_model(project, cfg, params):
    """An exported model folder (as `make export` would write) built without MLflow."""
    processed = project / cfg["paths"]["processed_dir"]
    schema = json.loads((processed / "schema.json").read_text())
    train = pd.read_csv(processed / "train.csv")
    features = [f["name"] for f in schema["features"]]
    pipe = build_pipeline(params).fit(train[features], train[cfg["data"]["target"]])
    out = project / "models"
    (out / "v7").mkdir(parents=True)
    joblib.dump(pipe, out / "v7" / "model.pkl")
    (out / "v7" / "schema.json").write_text(json.dumps(schema))
    (out / "v7" / "metadata.json").write_text(
        json.dumps({"model_version": 7, "dataset_version": "v1", "run_id": "abc123"})
    )
    (out / "current.json").write_text(
        json.dumps(
            {
                "version": 7,
                "model": "v7/model.pkl",
                "metadata": "v7/metadata.json",
                "schema": "v7/schema.json",
            }
        )
    )
    return out
