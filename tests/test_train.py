import json

import pytest
from mlflow import MlflowClient

from src.train import run_name, run_training


def test_train_logs_everything(project, cfg, params, mlflow_uri):
    res = run_training(cfg, params, root=project, uri=mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    run = client.get_run(res["run_id"])

    assert run.data.metrics["accuracy"] > 0.8
    assert "f1_macro" in run.data.metrics
    assert "precision_class_0" in run.data.metrics and "recall_class_2" in run.data.metrics
    assert run.data.tags["dataset_version"] == "v1"
    assert run.data.tags["dataset_dvc_md5"] == res["dvc_md5"]
    assert run.data.tags["train_csv_sha256"]
    assert run.data.tags["git_commit"]
    assert run.data.params["seed"] == "42"
    assert run.info.run_name == "rf-n10-seed42-data-v1"
    artifacts = {a.path for a in client.list_artifacts(res["run_id"])}
    assert {"confusion_matrix.png", "schema.json"} <= artifacts

    mv = client.get_model_version_by_alias(cfg["mlflow"]["registered_model"], "staging")
    assert str(mv.version) == res["version"]
    assert mv.tags["stage"] == "staging"
    assert mv.tags["dataset_version"] == "v1"
    assert json.loads((project / cfg["paths"]["metrics"]).read_text())["f1_macro"] > 0.8


def test_same_seed_gives_identical_metrics(project, cfg, params, mlflow_uri):
    a = run_training(cfg, params, root=project, uri=mlflow_uri)["metrics"]
    b = run_training(cfg, params, root=project, uri=mlflow_uri)["metrics"]
    assert a == b


def test_second_version_moves_staging_alias(project, cfg, params, mlflow_uri):
    first = run_training(cfg, params, root=project, uri=mlflow_uri)
    second = run_training(cfg, params, root=project, uri=mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    name = cfg["mlflow"]["registered_model"]
    assert str(client.get_model_version_by_alias(name, "staging").version) == second["version"]
    assert client.get_model_version(name, first["version"]).tags["stage"] == "superseded"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("random_forest", "rf-n10-seed42-data-v2"), ("logreg", "logreg-C1.0-seed42-data-v2")],
)
def test_run_name(params, model, expected):
    params["model"]["type"] = model
    assert run_name(params, "v2") == expected
