import pytest
from mlflow import MlflowClient

from src import registry
from src.evaluate import evaluate
from src.train import run_training


@pytest.fixture
def setup(project, cfg, params, mlflow_uri):
    client = MlflowClient(tracking_uri=mlflow_uri)
    name = cfg["mlflow"]["registered_model"]

    def train_and_evaluate(validate=True):
        res = run_training(cfg, params, root=project, uri=mlflow_uri)
        if validate:
            assert evaluate(cfg, root=project, uri=mlflow_uri)
        return res

    return client, name, train_and_evaluate


def test_promote_sets_production_alias_and_tag(cfg, setup):
    client, name, train = setup
    res = train()
    out = registry.promote(client, cfg)
    assert out["promoted"] == res["version"] and out["archived"] is None
    mv = client.get_model_version_by_alias(name, "production")
    assert str(mv.version) == res["version"]
    assert mv.tags["stage"] == "production"
    assert mv.tags["promoted_at"]


def test_promote_archives_previous_production(cfg, setup):
    client, name, train = setup
    v1 = train()
    registry.promote(client, cfg)
    v2 = train()
    out = registry.promote(client, cfg)
    assert out["archived"] == v1["version"]
    old = client.get_model_version(name, v1["version"])
    assert old.tags["stage"] == "archived" and old.tags["archived_at"]
    assert str(client.get_model_version_by_alias(name, "production").version) == v2["version"]


def test_promote_refuses_unvalidated_model(cfg, setup):
    client, _, train = setup
    train(validate=False)
    with pytest.raises(registry.PromotionError, match="validation_status"):
        registry.promote(client, cfg)
    assert registry.promote(client, cfg, force=True)["promoted"]


def test_promote_refuses_failed_gate(cfg, setup, project, mlflow_uri):
    client, _, train = setup
    train(validate=False)
    strict = {**cfg, "quality_gate": {"min_macro_f1": 1.01}}
    assert evaluate(strict, root=project, uri=mlflow_uri) is False
    with pytest.raises(registry.PromotionError, match="failed"):
        registry.promote(client, cfg)


def test_promote_refuses_worse_model(cfg, setup):
    client, name, train = setup
    train()
    registry.promote(client, cfg)
    v2 = train()
    client.log_metric(v2["run_id"], "f1_macro", 0.5, step=99)  # make the candidate worse
    with pytest.raises(registry.PromotionError, match="worse"):
        registry.promote(client, cfg)
    assert registry.promote(client, cfg, force=True)["promoted"] == v2["version"]
