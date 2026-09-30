import json
import math

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def env(exported_model, tmp_path):
    log = tmp_path / "logs" / "predictions.jsonl"
    with TestClient(create_app(exported_model, log)) as client:
        yield client, log


@pytest.fixture
def good(exported_model):
    schema = json.loads((exported_model / "v7" / "schema.json").read_text())
    return {f["name"]: (f["min"] + f["max"]) / 2 for f in schema["features"]}


def lines(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_valid_request(env, good):
    client, log = env
    r = client.post("/predict", json={"features": good})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1, 2)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["model_version"] == 7 and body["dataset_version"] == "v1"
    assert body["request_id"] and body["warnings"] == []
    (entry,) = lines(log)
    assert entry["status"] == "ok" and entry["request_id"] == body["request_id"]
    assert entry["input"] == good and entry["confidence"] > 0 and entry["latency_ms"] >= 0


def test_missing_and_extra_features(env, good):
    client, _ = env
    missing = dict(good)
    missing.pop("alcohol")
    r = client.post("/predict", json={"features": missing})
    assert r.status_code == 422 and r.json()["detail"]["missing_features"] == ["alcohol"]
    r = client.post("/predict", json={"features": {**good, "bogus": 1}})
    assert r.status_code == 422 and r.json()["detail"]["unknown_features"] == ["bogus"]


@pytest.mark.parametrize("bad", ["high", None, True, [1]])
def test_wrong_type(env, good, bad):
    client, _ = env
    r = client.post("/predict", json={"features": {**good, "alcohol": bad}})
    assert r.status_code == 422 and "alcohol" in r.json()["detail"]["invalid_values"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_inf(env, good, token):
    client, _ = env
    payload = json.dumps({"features": good}).replace(
        f'"alcohol": {good["alcohol"]}', f'"alcohol": {token}'
    )
    assert token in payload
    r = client.post("/predict", content=payload, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "finite" in r.json()["detail"]["invalid_values"]["alcohol"]
    assert not math.isfinite(float(token.replace("Infinity", "inf")))


def test_empty_body_and_invalid_json(env):
    client, log = env
    assert client.post("/predict").status_code == 422
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert client.post("/predict", json=[1, 2]).status_code == 422
    assert client.post("/predict", json={"nope": 1}).status_code == 422
    assert len(lines(log)) == 4


def test_out_of_range_warns_but_succeeds(env, good):
    client, log = env
    r = client.post("/predict", json={"features": {**good, "proline": 99999}})
    assert r.status_code == 200
    assert any("proline" in w for w in r.json()["warnings"])
    assert any("proline" in w for w in lines(log)[0]["warnings"])


def test_health_and_stats_and_one_log_line_per_request(env, good):
    client, log = env
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok" and h.json()["model_version"] == 7
    assert h.json()["mlflow_run_id"] == "abc123" and h.json()["uptime_seconds"] >= 0
    assert client.get("/stats").json()["request_count"] == 0

    client.post("/predict", json={"features": good})
    client.post("/predict", json={"features": good})
    client.post("/predict", json={"features": {}})
    assert len(lines(log)) == 3

    s = client.get("/stats").json()
    assert s["request_count"] == 3 and s["error_count"] == 1
    assert sum(s["class_distribution"].values()) == 2
    assert s["avg_latency_ms"] > 0 and s["p95_latency_ms"] > 0 and s["last_request_time"]


def test_model_load_failure_keeps_service_up(tmp_path, good):
    log = tmp_path / "predictions.jsonl"
    with TestClient(create_app(tmp_path / "missing-models", log)) as client:
        h = client.get("/health")
        assert h.status_code == 503
        assert h.json()["status"] == "unhealthy" and h.json()["model_loaded"] is False
        r = client.post("/predict", json={"features": good})
        assert r.status_code == 503
        assert len(lines(log)) == 1 and lines(log)[0]["status"] == "error"
