import json

import pandas as pd
import pytest

from src import preprocess


def test_outputs_exist_and_schema_is_correct(project, cfg):
    out = project / cfg["paths"]["processed_dir"]
    for name in ("train.csv", "test.csv", "schema.json"):
        assert (out / name).exists()
    schema = json.loads((out / "schema.json").read_text())
    assert [f["name"] for f in schema["features"]] == cfg["data"]["features"]
    assert schema["classes"] == [0, 1, 2]
    alcohol = next(f for f in schema["features"] if f["name"] == "alcohol")
    assert alcohol["min"] < alcohol["max"]
    assert schema["n_rows"]["clean"] == 178


def test_split_is_stratified_and_reproducible(project, cfg, params):
    out = project / cfg["paths"]["processed_dir"]
    first = (out / "train.csv").read_bytes()
    preprocess.run(cfg, params, root=project)
    assert (out / "train.csv").read_bytes() == first
    train = pd.read_csv(out / "train.csv")
    test = pd.read_csv(out / "test.csv")
    assert len(test) == pytest.approx(0.2 * 178, abs=2)
    ratio_all = pd.concat([train, test])["target"].value_counts(normalize=True)
    ratio_test = test["target"].value_counts(normalize=True)
    for cls in ratio_all.index:
        assert ratio_test[cls] == pytest.approx(ratio_all[cls], abs=0.03)


def test_rejects_unexpected_columns(project, cfg, params):
    raw = project / cfg["paths"]["raw_data"]
    df = pd.read_csv(raw).drop(columns=["alcohol"])
    df.to_csv(raw, index=False)
    with pytest.raises(ValueError, match="alcohol"):
        preprocess.run(cfg, params, root=project)
