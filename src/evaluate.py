"""Quality gate: evaluate the `staging` model on test.csv and record validation_status."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src import registry
from src.config import ROOT, load_config, path_for, tracking_uri


def evaluate(
    cfg: dict, alias: str | None = None, root: Path = ROOT, uri: str | None = None
) -> bool:
    client = registry.connect(uri or tracking_uri())
    model = cfg["mlflow"]["registered_model"]
    alias = alias or cfg["mlflow"]["aliases"]["staging"]
    mv = registry.version_for_alias(client, model, alias)
    if mv is None:
        raise SystemExit(f"No model version has alias '{alias}'. Run `make train` first.")

    processed = path_for(cfg, "processed_dir", root)
    target = cfg["data"]["target"]
    test_df = pd.read_csv(processed / "test.csv")
    pipe = mlflow.sklearn.load_model(f"models:/{model}@{alias}")
    features = list(pipe.feature_names_in_)
    pred = pipe.predict(test_df[features])

    f1 = float(f1_score(test_df[target], pred, average="macro"))
    threshold = cfg["quality_gate"]["min_macro_f1"]
    passed = f1 >= threshold
    status = "passed" if passed else "failed"

    print(f"Evaluating '{model}' v{mv.version} (alias '{alias}') on {len(test_df)} test rows\n")
    print(classification_report(test_df[target], pred, digits=3))
    client.set_model_version_tag(model, mv.version, "validation_status", status)
    client.set_model_version_tag(model, mv.version, "test_macro_f1", f"{f1:.4f}")
    print(f"macro F1 = {f1:.4f} (threshold {threshold}) -> validation_status={status}")
    return passed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alias", default=None)
    args = p.parse_args()
    sys.exit(0 if evaluate(load_config(), args.alias) else 1)


if __name__ == "__main__":
    main()
