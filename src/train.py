"""Train a wine classifier, log everything to MLflow, register it and set the `staging` alias."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import registry  # noqa: E402
from src.config import ROOT, load_config, load_params, path_for, tracking_uri  # noqa: E402
from src.utils import (  # noqa: E402
    detect_dataset_version,
    git_sha,
    md5_file,
    read_dvc_md5,
    sha256_file,
)


def build_pipeline(params: dict) -> Pipeline:
    kind = params["model"]["type"]
    seed = params["seed"]
    if kind == "random_forest":
        clf = RandomForestClassifier(
            random_state=seed, n_jobs=1, **params["model"]["random_forest"]
        )
    elif kind == "logreg":
        clf = LogisticRegression(random_state=seed, **params["model"]["logreg"])
    else:
        raise ValueError(f"Unknown model type {kind!r}; use random_forest or logreg")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def apply_overrides(params: dict, args: argparse.Namespace) -> dict:
    params = json.loads(json.dumps(params))  # deep copy
    if args.model:
        params["model"]["type"] = {"rf": "random_forest"}.get(args.model, args.model)
    if args.n_estimators is not None:
        params["model"]["random_forest"]["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["model"]["random_forest"]["max_depth"] = args.max_depth
    if args.C is not None:
        params["model"]["logreg"]["C"] = args.C
    if args.seed is not None:
        params["seed"] = args.seed
    return params


def run_name(params: dict, dataset_version: str) -> str:
    seed = params["seed"]
    if params["model"]["type"] == "random_forest":
        n = params["model"]["random_forest"]["n_estimators"]
        return f"rf-n{n}-seed{seed}-data-{dataset_version}"
    c = params["model"]["logreg"]["C"]
    return f"logreg-C{c}-seed{seed}-data-{dataset_version}"


def flat_params(params: dict) -> dict:
    kind = params["model"]["type"]
    out = {"model_type": kind, "seed": params["seed"]}
    out.update({f"{kind}.{k}": v for k, v in params["model"][kind].items()})
    return out


def run_training(cfg: dict, params: dict, root: Path = ROOT, uri: str | None = None) -> dict:
    client = registry.connect(uri or tracking_uri())
    processed = path_for(cfg, "processed_dir", root)
    target = cfg["data"]["target"]
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    schema_path = processed / "schema.json"
    schema = json.loads(schema_path.read_text())
    features = [f["name"] for f in schema["features"]]
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Dataset lineage: the md5 comes from the DVC pointer, which is what git versions.
    raw_csv = path_for(cfg, "raw_data", root)
    dvc_md5 = read_dvc_md5(path_for(cfg, "raw_dvc", root))
    md5_source = "dvc_pointer"
    if dvc_md5 is None:
        print("WARNING: no .dvc pointer found; hashing the raw CSV directly.")
        dvc_md5, md5_source = md5_file(raw_csv), "computed"
    dataset_version = detect_dataset_version(dvc_md5)

    pipe = build_pipeline(params)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    labels = schema["classes"]
    prec, rec, _, _ = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "train_accuracy": float(accuracy_score(y_train, pipe.predict(X_train))),
    }
    for label, p, r in zip(labels, prec, rec, strict=True):
        metrics[f"precision_class_{label}"] = float(p)
        metrics[f"recall_class_{label}"] = float(r)

    model_name = cfg["mlflow"]["registered_model"]
    experiment = mlflow.set_experiment(cfg["mlflow"]["experiment"])
    name = run_name(params, dataset_version)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(flat_params(params))
        mlflow.log_params(
            {"test_size": schema["split"]["test_size"], "split_seed": schema["split"]["seed"]}
        )
        mlflow.set_tags(
            {
                "dataset_version": dataset_version,
                "dataset_dvc_md5": dvc_md5,
                "dataset_md5_source": md5_source,
                "dataset_rows": str(schema["n_rows"]["raw"]),
                "train_csv_sha256": sha256_file(processed / "train.csv"),
                "git_commit": git_sha(root),
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "mlflow_version": mlflow.__version__,
            }
        )
        dataset = mlflow.data.from_pandas(
            train_df,
            source=str(raw_csv),
            name=f"wine-{dataset_version}",
            targets=target,
            digest=dvc_md5[:8],
        )
        mlflow.log_input(dataset, context="training", tags={"dvc_md5": dvc_md5})
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=labels, ax=ax)
        ax.set_title(f"Confusion matrix ({name})")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        mlflow.log_artifact(str(schema_path))

        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X_train, pipe.predict_proba(X_train)),
            input_example=X_train.head(3),
            registered_model_name=model_name,
            # pickle-based (model.pkl); MLflow >=3.x defaults to skops, which rejects tree models
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        version = str(info.registered_model_version)
        client.set_model_version_tag(model_name, version, "dataset_version", dataset_version)
        client.set_model_version_tag(model_name, version, "dataset_dvc_md5", dvc_md5)
        registry.set_stage(
            client, model_name, cfg["mlflow"]["aliases"]["staging"], version, "staging"
        )

    metrics_path = path_for(cfg, "metrics", root)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Run '{name}' ({run.info.run_id}) -> model '{model_name}' v{version} [staging]")
    print(f"Dataset {dataset_version} (md5 {dvc_md5}); macro F1 = {metrics['f1_macro']:.4f}")
    return {
        "run_id": run.info.run_id,
        "experiment_id": experiment.experiment_id,
        "version": version,
        "dataset_version": dataset_version,
        "dvc_md5": dvc_md5,
        "metrics": metrics,
        "run_name": name,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["random_forest", "rf", "logreg"])
    p.add_argument("--n-estimators", type=int)
    p.add_argument("--max-depth", type=int)
    p.add_argument("--C", type=float)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    params = apply_overrides(load_params(), args)
    run_training(load_config(), params)


if __name__ == "__main__":
    main()
