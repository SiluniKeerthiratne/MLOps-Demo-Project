"""Export the `production` model to models/v<N>/ so the API needs no MLflow at runtime."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import mlflow

from src import registry
from src.config import ROOT, load_config, path_for, tracking_uri


def export_production(cfg: dict, root: Path = ROOT, uri: str | None = None) -> Path:
    client = registry.connect(uri or tracking_uri())
    model = cfg["mlflow"]["registered_model"]
    alias = cfg["mlflow"]["aliases"]["production"]
    mv = registry.version_for_alias(client, model, alias)
    if mv is None:
        raise SystemExit(f"No version has alias '{alias}'. Run `make promote` first.")

    run = client.get_run(mv.run_id)
    out_dir = path_for(cfg, "models_dir", root) / f"v{mv.version}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pipe = mlflow.sklearn.load_model(f"models:/{model}@{alias}")
    joblib.dump(pipe, out_dir / "model.pkl")
    schema_src = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="schema.json", dst_path=str(out_dir / ".dl")
    )
    shutil.copy(schema_src, out_dir / "schema.json")
    shutil.rmtree(out_dir / ".dl")

    tags = run.data.tags
    metadata = {
        "model_name": model,
        "model_version": int(mv.version),
        "run_id": mv.run_id,
        "git_commit": tags.get("git_commit", "unknown"),
        "dataset_version": tags.get("dataset_version", "unknown"),
        "dataset_dvc_md5": tags.get("dataset_dvc_md5", "unknown"),
        "sklearn_version": tags.get("sklearn_version", "unknown"),
        "metrics": run.data.metrics,
        "params": run.data.params,
        "exported_at": registry.now_iso(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    current = {
        "version": int(mv.version),
        "model": f"v{mv.version}/model.pkl",
        "metadata": f"v{mv.version}/metadata.json",
        "schema": f"v{mv.version}/schema.json",
    }
    (out_dir.parent / "current.json").write_text(json.dumps(current, indent=2) + "\n")
    print(f"Exported production version {mv.version} to {out_dir}")
    print(f"models/current.json -> v{mv.version} (dataset {metadata['dataset_version']})")
    return out_dir


def main() -> None:
    export_production(load_config())


if __name__ == "__main__":
    main()
