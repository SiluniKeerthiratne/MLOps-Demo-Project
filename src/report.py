"""Print registry versions and runs with their dataset lineage (used by the demo and README)."""

from __future__ import annotations

from mlflow import MlflowClient

from src import registry
from src.config import load_config, tracking_uri


def _table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))


def main() -> None:
    cfg = load_config()
    client: MlflowClient = registry.connect(tracking_uri())
    model = cfg["mlflow"]["registered_model"]

    print(f"\nRegistered model '{model}':")
    aliases: dict[str, list[str]] = {}
    for alias in cfg["mlflow"]["aliases"].values():
        mv = registry.version_for_alias(client, model, alias)
        if mv is not None:
            aliases.setdefault(str(mv.version), []).append(alias)
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{model}'"), key=lambda m: int(m.version)):
        t = mv.tags
        rows.append(
            [
                f"v{mv.version}",
                ",".join(aliases.get(str(mv.version), [])) or "-",
                t.get("stage", "-"),
                t.get("validation_status", "-"),
                t.get("dataset_version", "-"),
                t.get("dataset_dvc_md5", "-")[:12],
                t.get("test_macro_f1", "-"),
            ]
        )
    _table(["version", "aliases", "stage", "validation", "data", "data md5", "macro F1"], rows)

    exp = client.get_experiment_by_name(cfg["mlflow"]["experiment"])
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    print(f"\nRuns in experiment '{exp.name}':")
    rows = [
        [
            r.info.run_name,
            r.data.tags.get("dataset_version", "-"),
            r.data.tags.get("dataset_dvc_md5", "-")[:12],
            f"{r.data.metrics.get('f1_macro', float('nan')):.4f}",
            r.info.run_id[:8],
        ]
        for r in runs
        if r.info.status == "FINISHED"
    ]
    _table(["run", "data", "data md5", "macro F1", "run id"], rows)


if __name__ == "__main__":
    main()
