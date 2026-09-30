"""MLflow Model Registry helpers: connection, aliases, stage tags, promotion."""

from __future__ import annotations

import urllib.request
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient


class PromotionError(Exception):
    """Raised when a promotion is refused."""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(uri: str) -> MlflowClient:
    """Point MLflow at `uri`, print the backend, and fail clearly if a server is unreachable."""
    print(f"MLflow tracking URI: {uri}")
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Cannot reach the MLflow server at {uri} ({exc}).\n"
                "Start it with `make up` (alias: `make mlflow`)."
            ) from exc
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def version_for_alias(client: MlflowClient, model: str, alias: str):
    try:
        return client.get_model_version_by_alias(model, alias)
    except mlflow.exceptions.MlflowException:
        return None


def macro_f1(client: MlflowClient, mv) -> float:
    """Macro F1 of a model version, taken from its training run."""
    return float(client.get_run(mv.run_id).data.metrics["f1_macro"])


def set_stage(client: MlflowClient, model: str, alias: str, version: str, stage: str) -> None:
    """Point `alias` at `version` and mirror it in the `stage` tag.

    A previous holder of the alias that is only tagged with this stage is demoted to "superseded"
    so tags never claim a state the alias no longer backs.
    """
    previous = version_for_alias(client, model, alias)
    if previous is not None and str(previous.version) != str(version):
        if previous.tags.get("stage") == stage:
            client.set_model_version_tag(model, previous.version, "stage", "superseded")
    client.set_registered_model_alias(model, alias, str(version))
    client.set_model_version_tag(model, str(version), "stage", stage)


def promote(client: MlflowClient, cfg: dict, force: bool = False) -> dict:
    """Move `production` to the current `staging` version, archiving the old production."""
    model = cfg["mlflow"]["registered_model"]
    staging_alias = cfg["mlflow"]["aliases"]["staging"]
    prod_alias = cfg["mlflow"]["aliases"]["production"]

    staged = version_for_alias(client, model, staging_alias)
    if staged is None:
        raise PromotionError(f"No version has the '{staging_alias}' alias. Run `make train` first.")
    current = version_for_alias(client, model, prod_alias)
    if current is not None and str(current.version) == str(staged.version):
        raise PromotionError(f"Version {staged.version} is already in production.")

    status = staged.tags.get("validation_status")
    if status != "passed" and not force:
        raise PromotionError(
            f"Version {staged.version} has validation_status={status!r}, not 'passed'. "
            "Run `make evaluate` (or use --force)."
        )
    if current is not None and not force:
        new_f1, old_f1 = macro_f1(client, staged), macro_f1(client, current)
        if new_f1 < old_f1:
            raise PromotionError(
                f"Version {staged.version} (macro F1 {new_f1:.4f}) is worse than production "
                f"version {current.version} ({old_f1:.4f}). Use --force to override."
            )

    ts = now_iso()
    archived = None
    if current is not None:
        client.set_model_version_tag(model, current.version, "stage", "archived")
        client.set_model_version_tag(model, current.version, "archived_at", ts)
        archived = str(current.version)
    set_stage(client, model, prod_alias, staged.version, "production")
    client.set_model_version_tag(model, staged.version, "promoted_at", ts)
    return {"promoted": str(staged.version), "archived": archived, "at": ts}
