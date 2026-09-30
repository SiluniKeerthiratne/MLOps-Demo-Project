"""Validate raw data, clean it, split (stratified), and write train/test CSVs plus schema.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config, load_params, path_for


def run(cfg: dict, params: dict, root: Path = ROOT) -> dict:
    raw = path_for(cfg, "raw_data", root)
    out_dir = path_for(cfg, "processed_dir", root)
    target = cfg["data"]["target"]
    features = cfg["data"]["features"]

    df = pd.read_csv(raw)
    expected = set(features) | {target}
    if set(df.columns) != expected:
        missing = sorted(expected - set(df.columns))
        extra = sorted(set(df.columns) - expected)
        raise ValueError(f"Raw data columns mismatch. Missing: {missing}. Unexpected: {extra}")
    df = df[[*features, target]]

    n_raw = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df[target] = df[target].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=df[target],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False, float_format="%.4f", lineterminator="\n")
    test_df.to_csv(out_dir / "test.csv", index=False, float_format="%.4f", lineterminator="\n")

    schema = {
        "features": [
            {
                "name": f,
                "dtype": "float",
                "min": float(df[f].min()),
                "max": float(df[f].max()),
            }
            for f in features
        ],
        "target": target,
        "classes": sorted(int(c) for c in df[target].unique()),
        "split": {"seed": params["seed"], "test_size": params["test_size"]},
        "n_rows": {"raw": n_raw, "clean": len(df), "train": len(train_df), "test": len(test_df)},
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(
        f"Preprocessed {n_raw} raw rows -> {len(df)} clean "
        f"(train={len(train_df)}, test={len(test_df)}) in {out_dir}"
    )
    return schema


def main() -> None:
    run(load_config(), load_params())


if __name__ == "__main__":
    main()
