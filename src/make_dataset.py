"""Deterministically generate a versioned wine dataset.

v1: the original 178 rows.
v2: v1 plus 120 seeded augmented rows (bootstrap resample + small multiplicative Gaussian noise).

The output is byte-identical for a given version, so its DVC md5 is reproducible on any machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from src.config import ROOT, load_config, path_for

SEED = 42
N_AUGMENTED = 120
NOISE_SCALE = 0.02
VERSIONS = ("v1", "v2")


def build_frame(version: str) -> pd.DataFrame:
    if version not in VERSIONS:
        raise ValueError(f"Unknown dataset version {version!r}; expected one of {VERSIONS}")
    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace("/", "_") for c in df.columns]
    if version == "v1":
        return df
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(df), size=N_AUGMENTED)
    aug = df.iloc[idx].reset_index(drop=True)
    features = aug.columns.drop("target")
    noise = rng.normal(0.0, NOISE_SCALE, size=(len(aug), len(features)))
    aug[features] = (aug[features].to_numpy() * (1.0 + noise)).round(4)
    return pd.concat([df, aug], ignore_index=True)


def csv_bytes(version: str) -> bytes:
    df = build_frame(version)
    text = df.to_csv(index=False, float_format="%.4f", lineterminator="\n")
    return text.encode("utf-8")


def write_dataset(version: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(csv_bytes(version))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or path_for(load_config(), "raw_data", ROOT)
    write_dataset(args.version, out)
    print(f"Wrote dataset {args.version} to {out}")


if __name__ == "__main__":
    main()
