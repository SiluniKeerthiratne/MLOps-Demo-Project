#!/usr/bin/env bash
# Try `dvc pull`; on a fresh clone the local remote is empty, so regenerate the dataset instead.
set -uo pipefail
cd "$(dirname "$0")/.."

if dvc pull 2>/dev/null && [[ -f data/raw/wine.csv ]]; then
  echo "Dataset restored with dvc pull."
  exit 0
fi

echo "dvc pull found nothing (expected on a fresh clone: ./dvc-storage is not in git)."
EXPECTED="$(python -c 'from src.utils import read_dvc_md5; from pathlib import Path; print(read_dvc_md5(Path("data/raw/wine.csv.dvc")) or "")')"
if [[ -z "$EXPECTED" ]]; then
  echo "ERROR: data/raw/wine.csv.dvc not found. Run 'make data VERSION=v1' first." >&2
  exit 1
fi
VERSION="$(python -m src.utils detect "$EXPECTED")"
if [[ "$VERSION" == "unknown" ]]; then
  echo "ERROR: pointer md5 $EXPECTED matches no known dataset version." >&2
  exit 1
fi
echo "Regenerating dataset $VERSION deterministically..."
python -m src.make_dataset --version "$VERSION"
ACTUAL="$(python -m src.utils md5 data/raw/wine.csv)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "ERROR: regenerated md5 $ACTUAL != pointer md5 $EXPECTED" >&2
  exit 1
fi
dvc commit -f data/raw/wine.csv.dvc >/dev/null 2>&1 || true
echo "Dataset $VERSION regenerated; matches committed md5 $EXPECTED."
