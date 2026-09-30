#!/usr/bin/env bash
# Generate a dataset version, verify it against what git has committed, dvc add + push.
# Usage: bash scripts/data.sh v1|v2
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-}"
if [[ "$VERSION" != "v1" && "$VERSION" != "v2" ]]; then
  echo "Usage: make data VERSION=v1|v2" >&2
  exit 2
fi
TAG="data-$VERSION"
DVC_FILE="data/raw/wine.csv.dvc"

# md5 recorded in a .dvc pointer, read from a git revision ("" if unavailable)
pointer_md5() {
  git show "$1:$DVC_FILE" 2>/dev/null | python -c \
    'import sys,yaml; print(yaml.safe_load(sys.stdin)["outs"][0]["md5"])' 2>/dev/null || true
}

python -m src.make_dataset --version "$VERSION"
ACTUAL="$(python -m src.utils md5 data/raw/wine.csv)"

EXPECTED="$(pointer_md5 "$TAG")"
WHERE="$TAG"
if [[ -z "$EXPECTED" ]]; then
  # No tag: fall back to the committed pointer, but only if it is for the same version.
  HEAD_MD5="$(pointer_md5 HEAD)"
  if [[ -n "$HEAD_MD5" && "$(python -m src.utils detect "$HEAD_MD5")" == "$VERSION" ]]; then
    EXPECTED="$HEAD_MD5"
    WHERE="HEAD"
  fi
fi

if [[ -n "$EXPECTED" && "$EXPECTED" != "$ACTUAL" ]]; then
  echo "ERROR: generated dataset $VERSION has md5 $ACTUAL but $WHERE records $EXPECTED." >&2
  echo "       The generator or library versions changed; the dataset is not reproducible." >&2
  exit 1
fi

dvc add data/raw/wine.csv
dvc push data/raw/wine.csv.dvc || echo "WARNING: dvc push failed (is ./dvc-storage writable?)" >&2

if [[ -n "$EXPECTED" ]]; then
  echo "Dataset matches committed version $TAG (md5 $ACTUAL)"
else
  echo "New dataset version $VERSION (md5 $ACTUAL). To record it:"
  echo "  git add $DVC_FILE data/raw/.gitignore && git commit -m 'Add dataset $VERSION' && git tag $TAG"
fi
