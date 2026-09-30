#!/usr/bin/env bash
# End-to-end demo: data versions -> pipeline -> registry -> export -> API -> monitoring.
# Needs only Docker, make and git on the host. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

MLFLOW_URL="http://localhost:${MLFLOW_PORT:-5000}"
API_URL="http://localhost:${API_PORT:-8000}"
GOOD='{"features":{"alcohol":13.2,"malic_acid":2.77,"ash":2.51,"alcalinity_of_ash":18.5,"magnesium":98,"total_phenols":1.96,"flavanoids":0.5,"nonflavanoid_phenols":0.5,"proanthocyanins":1.4,"color_intensity":5.0,"hue":0.9,"od280_od315_of_diluted_wines":2.0,"proline":1000}}'

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
post() { printf '\n$ curl -X POST /predict  # %s\n' "$1"; shift
         curl -s -w '\n(HTTP %{http_code})\n' -X POST "$API_URL/predict" -H 'content-type: application/json' "$@"; }

# Commit the DVC pointer for a dataset version and tag it (only if the tag does not exist yet).
tag_dataset() {
  local v="$1"
  if git rev-parse -q --verify "refs/tags/data-$v" >/dev/null; then
    echo "Tag data-$v already exists; keeping it."
    return
  fi
  local ident=()
  [[ -n "$(git config user.email || true)" ]] || ident=(-c user.name=demo -c user.email=demo@example.com)
  git add data/raw/wine.csv.dvc data/raw/.gitignore
  git ${ident[@]+"${ident[@]}"} commit -q -m "Add dataset $v (DVC pointer)"
  git ${ident[@]+"${ident[@]}"} tag "data-$v"
  echo "Committed pointer and tagged data-$v"
}

step "1. Start MLflow"
make up

step "2. Dataset v1: generate, dvc add, dvc push, tag"
make data VERSION=v1
tag_dataset v1
make data-status

step "3. Preprocess (dvc repro preprocess)"
make preprocess

step "4. Train two runs on data v1 (logistic regression, then random forest)"
make train ARGS="--model logreg"
make train

step "5. Evaluate the staging model (quality gate)"
make evaluate
make report

step "6. Promote to production and export"
make promote
make export

step "7. Dataset v2: same code, different data"
make data VERSION=v2
tag_dataset v2
make data-diff || true
make preprocess
make train
make evaluate
make report

step "8. Promote v2 model if it is not worse, then export"
make promote || echo "(promotion refused, as designed: production stays as it was)"
make export

step "9. Serve the exported model"
make serve
printf '\n$ curl /health\n'; curl -s "$API_URL/health"; echo
post "good request" -d "$GOOD"
post "missing features" -d '{"features":{"alcohol":13.2}}'
post "wrong type" -d "${GOOD/\"alcohol\":13.2/\"alcohol\":\"high\"}"
post "malformed JSON" -d '{oops'
post "out-of-range value (accepted, with a warning)" -d "${GOOD/\"proline\":1000/\"proline\":99999}"

step "10. Monitoring: prediction log and /stats"
tail -n 5 logs/predictions.jsonl
printf '\n$ curl /stats\n'; curl -s "$API_URL/stats"; echo

step "11. Summary"
make report
echo
echo "MLflow UI : $MLFLOW_URL   (experiment 'wine-classifier', model registry)"
echo "API docs  : $API_URL/docs"
