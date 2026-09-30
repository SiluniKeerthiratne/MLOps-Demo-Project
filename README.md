# MLOps Demo: from data to a deployed, monitored service

A small but complete ML project that shows every lifecycle stage in one repo: **versioned data**
(DVC) → **pipeline** → **experiment tracking and model registry** (MLflow) → **packaged API**
(FastAPI in Docker) → **monitoring** (structured prediction logs, `/health`, `/stats`).
Everything runs locally in Docker. No cloud accounts, tokens, or logins are needed.

The model is deliberately trivial (scikit-learn's Wine dataset, a random forest). The value is the
structure: each stage has one obvious place in the repo and runs with one command.

**Architecture in one paragraph.** `src/make_dataset.py` deterministically generates dataset `v1`
or `v2`; DVC tracks it (pointer file in git, bytes in a local remote). `dvc.yaml` preprocesses it
and trains a model; `src/train.py` logs params, metrics, dataset lineage and the model to an MLflow
server (Docker) and registers it with the `staging` alias. `src/evaluate.py` is a quality gate,
`src/promote.py` moves the `production` alias, and `src/export.py` writes the production model to
`models/`. The API container serves that folder (no MLflow at runtime) and appends every request to
`logs/predictions.jsonl`, which `/stats` summarises. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Prerequisites

- Docker with Compose v2 (`docker compose version`) and `make`. On Windows use WSL2.
- `git` (for the demo's commits and tags). Nothing else: Python, Poetry and DVC live in the images.

## Quick start

```bash
make up                  # start MLflow on http://localhost:5000
make data VERSION=v1     # generate dataset v1, dvc add, dvc push (local remote)
make preprocess          # dvc repro preprocess
make train               # train, log to MLflow, register, alias "staging"
make evaluate            # quality gate on the test split
make promote             # staging -> production
make export              # production model -> models/
make serve               # API on http://localhost:8000
```

Or everything, narrated, in one go (good for a screen recording): `make demo`.

Raw equivalents (what the Makefile runs):

| Make target | Raw command |
|---|---|
| `make up` | `docker compose up -d --build --wait mlflow` |
| `make data VERSION=v1` | `docker compose run --rm trainer bash scripts/data.sh v1` |
| `make preprocess` | `docker compose run --rm trainer dvc repro preprocess` |
| `make train ARGS="--model logreg"` | `docker compose run --rm trainer python -m src.train --model logreg` |
| `make evaluate` / `promote` / `export` | `docker compose run --rm trainer python -m src.evaluate` (`src.promote`, `src.export`) |
| `make serve` | `docker compose up -d --build --force-recreate --wait api` |
| `make test` / `make lint` | `docker compose run --rm tester pytest -q` / `ruff check .` |
| `make down` | `docker compose --profile tools down` |

Training accepts overrides: `--model logreg|rf`, `--n-estimators 200`, `--max-depth 5`, `--C 0.5`,
`--seed 7`. Defaults live in `params.yaml`. `make report` prints registry versions and runs.

## Walkthrough

### 1. Two dataset versions

```bash
make data VERSION=v1     # New dataset version v1 (md5 87a206c70a60bb7431293486bcd185bd)
git add data/raw/wine.csv.dvc data/raw/.gitignore && git commit -m "dataset v1" && git tag data-v1
make data VERSION=v2     # New dataset version v2 (md5 26381881ce0db229f49a8f6c6142e80a)
git add data/raw/wine.csv.dvc && git commit -m "dataset v2" && git tag data-v2
make data-diff           # Modified: data/raw/wine.csv
```

`v1` is the original 178 rows; `v2` adds 120 seeded augmented rows (bootstrap resample plus 2%
Gaussian noise). Generation is byte-deterministic, so the md5 above is the same on every machine.
The CSV is never committed: git holds `wine.csv.dvc` (a tiny pointer with the md5), DVC holds the
bytes in `./dvc-storage`. `make data-status` shows `dvc status` and the current hash;
`make data-checkout TAG=data-v1` switches the working dataset back to v1.

### 2. Several runs

```bash
make preprocess
make train ARGS="--model logreg"   # logreg-C1.0-seed42-data-v1   macro F1 0.9710
make train                         # rf-n100-seed42-data-v1       macro F1 1.0000
make evaluate                      # macro F1 = 1.0000 (threshold 0.9) -> validation_status=passed
```

Open http://localhost:5000. Each run shows params, metrics, a confusion matrix, the model with its
signature, `schema.json`, and the lineage tags `dataset_version`, `dataset_dvc_md5`,
`train_csv_sha256`, `git_commit` and library versions. The training dataset is also attached through
`mlflow.data`.

### 3. Staging versus production

```bash
make promote && make export
make data VERSION=v2 && make preprocess && make train && make evaluate
make report
```

```
version  aliases             stage       validation  data  data md5      macro F1
v2       -                   archived    passed      v1    87a206c70a60  1.0000
v3       staging,production  production  passed      v2    26381881ce0d  1.0000

run                         data  data md5      macro F1
logreg-C1.0-seed42-data-v1  v1    87a206c70a60  0.9710
rf-n100-seed42-data-v1      v1    87a206c70a60  1.0000
rf-n100-seed42-data-v2      v2    26381881ce0d  1.0000
```

Same code, same seed, different data hash, different model. `make promote` refuses a model whose
`validation_status` is not `passed`, or whose macro F1 is lower than production's (override with
`make promote ARGS=--force`). The replaced production version is tagged `stage=archived` with an
`archived_at` timestamp. (Note: `v2` rows are bootstrap copies of `v1` rows, so its test split is
not independent. The metrics are for demonstrating lineage, not for claiming accuracy.)

### 4. Serve, and try good and bad requests

```bash
make serve
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"model_version":3,"dataset_version":"v2","mlflow_run_id":"...","uptime_seconds":1.7,"error":null}

curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"features":{
  "alcohol":13.2,"malic_acid":2.77,"ash":2.51,"alcalinity_of_ash":18.5,"magnesium":98,
  "total_phenols":1.96,"flavanoids":0.5,"nonflavanoid_phenols":0.5,"proanthocyanins":1.4,
  "color_intensity":5.0,"hue":0.9,"od280_od315_of_diluted_wines":2.0,"proline":1000}}'
# {"prediction":2,"probabilities":{"0":0.22,"1":0.05,"2":0.73},"model_version":3,"dataset_version":"v2","request_id":"...","warnings":[]}

curl -s -X POST localhost:8000/predict -d '{"features":{"alcohol":13.2}}'   # 422, lists missing features
curl -s -X POST localhost:8000/predict -d '{oops'                            # 400, malformed JSON
```

| Input | Response |
|---|---|
| valid | 200 with prediction, probabilities, model and dataset version, `request_id` |
| missing / unknown feature, wrong type, NaN/inf, empty body, wrong shape | 422 with a precise message |
| malformed JSON | 400 |
| value outside the training range | 200, with a `warnings` entry (also logged) |
| model failed to load | service still starts; `/health` returns 503 `unhealthy`; `/predict` returns 503 |

### 5. Logs and `/stats`

Every `/predict` request, successful or not, appends one line to `logs/predictions.jsonl`
(`ts, request_id, model_version, dataset_version, latency_ms, status, http_status, input,
prediction, confidence, warnings`, plus `error` on failures).

```bash
tail -n 3 logs/predictions.jsonl
curl -s localhost:8000/stats
# {"request_count":5,"error_count":3,"avg_latency_ms":2.989,"p95_latency_ms":11.762,
#  "class_distribution":{"2":2},"last_request_time":"2026-09-30T19:55:12.472+00:00"}
```

The log stores raw inputs to keep the demo transparent. Real systems should avoid logging
sensitive input, or hash/redact it.

## Layout and lifecycle stages

| Path | Lifecycle stage |
|---|---|
| `src/make_dataset.py`, `data/raw/*.dvc`, `.dvc/`, `dvc-storage/` | Data versioning |
| `src/preprocess.py`, `dvc.yaml`, `data/processed/` | Validation, cleaning, split, `schema.json` |
| `src/train.py`, `params.yaml`, `config.yaml` | Training and experiment tracking |
| `src/evaluate.py`, `src/registry.py`, `src/promote.py` | Quality gate, staging, promotion |
| `src/export.py`, `models/` | Hand-off from registry to serving |
| `api/`, `Dockerfile` (`api` target) | Serving |
| `logs/predictions.jsonl`, `/health`, `/stats` | Monitoring |
| `docker-compose.yml`, `docker/mlflow.Dockerfile` | Local infrastructure |
| `tests/`, `.github/workflows/ci.yml` | Verification |
| `scripts/demo.sh`, `Makefile` | Automation |

## Design decisions

- **DVC with a local remote, plus deterministic generation.** No cloud account is needed, yet the
  git pointer still identifies the exact bytes. Because `./dvc-storage` is git-ignored, a fresh clone
  has nothing to `dvc pull`; `make data VERSION=v1` regenerates the file and checks that its md5
  equals the committed pointer (`Dataset matches committed version data-v1 (md5 ...)`). `make pull`
  tries `dvc pull` first and falls back to regeneration.
- **The dataset version is derived, not stored.** `train.py` reads the md5 from the `.dvc` pointer
  and maps it to `v1`/`v2` by regenerating both in memory. There is no separate version file that
  could drift from the data.
- **MLflow aliases, not stages.** Classic stages are deprecated. `staging` and `production` aliases
  are mirrored in a `stage` tag (`staging`, `production`, `archived`, `superseded`) so the UI shows
  state at a glance.
- **Export to `models/`.** The API image then needs only scikit-learn and FastAPI, not MLflow, and
  a deployed model is a folder you can inspect: `model.pkl`, `metadata.json` (version, run ID, git
  SHA, dataset version and md5, metrics, params), `schema.json`.
- **Pickle serialization.** MLflow 3.x defaults to `skops`, which rejects tree models, so
  `train.py` asks for cloudpickle explicitly. `export.py` re-saves with joblib. Both images install
  the same scikit-learn from the same `poetry.lock`.
- **JSONL logs.** One self-describing line per request; greppable, appendable, and enough to compute
  `/stats` with the standard library.
- **Extra `tester` compose service.** The `trainer` image has only `main,train,data` groups as
  specified, so `make test`/`make lint` use a `dev` image target that adds the API and dev groups.
- **MLflow `--allowed-hosts`.** Recent MLflow answers 403 to unknown `Host` headers. The trainer
  calls `http://mlflow:5000`, so that name is explicitly allowed (plus `localhost:*`).
- **Left out on purpose:** Kubernetes, Airflow/Kubeflow, Prometheus, cloud SDKs, drift detection,
  authentication on the API, and any secret. The Docker images are never pushed anywhere.

## Troubleshooting

- **Port already in use (5000 or 8000).** On macOS, AirPlay Receiver uses 5000. Copy `.env.example`
  to `.env` and set `MLFLOW_PORT=5001` / `API_PORT=8001`. Note that `MLFLOW_TRACKING_URI` inside
  containers stays `http://mlflow:5000`.
- **Docker permission denied (Linux).** Add your user to the `docker` group or use `sudo`. Files the
  trainer creates in the mounted repo are owned by root; fix with `sudo chown -R $USER .`. If the
  API cannot write `logs/`, run `chmod 777 logs` (the container user has uid 1000).
- **MLflow 403 "Invalid Host header" or "Cannot reach the MLflow server".** Start it with
  `make up`; check `docker compose logs mlflow`. If you rename the compose service, update
  `--allowed-hosts` in `docker/mlflow.Dockerfile`.
- **DVC or git "dubious ownership".** The trainer image runs
  `git config --global --add safe.directory /app`. If you run git/DVC yourself in another container
  with a mounted repo, do the same.
- **`make data` says the hash does not match.** The generator or a library (numpy, pandas,
  scikit-learn) changed. Use the versions in `poetry.lock`.
- **Windows line endings.** `.gitattributes` forces LF for scripts. If you see `bash\r` errors, run
  `git config core.autocrlf false`, re-clone, and use WSL2.
- **Poetry lock mismatch.** After editing `pyproject.toml` run `poetry lock` and commit
  `poetry.lock`. `poetry check --lock` (also run in CI) verifies they agree.
- **API container is `unhealthy`.** `models/current.json` is missing: run `make promote export`
  before `make serve`. `curl localhost:8000/health` shows the load error.
- **Stale `dvc status`.** `make train` (plain Python) does not update `dvc.lock`; `make repro` runs
  the full DVC pipeline (preprocess and train) and leaves `dvc status` clean.

## Reproducible, deployable, observable

- **Reproducible:** pinned dependencies (`poetry.lock`), versioned and hash-verified data, seeded
  splits and models, parameters in `params.yaml`, and every run tagged with dataset version and md5,
  train-set SHA-256 and git commit. The same seed and dataset version give identical metrics
  (tested).
- **Deployable:** one multi-stage Dockerfile, a non-root API image with a `HEALTHCHECK` and no
  training or dev dependencies, a file-based hand-off (`models/`), and CI that builds and smoke-tests
  the image.
- **Observable:** every prediction is logged with model and dataset version, latency and warnings;
  `/health` and `/stats` summarise the service; every model links back to its MLflow run.

## Running without Docker (optional)

Needs Python 3.11 and Poetry. The Makefile's `RUN`/`DEVRUN` variables swap the Docker wrapper for
host Python:

```bash
poetry install --no-root
poetry run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow-local/mlflow.db \
  --artifacts-destination ./mlflow-local/artifacts --serve-artifacts &
export MLFLOW_TRACKING_URI=http://localhost:5000
make RUN="poetry run" DEVRUN="poetry run" data VERSION=v1
make RUN="poetry run" preprocess train evaluate promote export
```

CI does the same against a SQLite MLflow store without a server
(`MLFLOW_TRACKING_URI=sqlite:///.../mlflow.db`).
