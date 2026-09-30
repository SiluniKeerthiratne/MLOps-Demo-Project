# Architecture

```mermaid
flowchart LR
    subgraph Data["Data (DVC, local remote)"]
        GEN[make_dataset.py<br/>deterministic v1 / v2] --> RAW[data/raw/wine.csv]
        RAW -- dvc add --> PTR[wine.csv.dvc<br/>in git, tags data-v1 / data-v2]
        RAW -- dvc push --> STORE[(dvc-storage/<br/>git-ignored)]
    end

    subgraph Pipeline["dvc.yaml pipeline (trainer container)"]
        RAW --> PRE[preprocess.py]
        PRE --> PROC[train.csv, test.csv,<br/>schema.json]
        PROC --> TR[train.py]
        PARAMS[params.yaml] --> TR
    end

    subgraph MLflow["MLflow server (Docker, SQLite + volume)"]
        RUNS[(Runs: params, metrics,<br/>dataset version + md5)]
        REG[(Model Registry<br/>aliases: staging, production)]
    end

    TR -- log + register --> RUNS
    RUNS --> REG
    TR -- alias staging --> REG
    EV[evaluate.py<br/>quality gate] -- validation_status --> REG
    PR[promote.py] -- alias production,<br/>old one archived --> REG
    REG --> EX[export.py]
    EX --> MODELS[models/vN/<br/>model.pkl, metadata.json,<br/>schema.json + current.json]

    subgraph Serving["API container (non-root, no MLflow)"]
        MODELS -- read-only mount --> API[FastAPI<br/>/predict /health /stats]
    end

    CLIENT([client]) --> API
    API -- one line per request --> LOG[(logs/predictions.jsonl)]
    LOG --> API
```

`/stats` is computed from the JSONL log, so monitoring survives container restarts as long as
`./logs` is kept.

## Model card

| | |
|---|---|
| **Model** | scikit-learn `Pipeline`: `StandardScaler` + `RandomForestClassifier` (default) or `LogisticRegression`, chosen in `params.yaml` |
| **Task** | 3-class classification of wine cultivar from 13 chemical measurements |
| **Data** | scikit-learn Wine dataset. `v1` = 178 original rows. `v2` = `v1` + 120 seeded augmented rows (bootstrap resample, 2% multiplicative Gaussian noise) |
| **Split** | Stratified 80/20, seed 42 |
| **Metrics** | Accuracy, macro F1, per-class precision and recall on the held-out split. Quality gate: macro F1 >= 0.90 |
| **Inputs** | 13 numeric features listed in `schema.json` (training min/max are used for out-of-range warnings) |
| **Outputs** | Predicted class (0, 1, 2) and class probabilities |
| **Intended use** | Demonstrating an MLOps lifecycle. Not for any real decision |
| **Limitations** | Tiny, easy, clean dataset: metrics near 1.0 say little about generalisation. `v2` rows are noisy copies of `v1` rows, so its test split leaks training information. No drift detection, no authentication |
| **Lineage** | Each model version links to its MLflow run, which records dataset version, DVC md5, train-set SHA-256, git commit, and Python/library versions |
