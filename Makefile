# Everything runs in Docker by default. Override RUN to run on the host instead, e.g.
#   make RUN="poetry run" DEVRUN="poetry run" train
COMPOSE ?= docker compose
RUN     ?= $(COMPOSE) run --rm trainer
DEVRUN  ?= $(COMPOSE) run --rm tester
ARGS    ?=

.PHONY: up mlflow data data-status data-checkout data-diff pull preprocess train evaluate \
        promote export serve report repro test lint lock-check demo down clean help

help:
	@grep -E '^#> ' Makefile | sed 's/^#> //'

#> up            start the MLflow server (http://localhost:5000)
up:
	$(COMPOSE) up -d --build --wait mlflow

mlflow: up

#> data VERSION=v1|v2   generate dataset, dvc add, dvc push, verify hash
data:
	@test -n "$(VERSION)" || (echo "Usage: make data VERSION=v1|v2" && exit 2)
	$(RUN) bash scripts/data.sh $(VERSION)

#> data-status   dvc status and dataset hash
data-status:
	$(RUN) dvc status
	$(RUN) python -m src.utils info

#> data-checkout TAG=data-v1   restore a dataset version from git + DVC
data-checkout:
	@test -n "$(TAG)" || (echo "Usage: make data-checkout TAG=data-v1" && exit 2)
	$(RUN) bash -c "git checkout $(TAG) -- data/raw/wine.csv.dvc && (dvc checkout data/raw/wine.csv.dvc || bash scripts/pull.sh)"
	$(RUN) python -m src.utils info

#> data-diff     dvc diff between data-v1 and data-v2
data-diff:
	$(RUN) dvc diff data-v1 data-v2 --targets data/raw/wine.csv

#> pull          dvc pull, or regenerate the dataset on a fresh clone
pull:
	$(RUN) bash scripts/pull.sh

#> preprocess    dvc repro preprocess
preprocess:
	$(RUN) dvc repro preprocess

#> train         train + register (staging). Extra args: make train ARGS="--model logreg"
train:
	$(RUN) python -m src.train $(ARGS)

evaluate:
	$(RUN) python -m src.evaluate $(ARGS)

#> promote       staging -> production (make promote ARGS=--force to override checks)
promote:
	$(RUN) python -m src.promote $(ARGS)

export:
	$(RUN) python -m src.export

#> serve         build and (re)start the API on http://localhost:8000
serve:
	$(COMPOSE) up -d --build --force-recreate --wait api

#> report        show registry versions and MLflow runs with their dataset versions
report:
	$(RUN) python -m src.report

#> repro         dvc repro: preprocess + train as a DVC pipeline
repro:
	$(RUN) dvc repro

test:
	$(DEVRUN) pytest -q

lint:
	$(DEVRUN) ruff check .

lock-check:
	poetry check --lock

demo:
	bash scripts/demo.sh

down:
	$(COMPOSE) --profile tools down

#> clean         stop everything, DELETE the MLflow volume and generated files
clean:
	$(COMPOSE) --profile tools down -v
	rm -rf models/v* models/current.json logs/predictions.jsonl data/processed metrics.json
