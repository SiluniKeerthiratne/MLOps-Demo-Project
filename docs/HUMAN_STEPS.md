# Steps that need a human

The project itself needs no credentials. These are the only manual steps, all optional except the
last one.

1. **Publish to GitHub (optional).** Run `gh auth login`, create a repository, and push the commits
   *and tags* (`git push origin main --tags`). Make the repo public if others should clone it. CI
   (`.github/workflows/ci.yml`) then runs with no secrets.
2. **Docker Hub login (optional).** Only if Docker Hub rate limits block pulling the
   `python:3.11-slim` base image: `docker login`. Nothing in this project pushes images.
3. **Commit the build and create the dataset tags.** The build leaves everything uncommitted.
   Commit, then make sure the tags point at commits whose `data/raw/wine.csv.dvc` has the matching
   md5 (v1: `87a206c70a60bb7431293486bcd185bd`, v2: `26381881ce0db229f49a8f6c6142e80a`).
   `make demo` does this automatically on a fresh run. To do it by hand:

   ```bash
   git add -A && git commit -m "MLOps demo project"      # pointer currently holds dataset v1
   git tag data-v1
   make data VERSION=v2
   git add data/raw/wine.csv.dvc && git commit -m "Dataset v2" && git tag data-v2
   ```

   Commit `dvc.lock` and `data/.gitignore` as well. Do not commit `data/raw/wine.csv`, `dvc-storage/`
   or `.env` (all are git-ignored).
