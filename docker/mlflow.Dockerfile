FROM python:3.11-slim

# Keep in sync with the mlflow version pinned in poetry.lock
RUN pip install --no-cache-dir mlflow==3.16.1

WORKDIR /mlflow
EXPOSE 5000

# --allowed-hosts: recent MLflow rejects unknown Host headers (HTTP 403). The trainer container
# reaches this server as "mlflow:5000", so that name must be allowed alongside localhost.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:*,127.0.0.1:*"]
