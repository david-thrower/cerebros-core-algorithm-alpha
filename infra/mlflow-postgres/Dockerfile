FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install MLflow with Postgres driver
RUN pip install --upgrade pip && \
    pip install mlflow psycopg2-binary gunicorn

WORKDIR /app
EXPOSE 5000

# Note: backend-store-uri points to the postgres service defined in docker-compose
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow_artifacts"]
