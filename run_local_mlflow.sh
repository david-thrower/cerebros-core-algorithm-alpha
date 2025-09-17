#!/usr/bin/env bash
# Minimal local MLflow UI for R&D (isolated from any shared tracking)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export MLFLOW_TRACKING_URI="file:${SCRIPT_DIR}/mlruns"
echo "[R&D] MLflow tracking URI: ${MLFLOW_TRACKING_URI}"
echo "Starting MLflow UI at http://127.0.0.1:5000"
mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI}" --host 127.0.0.1 --port 5000