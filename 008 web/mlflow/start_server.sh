#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mlflow server \
    --backend-store-uri "sqlite:///$SCRIPT_DIR/mlflow.db" \
    --default-artifact-root "$SCRIPT_DIR/artifacts" \
    --host 0.0.0.0 \
    --port 5000
