#!/bin/bash
MLFLOW_DIR="$HOME/mlflow-data"
mkdir -p "$MLFLOW_DIR/artifacts"

mlflow server \
  --backend-store-uri "sqlite:////home/ec2-user/mlflow-data/mlflow.db" \
  --artifacts-destination "/home/ec2-user/mlflow-data/artifacts" \
  --default-artifact-root "mlflow-artifacts:/" \
  --serve-artifacts \
  --host 0.0.0.0 --port 5000 &






