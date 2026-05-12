#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${1:-dino-api}"
MLFLOW_PORT=5000
API_PORT=8000
MLFLOW_DIR="/home/ec2-user/mlflow-data"

# ── 1. MLflow server ──────────────────────────────────────────────────────────
echo "Starting MLflow server..."
mkdir -p "$MLFLOW_DIR/artifacts"

mlflow server \
  --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
  --artifacts-destination "$MLFLOW_DIR/artifacts" \
  --default-artifact-root "mlflow-artifacts:/" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" &

MLFLOW_PID=$!

until curl -s "http://localhost:$MLFLOW_PORT/health" > /dev/null 2>&1; do
  sleep 1
done
echo "MLflow ready (PID $MLFLOW_PID) → http://localhost:$MLFLOW_PORT"

# ── 2. Docker build ───────────────────────────────────────────────────────────
# Remove existing container if present
docker rm -f "$IMAGE_NAME" 2>/dev/null || true

echo "Building Docker image $IMAGE_NAME..."

STAGING=$(mktemp -d)
trap "rm -rf '$STAGING'" EXIT

cp "$PROJECT_ROOT/008 web/app/main.py"          "$STAGING/main.py"
cp "$PROJECT_ROOT/008 web/app/requirements.txt" "$STAGING/requirements.txt"
cp "$PROJECT_ROOT/001 Proprocesamiento/token_encoder.py" "$STAGING/token_encoder.py"
cp "$PROJECT_ROOT/Dockerfile"                   "$STAGING/Dockerfile"

mkdir -p "$STAGING/assets"
cp "$PROJECT_ROOT/001 Proprocesamiento/tokens/vocabulary.txt" "$STAGING/assets/vocabulary.txt"
cp "$PROJECT_ROOT/001 Proprocesamiento/tokens/merges.csv"     "$STAGING/assets/merges.csv"

docker build -t "$IMAGE_NAME" "$STAGING"
echo "Image $IMAGE_NAME built."

# ── 3. Run API container ──────────────────────────────────────────────────────
echo "Starting API container..."
docker run -d \
  --name "$IMAGE_NAME" \
  --add-host=host.docker.internal:host-gateway \
  -p "$API_PORT:8000" \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:$MLFLOW_PORT" \
  "$IMAGE_NAME"

echo ""
echo "MLflow → http://localhost:$MLFLOW_PORT"
echo "API    → http://localhost:$API_PORT"
