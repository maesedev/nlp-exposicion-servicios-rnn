#!/bin/bash
set -e

# ── Sistema ───────────────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y python3-pip git docker.io

# ── MLflow ────────────────────────────────────────────────────────────────────
python3 -m pip install mlflow boto3 --user --break-system-packages

# ── Docker ────────────────────────────────────────────────────────────────────
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# ── Repo ──────────────────────────────────────────────────────────────────────
REPO_DIR="/home/ubuntu/inference-endpoint"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull
else
  git clone https://github.com/maesedev/nlp-exposicion-servicios-rnn.git "$REPO_DIR"
fi

# ── Arrancar MLflow + FastAPI ─────────────────────────────────────────────────
cd "$REPO_DIR"
chmod +x build.sh
env PATH=$PATH:/usr/local/bin:$HOME/.local/bin bash build.sh
