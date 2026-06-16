#!/bin/bash
# =============================================================
# App Service startup script — runs on Azure every time the
# container starts (deploy, restart, or slot warm-up).
#
# Runtime env vars it expects (set in App Service App Settings,
# NOT in GitHub Secrets):
#   MODEL_ARTIFACT_SAS_URL  – read-only SAS URL to the .pkl blob
# =============================================================
set -e

MODEL_PATH="models/match_score_model.pkl"
mkdir -p models

echo "[startup] Downloading model artifact..."
curl -fsSL "${MODEL_ARTIFACT_SAS_URL}" -o "${MODEL_PATH}"
echo "[startup] Model ready — $(du -m ${MODEL_PATH} | cut -f1) MB"

echo "[startup] Starting Streamlit on port 8000..."
exec python -m streamlit run app/streamlit_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --server.headless true
