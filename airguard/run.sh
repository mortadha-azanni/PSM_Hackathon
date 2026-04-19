#!/bin/bash
# AirGuard TN — full setup and launch
# Usage: bash run.sh
# 
# What this does:
#   1. Creates .env if it doesn't exist
#   2. Installs Python deps
#   3. Fetches all data from Open-Meteo (H1)
#   4. Trains the XGBoost model (H3)
#   5. Pre-computes green buffers (H8)
#   6. Builds episode replay data (H10)
#   7. Starts FastAPI on port 8000
#
# Run the frontend separately:
#   cd frontend && python -m http.server 8080

set -e
cd "$(dirname "$0")"

# ── .env ─────────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠  .env created from .env.example"
  echo "   Fill in SMTP_USER + SMTP_PASS for email alerts, then re-run."
fi

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ -d "venv" ]; then
  echo "Activating virtualenv..."
  source venv/bin/activate
else
  echo "Creating virtualenv..."
  python3 -m venv venv
  source venv/bin/activate
fi

# ── Python deps ───────────────────────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# ── Data fetch (Open-Meteo, ~2 min) ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " H1: Fetching data from Open-Meteo..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python backend/data/fetch.py

# ── Train model (~90 seconds) ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " H3: Training XGBoost model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python backend/model/train.py

# ── Pre-compute green buffers ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " H8: Computing green buffers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python backend/data/compute_buffers.py

# ── Episode replay data ────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " H10: Building episode replay data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python backend/data/build_episode_replay.py

# ── Start API ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Starting API on http://localhost:8000"
echo " Docs at  http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Run frontend separately:"
echo "   cd frontend && python -m http.server 8080"
echo "   Open: http://localhost:8080"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
