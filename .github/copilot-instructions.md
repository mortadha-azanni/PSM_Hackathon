# Copilot Instructions for AirGuard TN

## Build, run, and verification commands

The runnable project lives in `airguard/`.

```bash
cd airguard
```

Primary end-to-end setup and launch:

```bash
bash run.sh
```

Manual pipeline (when you need one step at a time):

```bash
pip install -r requirements.txt
python backend/data/fetch.py
python backend/model/train.py
python backend/data/compute_buffers.py
python backend/data/build_episode_replay.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (separate terminal):

```bash
cd airguard/frontend && python -m http.server 8080
```

Project currently has no dedicated lint or automated test command/config. Use API health as a quick smoke check:

```bash
curl http://localhost:8000/api/v1/health
```

There is no single-test command because no test suite is present yet.

## High-level architecture

- `backend/data/fetch.py` builds training datasets per city by merging Open-Meteo CAMS NO₂ and ERA5 weather into `backend/data/raw/{city}_daily.csv`.
- `backend/model/train.py` trains per-city XGBoost horizon models (`24h`, `48h`, `72h`) and writes artifacts to `backend/model/saved/`.
- `backend/model/predict.py` loads artifacts once at API startup, fetches current NO₂/weather context from Open-Meteo, builds one feature row, predicts city-level NO₂, then expands into grid-cell forecasts with confidence bounds and danger scoring.
- `backend/main.py` exposes FastAPI endpoints for forecast, green buffers, alert triggering, and recent alerts; it also serves the static app under `/app`.
- `backend/data/compute_buffers.py` and `backend/data/build_episode_replay.py` precompute JSON files into `frontend/data/`; API/frontend expect those generated files to exist.
- `frontend/index.html` is a single-file Leaflet + Chart.js app that polls forecast/alerts every 30s and renders map heat, risk zones, compliance status, alert center, and replay.

## Key codebase conventions

- City scope is intentionally fixed to `monastir` and `mahdia` across pipeline, model loading, and UI controls. Additions require coordinated updates in `config/cities.py`, training loops, and frontend selectors.
- `config/cities.py` is the source of truth for coordinates, bboxes, grid geometry, thresholds, historical range, and allowed horizons.
- **Feature contract is strict**: `FEATURE_COLS` in `backend/model/train.py` must stay aligned with `_build_feature_row()` in `backend/model/predict.py`. Any feature change must be applied to both and requires retraining artifacts.
- Horizon keys in model artifacts are string labels (`"24h"`, `"48h"`, `"72h"`), while API inputs are integer hours (`24`, `48`, `72`); inference converts with `f"{horizon}h"`.
- Generated runtime artifacts are treated as part of normal flow:
  - raw datasets: `backend/data/raw/`
  - model artifacts: `backend/model/saved/`
  - precomputed frontend JSON: `frontend/data/`
  Missing artifacts should be regenerated with the pipeline scripts, not patched manually.
- Alerting is intentionally debounced by zone for 6 hours and persisted in `backend/data/alerts.json`.
- External-data fallbacks are expected behavior (not exceptional):
  - missing 850hPa temperature falls back to BLH-based inversion logic
  - Overpass failures fall back to synthetic grid buffer candidates
