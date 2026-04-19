# AirGuard TN — Agent Setup Guide

## Start here

```bash
git clone <your-repo>
cd airguard
bash run.sh
```

`run.sh` does everything in sequence. The only thing that can fail is
the SMTP config — email alerts won't send, but the rest works fine.

---

## File map

```
airguard/
├── run.sh                          ← START HERE (full setup + launch)
├── requirements.txt
├── .env.example                    → copy to .env, fill SMTP_*
│
├── config/
│   └── cities.py                   ← coordinates, thresholds, feature list
│
├── backend/
│   ├── main.py                     ← FastAPI app (port 8000)
│   │
│   ├── data/
│   │   ├── fetch.py                ← H1: pulls Open-Meteo data → raw/*.csv
│   │   ├── compute_buffers.py      ← H8: green buffer JSON → frontend/data/
│   │   └── build_episode_replay.py ← H10: episode JSON → frontend/data/
│   │
│   ├── model/
│   │   ├── train.py                ← H3: trains XGBoost → saved/*.pkl
│   │   └── predict.py              ← inference, called at runtime
│   │
│   └── utils/
│       └── alerts.py               ← danger score → email alert dispatch
│
└── frontend/
    ├── index.html                  ← full Leaflet.js app (serve on port 8080)
    └── data/                       ← auto-generated JSON files
        ├── green_buffers_monastir.json
        ├── green_buffers_mahdia.json
        ├── episode_monastir.json
        └── episode_mahdia.json
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/forecast?city=monastir&horizon=48` | Grid forecast JSON |
| GET | `/api/v1/green-buffers?city=monastir` | Buffer recommendations |
| POST | `/api/v1/alerts/trigger` body: `{"city":"monastir","horizon":48}` | Fire alerts |
| GET | `/api/v1/alerts/recent` | Last 20 alert records |
| GET | `/api/v1/health` | Liveness check |
| GET | `/docs` | FastAPI Swagger UI |

---

## Manual step-by-step (if run.sh fails)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Data
python backend/data/fetch.py

# 3. Train
python backend/model/train.py

# 4. Buffers
python backend/data/compute_buffers.py

# 5. Replay
python backend/data/build_episode_replay.py

# 6. API (terminal 1)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Frontend (terminal 2)
cd frontend && python -m http.server 8080
# open http://localhost:8080
```

---

## What the agent should do next

Person A tasks (ML/backend):
- Run `fetch.py` first and verify both CSVs exist in `backend/data/raw/`
- Check that `blh_mean` column is populated (not all NaN)
- Run `train.py` and paste the real MAE/RMSE into this README
- If 850hPa temperature fails, the inversion flag falls back to BLH<500m — that is fine

Person B tasks (frontend):
- Open `frontend/index.html` in a browser with `python -m http.server 8080`
- Change `const API = 'http://localhost:8000/api/v1'` if the backend runs on a different port
- The map loads mock data from the API automatically once the backend is up
- Episode replay requires `frontend/data/episode_monastir.json` — built by `build_episode_replay.py`

---

## Known issues to watch for

1. **boundary_layer_height missing from ERA5**: If the historical API doesn't
   return BLH, train.py will fail. Fix: remove `blh_mean` and `blh_min` from
   FEATURE_COLS in config/cities.py and add `"blh_mean": 800` as a default
   in predict.py's fallback dict.

2. **Overpass API timeout**: compute_buffers.py will fall back to a synthetic
   grid automatically. The green buffer panel will still work.

3. **CORS error in browser**: Make sure uvicorn is running BEFORE opening the
   frontend. The frontend polls every 30 seconds — it will auto-recover once
   the API is up.
