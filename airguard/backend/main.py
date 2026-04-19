"""
AirGuard TN — FastAPI backend
Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET  /api/v1/forecast?city=monastir&horizon=48
  GET  /api/v1/green-buffers?city=monastir
  POST /api/v1/alerts/trigger   body: {"city": "monastir", "horizon": 48}
  GET  /api/v1/alerts/recent
  GET  /api/v1/health
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.model.predict import load_models, predict_city
from backend.utils.alerts import check_and_trigger, get_recent_alerts
from config.cities import CITIES, FORECAST_HORIZONS

app = FastAPI(
    title="AirGuard TN API",
    description="NO₂ spatio-temporal forecasting for Monastir–Mahdia corridor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend at /
FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

GREEN_BUFFER_DIR = Path(__file__).parent.parent / "frontend" / "data"


@app.on_event("startup")
async def startup():
    print("\nAirGuard TN — starting up")
    load_models()
    print("  Ready.\n")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
def health():
    return {"status": "ok", "cities": list(CITIES.keys())}


# ── Forecast ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/forecast")
def forecast(
    city:    str = Query(...,  description="monastir or mahdia"),
    horizon: int = Query(48,   description="Hours ahead: 24, 48, 72"),
):
    city = city.lower()
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}. Use: {list(CITIES.keys())}")
    if horizon not in FORECAST_HORIZONS:
        raise HTTPException(400, f"Horizon must be one of {FORECAST_HORIZONS}")
    try:
        payload = predict_city(city, horizon)
        return payload
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Green buffers ─────────────────────────────────────────────────────────────
@app.get("/api/v1/green-buffers")
def green_buffers(city: str = Query(..., description="monastir or mahdia")):
    city = city.lower()
    path = GREEN_BUFFER_DIR / f"green_buffers_{city}.json"
    if not path.exists():
        raise HTTPException(404, f"Green buffer data for {city} not found. "
                                 f"Run backend/data/compute_buffers.py first.")
    with open(path) as f:
        return json.load(f)


# ── Alert trigger ─────────────────────────────────────────────────────────────
class AlertRequest(BaseModel):
    city:    str
    horizon: int = 48


@app.post("/api/v1/alerts/trigger")
def trigger_alert(req: AlertRequest):
    city = req.city.lower()
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city: {city}")
    try:
        payload   = predict_city(city, req.horizon)
        triggered = check_and_trigger(payload)
        return {
            "alerts_triggered": len(triggered),
            "alerts": triggered,
            "danger_score_max": max(
                (c["danger_score"] for c in payload["grid_cells"]), default=0
            ),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Recent alerts ─────────────────────────────────────────────────────────────
@app.get("/api/v1/alerts/recent")
def recent_alerts(limit: int = Query(20)):
    return {"alerts": get_recent_alerts(limit)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True,
    )
