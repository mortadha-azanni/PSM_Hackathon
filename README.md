# AirGuard TN 🌍
**AI-Powered Air Quality Forecasting & Proactive Mitigation**

Turning environmental data into actionable urban intelligence for the cities of Monastir and Mahdia.

## 📖 Overview
Urbanization and industrial activity lead to hazardous spikes in Nitrogen Dioxide (NO₂), affecting public health. Current environmental monitoring strictly looks at *past* or *current* data, making it reactive. 

**AirGuard TN** is an end-to-end predictive monitoring system. It doesn't just measure air quality—it forecasts NO₂ pollution levels up to 3 days in advance, automatically triggers localized alerts, and recommends optimal locations for **Green Buffers** (vegetation zones) to offset NO₂ concentration.

---

## ✨ Key Features
- **Predictive Intelligence:** 24h, 48h, and 72h NO₂ forecasts with confidence bounds using XGBoost.
- **Risk-Based Alerting:** Automated email alerts when a multi-variable Danger Score (combining NO₂, wind stagnation, and boundary layer height) is breached.
- **Strategic Green Buffers:** Spatial AI algorithms recommend where to place buffer zones (parks, tree lines) upwind of vulnerable grid cells to block pollution flow.
- **Episode Replay:** Historical data playback for stakeholder review and regulatory compliance.
- **Real-Time Monitoring Dashboard:** A lightweight Map UI (Leaflet.js + Chart.js) that polls fast asynchronous backend endpoints.

---

## 🧩 System Architecture

### 1. Data Ingestion
- **Open-Meteo API:** CAMS NO₂ levels + ERA5 weather data (temperature, wind speed/direction, boundary layer height).
- **Overpass API (OSM):** Extracts geospatial data to identify existing vegetation and candidate zones for buffers.

### 2. AI & Analytics Engine (XGBoost)
- Custom predictive models trained specifically for Monastir & Mahdia on 1,900+ days of historical data.
- Expands macroscopic city predictions into high-res grid cells with calculated risk metrics.

### 3. Backend (FastAPI)
- Handles inference, dynamic alerting logic (debounced zone alerts), and serving pre-computed episodes.

### 4. Client Dashboard (Leaflet.js / HTML5)
- Renders live heatmaps, risk layers, and notifications via continuous 30-second polling.

---

## 🚀 Quick Start Pipeline

The project is structured as a plug-and-play agent. Ensure you have Python installed.

```bash
cd airguard/

# 1. Automatic E2E Setup & Launch:
bash run.sh
```

**Manual Execution Steps:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch live data
python backend/data/fetch.py
# 2. Train XGBoost models
python backend/model/train.py
# 3. Generate green buffers & historical episodes
python backend/data/compute_buffers.py
python backend/data/build_episode_replay.py
# 4. Launch the API serving layer
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*In a separate terminal, launch the frontend:*
```bash
cd airguard/frontend && python -m http.server 8080
```