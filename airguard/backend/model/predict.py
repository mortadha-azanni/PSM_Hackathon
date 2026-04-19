"""
Inference module — loaded once at FastAPI startup, called per request.

predict_city(city_key, horizon_h) → dict with:
  - grid of {lat, lon, no2_pred, ci_low, ci_high, danger_score, flags}
  - weather_summary dict
  - compliance status per standard
"""

import os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import requests_cache
import openmeteo_requests
from retry_requests import retry
from pathlib import Path
from config.cities import CITIES, THRESHOLDS

MODEL_DIR = Path(__file__).parent / "saved"

_cache = requests_cache.CachedSession(
    str(Path(__file__).parent.parent / "data" / "raw" / ".cache"),
    expire_after=3600
)
_session = retry(_cache, retries=3, backoff_factor=0.2)
_om = openmeteo_requests.Client(session=_session)

# Load all models into memory once
_artifacts: dict = {}

def load_models():
    for city in ["monastir", "mahdia"]:
        path = MODEL_DIR / f"{city}_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _artifacts[city] = pickle.load(f)
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _artifacts["metadata"] = json.load(f)
    print(f"  ✓ Models loaded for: {[k for k in _artifacts if k != 'metadata']}")


def _fetch_current_weather(city_key: str) -> dict:
    """Fetch today's weather from Open-Meteo Forecast API."""
    city = CITIES[city_key]
    params = {
        "latitude":  city["lat"],
        "longitude": city["lon"],
        "hourly": [
            "temperature_2m", "wind_speed_10m",
            "wind_direction_10m", "precipitation",
            "boundary_layer_height",
        ],
        "forecast_days": 4,
        "timezone": "Africa/Tunis",
    }
    try:
        r = _om.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params=params
        )[0].Hourly()
        times = pd.date_range(
            start=pd.Timestamp(r.Time(), unit="s", tz="Africa/Tunis"),
            end=pd.Timestamp(r.TimeEnd(), unit="s", tz="Africa/Tunis"),
            freq=pd.Timedelta(seconds=r.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({
            "time":           times,
            "temperature_2m": r.Variables(0).ValuesAsNumpy(),
            "wind_speed_10m": r.Variables(1).ValuesAsNumpy(),
            "wind_dir_10m":   r.Variables(2).ValuesAsNumpy(),
            "precipitation":  r.Variables(3).ValuesAsNumpy(),
            "blh":            r.Variables(4).ValuesAsNumpy(),
        })
        df["date"] = df["time"].dt.date
        daily = df.groupby("date").agg(
            temp_2m_mean=("temperature_2m",  "mean"),
            wind_speed_mean=("wind_speed_10m","mean"),
            wind_speed_min=("wind_speed_10m", "min"),
            wind_dir_mean=("wind_dir_10m",    "mean"),
            precip_sum=("precipitation",      "sum"),
            blh_mean=("blh",                  "mean"),
            blh_min=("blh",                   "min"),
        ).reset_index()
        return daily
    except Exception as e:
        print(f"  ⚠ Weather fetch failed: {e}, using fallback defaults")
        return None


def _fetch_current_no2(city_key: str) -> pd.DataFrame:
    """Fetch recent CAMS NO₂ for lag features."""
    city = CITIES[city_key]
    params = {
        "latitude":    city["lat"],
        "longitude":   city["lon"],
        "hourly":      "nitrogen_dioxide",
        "past_days":   14,
        "forecast_days": 4,
        "timezone":    "Africa/Tunis",
    }
    try:
        r = _om.weather_api(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params=params
        )[0].Hourly()
        times = pd.date_range(
            start=pd.Timestamp(r.Time(), unit="s", tz="Africa/Tunis"),
            end=pd.Timestamp(r.TimeEnd(), unit="s", tz="Africa/Tunis"),
            freq=pd.Timedelta(seconds=r.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({
            "time":    times,
            "no2_ug":  r.Variables(0).ValuesAsNumpy(),
        })
        df["date"] = df["time"].dt.date
        return df.groupby("date")["no2_ug"].mean().reset_index(
            ).rename(columns={"no2_ug": "cams_no2_daily"})
    except Exception as e:
        print(f"  ⚠ NO₂ fetch failed: {e}")
        return None


def _build_feature_row(city_key: str, horizon_h: int) -> dict | None:
    """Build a single feature row for the requested forecast horizon."""
    art = _artifacts.get(city_key)
    if not art:
        return None

    no2_df = _fetch_current_no2(city_key)
    wx_df  = _fetch_current_weather(city_key)
    if no2_df is None or wx_df is None:
        return None

    # Find "today" row in weather (first forecast day)
    wx_today = wx_df.iloc[0]
    no2_recent = no2_df.tail(10).reset_index(drop=True)

    # Build flags
    inversion = int(wx_today["blh_min"] < 500 or wx_today["blh_mean"] < 700)
    stagnation = int(wx_today["wind_speed_mean"] < 2.0)
    south = 120 <= wx_today["wind_dir_mean"] <= 220
    sirocco = int(south and wx_today["wind_speed_mean"] > 8.0)
    rain = int(wx_today["precip_sum"] > 2.0)

    import datetime
    doy = pd.Timestamp.today().day_of_year
    doy_sin = float(np.sin(2 * np.pi * doy / 365))
    doy_cos = float(np.cos(2 * np.pi * doy / 365))

    no2_vals = no2_recent["cams_no2_daily"].values
    cams_today = float(no2_vals[-1]) if len(no2_vals) else 20.0

    def lag(n):
        idx = len(no2_vals) - n
        return float(no2_vals[idx]) if idx >= 0 else cams_today

    ma7 = float(np.mean(no2_vals[-7:])) if len(no2_vals) >= 7 else cams_today

    row = {
        "cams_no2_daily":   cams_today,
        "no2_lag1":         lag(1),
        "no2_lag2":         lag(2),
        "no2_lag3":         lag(3),
        "no2_lag7":         lag(7),
        "no2_ma7":          ma7,
        "blh_mean":         float(wx_today["blh_mean"]),
        "blh_min":          float(wx_today["blh_min"]),
        "wind_speed_mean":  float(wx_today["wind_speed_mean"]),
        "wind_speed_min":   float(wx_today["wind_speed_min"]),
        "temp_2m_mean":     float(wx_today["temp_2m_mean"]),
        "precip_sum":       float(wx_today["precip_sum"]),
        "inversion_flag":   inversion,
        "stagnation_flag":  stagnation,
        "sirocco_flag":     sirocco,
        "rain_flag":        rain,
        "doy_sin":          doy_sin,
        "doy_cos":          doy_cos,
    }

    return {
        "row": row,
        "flags": {
            "inversion":  bool(inversion),
            "stagnation": bool(stagnation),
            "sirocco":    bool(sirocco),
            "rain":       bool(rain),
            "sea_breeze": False,  # approximated: requires coastal cell logic
        },
        "weather_summary": {
            "wind_speed_ms":           float(round(wx_today["wind_speed_mean"], 1)),
            "wind_direction_deg":      float(round(wx_today["wind_dir_mean"], 0)),
            "boundary_layer_height_m": float(round(wx_today["blh_mean"], 0)),
            "inversion_detected":      bool(inversion),
            "sirocco_active":          bool(sirocco),
        },
    }


def compute_danger_score(no2_pred: float, wind_speed: float,
                          inversion: bool, blh: float) -> int:
    """Composite danger score 1–10."""
    no2_factor   = min(no2_pred / THRESHOLDS["WHO_AQG_daily"], 2.0)
    stagnation   = 1.0 - min(wind_speed / 4.0, 1.0)
    inv_penalty  = 1.5 if inversion else 1.0
    blh_factor   = max(0.0, 1.0 - blh / 1500.0)

    raw = (0.45 * no2_factor + 0.30 * stagnation + 0.25 * blh_factor) * inv_penalty
    return int(np.clip(round(raw * 10), 1, 10))


def predict_city(city_key: str, horizon_h: int) -> dict:
    """
    Main inference function called by FastAPI.
    Returns full grid JSON for the requested city and horizon.
    """
    art = _artifacts.get(city_key)
    if not art:
        raise ValueError(f"No model loaded for city: {city_key}")

    horizon_label = f"{horizon_h}h"
    if horizon_label not in art["models"]:
        raise ValueError(f"Horizon {horizon_label} not available")

    model   = art["models"][horizon_label]
    scaler  = art["scalers"][horizon_label]
    feat_cols = art["feature_cols"]
    meta    = _artifacts.get("metadata", {})
    uncertainty = (meta.get("validation_metrics", {})
                       .get(city_key, {})
                       .get(horizon_label, {})
                       .get("uncertainty_sigma", 5.0))

    ctx = _build_feature_row(city_key, horizon_h)
    if ctx is None:
        raise RuntimeError("Failed to build feature row — check data connection")

    X = pd.DataFrame([ctx["row"]])[feat_cols]
    X_s = scaler.transform(X)
    no2_pred = float(model.predict(X_s)[0])
    ci_low   = max(0.0, no2_pred - uncertainty)
    ci_high  = no2_pred + uncertainty

    city = CITIES[city_key]
    wx   = ctx["weather_summary"]

    # Build grid cells (each cell gets small random spatial variation for demo)
    rng = np.random.default_rng(seed=42)
    grid = []
    for lat in city["grid_lats"]:
        for lon in city["grid_lons"]:
            jitter   = float(rng.normal(0, 2.5))
            cell_no2 = max(0.0, no2_pred + jitter)
            ds = compute_danger_score(
                cell_no2, wx["wind_speed_ms"],
                ctx["flags"]["inversion"], wx["boundary_layer_height_m"]
            )
            grid.append({
                "lat":           lat,
                "lon":           lon,
                "no2_predicted": round(cell_no2, 2),
                "no2_unit":      "μg/m³",
                "ci_low":        round(max(0, ci_low + jitter), 2),
                "ci_high":       round(ci_high + jitter, 2),
                "danger_score":  ds,
                "zone_type":     "red" if ds >= 7 else "orange" if ds >= 5 else "green",
                "inversion_flag": ctx["flags"]["inversion"],
                "stagnation_flag": ctx["flags"]["stagnation"],
            })

    # Compliance check
    compliance = {}
    for std_name, threshold in THRESHOLDS.items():
        compliance[std_name] = {
            "threshold_ug_m3": threshold,
            "predicted_ug_m3": round(no2_pred, 2),
            "status": "breach_likely" if no2_pred > threshold else "within_limit",
        }

    return {
        "city":            city_key,
        "horizon_hours":   horizon_h,
        "no2_mean_pred":   round(no2_pred, 2),
        "ci_low":          round(ci_low, 2),
        "ci_high":         round(ci_high, 2),
        "grid_cells":      grid,
        "weather_summary": ctx["weather_summary"],
        "flags":           ctx["flags"],
        "compliance":      compliance,
    }
