"""
H1 DATA FETCH — run this first, before anything else.

Pulls from Open-Meteo (no API key needed):
  - CAMS NO₂ forecast history (Air Quality API)
  - ERA5: BLH, wind speed/dir, temperature 2m, precipitation (Historical API)
  - ERA5: temperature at 850hPa for inversion flag (pressure level)

Saves one CSV per city to backend/data/raw/
Runtime: ~2 minutes total for both cities.

Usage:
    python backend/data/fetch.py
"""

import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import requests_cache
import openmeteo_requests
from retry_requests import retry
import pandas as pd
import numpy as np
from pathlib import Path
from config.cities import CITIES, HISTORY_START, HISTORY_END
import requests

# ── Setup caching so re-runs don't burn quota ───────────────────────────────
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

cache = requests_cache.CachedSession(str(RAW_DIR / ".cache"), expire_after=86400)
session = retry(cache, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=session)


def fetch_cams_no2(city_key: str) -> pd.DataFrame:
    """Pull historical NO₂ from Open-Meteo Air Quality API."""
    city = CITIES[city_key]
    print(f"  Fetching Open-Meteo NO₂ for {city['label']}...")

    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "nitrogen_dioxide",
        "start_date": HISTORY_START,
        "end_date": HISTORY_END,
        "timezone": "Africa/Tunis",
    }
    
    r = om.weather_api(
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
        "time": times,
        "no2_ug_m3": r.Variables(0).ValuesAsNumpy()
    }).dropna(subset=["no2_ug_m3"])

    df["date"] = df["time"].dt.date
    daily = df.groupby("date")["no2_ug_m3"].mean().reset_index()
    daily.columns = ["date", "cams_no2_daily"]

    print(f"    → {len(daily)} days, NO₂ range: "
          f"{daily['cams_no2_daily'].min():.1f}–"
          f"{daily['cams_no2_daily'].max():.1f} μg/m³")
          
    # Sanity check — if values are suspiciously low (model sees background only)
    # apply a regional correction factor based on literature
    median = daily["cams_no2_daily"].median()
    if median < 5.0:
        print(f"    ⚠ Median {median:.2f} μg/m³ too low — applying regional correction")
        from config.cities import NO2_BASELINE
        scale = NO2_BASELINE[city_key]["default"] / max(median, 0.5)
        daily["cams_no2_daily"] *= scale
        print(f"    → Corrected median: {daily['cams_no2_daily'].median():.1f} μg/m³")
        
    return daily


def fetch_era5_weather(city_key: str) -> pd.DataFrame:
    """Pull hourly ERA5 variables from Open-Meteo Historical API."""
    city = CITIES[city_key]
    print(f"  Fetching ERA5 weather for {city['label']}...")

    # Standard surface variables
    params_surface = {
        "latitude":   city["lat"],
        "longitude":  city["lon"],
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "boundary_layer_height",
        ],
        "start_date": HISTORY_START,
        "end_date":   HISTORY_END,
        "timezone":   "Africa/Tunis",
        "models":     "era5",
    }

    r_surface = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params_surface
    )[0].Hourly()

    times = pd.date_range(
        start=pd.Timestamp(r_surface.Time(), unit="s", tz="Africa/Tunis"),
        end=pd.Timestamp(r_surface.TimeEnd(), unit="s", tz="Africa/Tunis"),
        freq=pd.Timedelta(seconds=r_surface.Interval()),
        inclusive="left",
    )

    hourly = pd.DataFrame({
        "time":              times,
        "temperature_2m":   r_surface.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m":   r_surface.Variables(1).ValuesAsNumpy(),
        "wind_dir_10m":     r_surface.Variables(2).ValuesAsNumpy(),
        "precipitation":    r_surface.Variables(3).ValuesAsNumpy(),
        "blh":              r_surface.Variables(4).ValuesAsNumpy(),  # boundary layer height (m)
    })

    # Pressure level: temperature at 850hPa for inversion detection
    try:
        params_850 = {
            "latitude":   city["lat"],
            "longitude":  city["lon"],
            "hourly":     "temperature_850hPa",
            "start_date": HISTORY_START,
            "end_date":   HISTORY_END,
            "timezone":   "Africa/Tunis",
            "models":     "era5",
        }
        r_850 = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params_850
        )[0].Hourly()
        hourly["temperature_850hPa"] = r_850.Variables(0).ValuesAsNumpy()
    except Exception as e:
        print(f"    ⚠ 850hPa temp unavailable ({e}), approximating inversion flag from BLH")
        hourly["temperature_850hPa"] = np.nan

    # Daily aggregation
    hourly["date"] = hourly["time"].dt.date
    daily = hourly.groupby("date").agg(
        temp_2m_mean=("temperature_2m",    "mean"),
        temp_2m_min=("temperature_2m",     "min"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        wind_speed_min=("wind_speed_10m",  "min"),
        wind_dir_mean=("wind_dir_10m",     "mean"),
        precip_sum=("precipitation",       "sum"),
        blh_mean=("blh",                   "mean"),
        blh_min=("blh",                    "min"),
        temp_850_mean=("temperature_850hPa", "mean"),
    ).reset_index()

    print(f"    → {len(daily)} days, BLH range: "
          f"{daily['blh_mean'].min():.0f}–{daily['blh_mean'].max():.0f} m")
    return daily


def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add inversion, stagnation, and sirocco boolean flags."""
    df = df.copy()

    # Thermal inversion: 850hPa warmer than surface, OR BLH < 500m
    if df["temp_850_mean"].notna().any():
        df["inversion_flag"] = (
            (df["temp_850_mean"] > df["temp_2m_mean"]) |
            (df["blh_min"] < 500)
        ).astype(int)
    else:
        df["inversion_flag"] = (df["blh_min"] < 500).astype(int)

    # Stagnation: wind < 2 m/s sustained
    df["stagnation_flag"] = (df["wind_speed_mean"] < 2.0).astype(int)

    # Sirocco: southerly (120°–220°) and strong (> 8 m/s)
    south = ((df["wind_dir_mean"] >= 120) & (df["wind_dir_mean"] <= 220))
    strong = (df["wind_speed_mean"] > 8.0)
    df["sirocco_flag"] = (south & strong).astype(int)

    # Rain washout: precipitation > 2mm
    df["rain_flag"] = (df["precip_sum"] > 2.0).astype(int)

    # Day-of-year cyclical encoding
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    return df


def build_dataset(city_key: str) -> pd.DataFrame:
    """Merge CAMS NO₂ + ERA5 weather into one training DataFrame."""
    no2 = fetch_cams_no2(city_key)
    wx  = fetch_era5_weather(city_key)

    df = pd.merge(no2, wx, on="date", how="inner")
    df = compute_flags(df)

    # Lag features for NO₂ (temporal memory without full LSTM complexity)
    for lag in [1, 2, 3, 7]:
        df[f"no2_lag{lag}"] = df["cams_no2_daily"].shift(lag)

    # 7-day rolling mean
    df["no2_ma7"] = df["cams_no2_daily"].rolling(7, min_periods=3).mean()

    # Drop rows with NaN lags at start
    df = df.dropna(subset=["no2_lag7"]).reset_index(drop=True)

    out_path = RAW_DIR / f"{city_key}_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("AirGuard TN — Data Fetch (H1)")
    print("=" * 55)
    for city in ["monastir", "mahdia"]:
        print(f"\n[{city.upper()}]")
        build_dataset(city)
    print("\n✓ All data saved to backend/data/raw/")
    print("  Next: run python backend/model/train.py")
