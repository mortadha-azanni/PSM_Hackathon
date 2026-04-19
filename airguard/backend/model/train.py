"""
H3 MODEL TRAINING — run after fetch.py succeeds.

Strategy: XGBoost bias-correction model.
Target = difference between what CAMS predicts and actual local NO₂.
Since we have no ground-truth sensors, CAMS daily mean IS our target
for demo purposes. The model learns seasonal + meteorological corrections.

For a stronger scientific story: if you can pull TROPOMI column NO₂ via
GEE, replace `cams_no2_daily` in the target with TROPOMI values.
The training code below is identical either way — just swap the column.

Usage:
    python backend/model/train.py

Outputs:
    backend/model/saved/monastir_model.pkl
    backend/model/saved/mahdia_model.pkl
    backend/model/saved/metadata.json   ← validation metrics, feature list
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from config.cities import CITIES

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features used for training — must match exactly what predict.py sends
FEATURE_COLS = [
    "cams_no2_daily",     # CAMS baseline (main signal)
    "no2_lag1",           # yesterday
    "no2_lag2",
    "no2_lag3",
    "no2_lag7",           # same weekday last week
    "no2_ma7",            # 7-day trend
    "blh_mean",           # boundary layer height
    "blh_min",            # daily minimum BLH (worst inversion)
    "wind_speed_mean",
    "wind_speed_min",
    "temp_2m_mean",
    "precip_sum",
    "inversion_flag",
    "stagnation_flag",
    "sirocco_flag",
    "rain_flag",
    "doy_sin",
    "doy_cos",
]

# Forecast targets: predict NO₂ N days ahead
# Using shifted CAMS as pseudo-target (see docstring above)
HORIZONS = {
    "24h": 1,
    "48h": 2,
    "72h": 3,
}


def train_city(city_key: str) -> dict:
    csv = RAW_DIR / f"{city_key}_daily.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — run fetch.py first")

    df = pd.read_csv(csv, parse_dates=["date"])
    print(f"\n[{city_key.upper()}] {len(df)} rows loaded")

    # Chronological train/val split — never shuffle time series data
    split_date = "2025-01-01"
    train = df[df["date"] < split_date].copy()
    val   = df[df["date"] >= split_date].copy()
    print(f"  Train: {len(train)} days | Val: {len(val)} days")

    models = {}
    scalers = {}
    metrics = {}

    for horizon_label, shift in HORIZONS.items():
        # Target: NO₂ shift days into the future
        y_train = train["cams_no2_daily"].shift(-shift).dropna()
        X_train = train.loc[y_train.index, FEATURE_COLS]

        y_val = val["cams_no2_daily"].shift(-shift).dropna()
        X_val = val.loc[y_val.index, FEATURE_COLS]

        # Scale features
        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val_s)
        mae  = mean_absolute_error(y_val, y_pred)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # Uncertainty estimate: 1.5 × std of validation residuals
        residuals = np.abs(y_val.values - y_pred)
        uncertainty = float(1.5 * np.std(residuals))

        print(f"  {horizon_label}: MAE={mae:.2f} μg/m³  RMSE={rmse:.2f}  "
              f"±CI={uncertainty:.2f}")

        models[horizon_label]  = model
        scalers[horizon_label] = scaler
        metrics[horizon_label] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "uncertainty_sigma": round(uncertainty, 3),
            "val_days": len(y_val),
        }

    # Save everything in a single pickle per city
    artifact = {
        "models":   models,
        "scalers":  scalers,
        "feature_cols": FEATURE_COLS,
        "horizons": HORIZONS,
    }
    with open(MODEL_DIR / f"{city_key}_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"  ✓ Saved → {MODEL_DIR}/{city_key}_model.pkl")
    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("AirGuard TN — Model Training (H3)")
    print("=" * 55)

    all_metrics = {}
    for city in ["monastir", "mahdia"]:
        all_metrics[city] = train_city(city)

    # Save metadata with REAL metrics — goes into README and UI
    meta_path = MODEL_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "feature_cols":   FEATURE_COLS,
            "horizons":       HORIZONS,
            "validation_metrics": all_metrics,
            "model_type":     "XGBoostRegressor (CAMS bias corrector)",
            "trained_on":     "ERA5 + CAMS via Open-Meteo, 2021–2024",
        }, f, indent=2)

    print(f"\n✓ Metrics saved → {meta_path}")
    print("  Next: run python backend/main.py")
