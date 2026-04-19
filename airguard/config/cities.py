"""
Single source of truth for city coordinates and bounding boxes.
All lat/lon values verified against OSM for Monastir and Mahdia.
"""

CITIES = {
    "monastir": {
        "lat": 35.7643,
        "lon": 10.8113,
        "bbox": {
            "lat_min": 35.70, "lat_max": 35.82,
            "lon_min": 10.76, "lon_max": 10.88,
        },
        # Grid resolution: ~5km cells across the corridor
        "grid_lats": [35.71, 35.73, 35.75, 35.77, 35.79, 35.81],
        "grid_lons": [10.77, 10.79, 10.81, 10.83, 10.85, 10.87],
        "label": "Monastir",
        "industrial_note": "Moknine textile cluster, Ksar Hellal dyeing plants",
    },
    "mahdia": {
        "lat": 35.5047,
        "lon": 11.0622,
        "bbox": {
            "lat_min": 35.45, "lat_max": 35.55,
            "lon_min": 11.00, "lon_max": 11.10,
        },
        "grid_lats": [35.46, 35.48, 35.50, 35.52, 35.54],
        "grid_lons": [11.01, 11.03, 11.05, 11.07, 11.09],
        "label": "Mahdia",
        "industrial_note": "Coastal port, fishing industry, construction materials",
    },
}

# WHO + Tunisian regulatory thresholds for NO₂
# All in μg/m³
THRESHOLDS = {
    "WHO_AQG_daily":    25.0,   # WHO 2021 daily mean
    "WHO_AQG_annual":   10.0,   # WHO 2021 annual mean
    "EU_daily_1h_peak": 200.0,  # EU 2008/50/EC 1-hour peak
    "EU_annual":        40.0,   # EU 2008/50/EC annual mean
    "TN_2018_447":      40.0,   # Tunisia Decree 2018-447 (mirrors EU annual)
}

# Training data range
HISTORY_START = "2021-01-01"
HISTORY_END   = "2026-04-18"   # yesterday — never include today (partial day)

# Forecast horizons in hours
FORECAST_HORIZONS = [24, 48, 72]
