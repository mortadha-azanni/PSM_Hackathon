"""
Single source of truth for city coordinates and bounding boxes.
All lat/lon values verified against OSM for Monastir and Mahdia.
"""

import datetime

CITIES = {
    "monastir": {
        "lat": 35.7643,       # representative center point for API calls
        "lon": 10.8113,
        "bbox": {
            "lat_min": 35.45, "lat_max": 35.92,   # Jemmal → Bembla
            "lon_min": 10.60, "lon_max": 11.05,   # coast → inland
        },
        "grid_lats": [35.48, 35.53, 35.58, 35.63, 35.68,
                      35.73, 35.78, 35.83, 35.88],
        "grid_lons": [10.63, 10.69, 10.75, 10.81, 10.87, 10.93, 10.99],
        "label": "Monastir Governorate",
        "industrial_note": "Moknine, Ksar Hellal, Jemmal, Zeramdine textile cluster",
    },
    "mahdia": {
        "lat": 35.3547,       # shifted south to cover full governorate
        "lon": 11.0000,
        "bbox": {
            "lat_min": 35.00, "lat_max": 35.65,   # El Jem → Ksour Essef
            "lon_min": 10.75, "lon_max": 11.25,
        },
        "grid_lats": [35.03, 35.10, 35.17, 35.24, 35.31,
                      35.38, 35.45, 35.52, 35.59],
        "grid_lons": [10.78, 10.85, 10.92, 10.99, 11.06, 11.13, 11.20],
        "label": "Mahdia Governorate",
        "industrial_note": "Mahdia port, Chebba, El Jem, Bou Merdes",
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
HISTORY_END   = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

# Forecast horizons in hours
FORECAST_HORIZONS = [24, 48, 72]

# Observed approximate NO₂ ranges for this corridor from literature
# Used as fallback baseline if live API unavailable
NO2_BASELINE = {
    "monastir": {"winter": 45.0, "summer": 22.0, "default": 34.0},
    "mahdia":   {"winter": 38.0, "summer": 18.0, "default": 28.0},
}