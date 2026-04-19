# AirGuard TN: Technical Data Documentation

This document outlines the data sources, processing methodologies, and reproducibility protocols for the AirGuard TN forecasting and mitigation system.

---

## 1. Description of Data Sources & Referencing

The AirGuard TN pipeline integrates multiple continuous live APIs and historical datasets to form a cohesive view of urban air quality and meteorology.

### 1.1 Meteorological Data (Weather & Atmosphere)
*   **Source:** Open-Meteo API (ERA5 Reanalysis archive for training; Live Forecast API for runtime).
*   **Variables Extracted:** 
    *   Surface level: `temperature_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`.
    *   Atmospheric level: `boundary_layer_height` (BLH), `temperature_850hPa`.
*   **Reference:** European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 dataset.

### 1.2 Air Quality Data (NO₂)
*   **Source:** Open-Meteo Copernicus Atmosphere Monitoring Service (CAMS) API.
*   **Variables Extracted:** Hourly NO₂ concentrations at surface level.
*   **Reference:** Copernicus Atmosphere Monitoring Service (CAMS) European reanalysis and global forecasting systems.

### 1.3 Geospatial & Land Use Data
*   **Source:** OpenStreetMap (OSM) via the Overpass API.
*   **Variables Extracted:** Urban infrastructure, land use polygons (industrial zones, farmland, scrub, grass, forests, orchards), and road networks.
*   **Reference:** OpenStreetMap contributors. (Note: A hardcoded geographic fallback table is maintained in the system for Overpass API rate-limit/timeout protections).

### 1.4 Regulatory Thresholds & Baselines
*   **Safety Thresholds:** 
    *   World Health Organization (WHO) 2021 Air Quality Guidelines (25 μg/m³ for 24h NO₂).
    *   EU Directive 2008/50/EC and Tunisian Decree 2018-447 (40 μg/m³).
*   **Baseline Calibration:** Regional NO₂ baselines derived from literature for the Sahel industrial corridor (Monastir: ~34.0 μg/m³, Mahdia: ~28.0 μg/m³).

---

## 2. Data Processing & Transformation Steps

The pipeline applies several deterministic transformations to convert raw observations into actionable predictive features.

### 2.1 Aggregation and Alignment
Raw hourly NO₂ and meteorological data fetched from APIs are mathematically aggregated to daily metrics (e.g., `groupby("date").mean()`) to align with the XGBoost time-series training horizons (+24h, +48h, +72h).

### 2.2 Bias Correction & Scaling (NO₂)
Due to spatial averaging inherent in satellite/reanalysis data, raw CAMS NO₂ values often underrepresent hyper-local industrial peaks. 
*   **Transformation:** We apply a multiplicative correction factor (×7 to ×12). If the raw median is < 5.0 μg/m³, it is multiplied by `NO2_BASELINE[city]["default"] / median` to align the baseline with scientifically realistic surface concentrations for Monastir and Mahdia.

### 2.3 Feature Engineering (Weather Flags)
Meteorological variables are transformed into categorical risk flags used by the alerting system:
*   **Inversion Flag:** Computed using atmospheric sounding. Triggers when `boundary_layer_height < 500m` OR `temperature_850hPa > temperature_2m` (indicating a trapped air mass).
*   **Stagnation Flag:** Triggers when `wind_speed_10m < 2.0 m/s`.
*   **Precipitation Flag:** Triggers when rain exceeds `2mm` (acts as a natural pollution scrubber).

### 2.4 Risk Scoring Generation
Instead of evaluating NO₂ in isolation, the system calculates a **1–10 Danger Score** using a weighted multi-variable index:
* **45% Weight:** Normalized NO₂ concentration against WHO/EU thresholds.
* **30% Weight:** Atmospheric stagnation (wind speed).
* **15% Weight:** Vertical trapping (Boundary Layer Height).

### 2.5 Spatial Grid Interpolation (Synthetic Expansion)
Because the core XGBoost model predicts a single macroscopic NO₂ value per city, spatial mapping is applied at runtime:
*   A distance-based decay algorithm mixed with a controlled Gaussian jitter variant ($\sigma = 2.5$) is applied to expand the single prediction across high-resolution grid cells for heatmap generation.

---

## 3. Reproducibility

The system is designed to be fully reproducible across environments, requiring no pre-compiled binaries or proprietary databases.

### 3.1 Environment Setup
Dependencies are explicitly locked. To prepare the environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Execution Pipeline
The entire MLOps and serving pipeline can be cleanly reproduced using the provided shell script:
```bash
bash run.sh
```

Alternatively, the sequential pipeline can be reproduced manually to inspect intermediate artifacts:
1. **Data Ingestion:** `python backend/data/fetch.py` (Outputs to `backend/data/raw/`)
2. **Model Training:** `python backend/model/train.py` (Outputs artifacts to `backend/model/saved/`)
3. **Data Precomputation:** 
   * `python backend/data/compute_buffers.py` (Generates green buffer placement grids)
   * `python backend/data/build_episode_replay.py` (Compiles historical review episodes)
4. **API Serving:** `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

All dynamic data is fetched live upon script execution, ensuring that running the pipeline always incorporates the latest Copernicus and ERA5 readings.