# Technical Report: AirGuard Spatio-Temporal NO₂ Forecasting System

## 1. Problem Statement
The Monastir-Mahdia coastal corridor faces recurrent and localized Nitrogen Dioxide (NO₂) air pollution spikes, exacerbated by industrial activities and complex spatio-temporal atmospheric interactions (such as atmospheric inversions and specific coastal wind patterns like the Sirocco). Currently, physical air quality sensors are too sparse to provide a continuous, high-resolution view of the local pollution dynamics. This data gap significantly hinders proactive environmental interventions, public health warnings, and compliance monitoring with legal guidelines. There is a critical need for a unified, predictive system capable of continuous forecasting, risk assessment, and actionable mitigation for local stakeholders.

## 2. Data Sources
To supplement the lack of extensive physical sensor grids, the system integrates robust, high-fidelity streams of localized environmental data:
*   **Open-Meteo European Forecast Base**: Provides crucial meteorological factors including 2m Surface Temperature, Vectorized Wind Speed and Direction, Boundary Layer Height (BLH), and Precipitation levels. These metrics act as primary input features for identifying inversion layers and washout potential.
*   **Copernicus Atmosphere Monitoring Service (CAMS)**: Acquired via Open-Meteo APIs, this provides historical, near-real-time, and high-resolution atmospheric composition reanalysis covering Nitrogen Dioxide (NO₂) concentration data.
*   **CAMS Global Models**: Utilized to assess global background concentrations, allowing the system to adjust and bias-correct local models.
*   **Regulatory Baselines**: Incorporates Tunisian Decree No. 2018-447 and WHO guidelines, which provide the deterministic threshold values used for automated compliance breaches and risk scoring.

## 3. Methodology
The solution leverages a decoupled architecture processing data through five advanced engineering stages, merging geostatistical interpolation with machine learning:
*   **Unified Data Ingestion**: Aggregating raw, unaligned Copernicus CAMS atmospheric data with ERA5 meteorological reanalysis.
*   **Physical Feature Engineering**: Transforming raw data into physical atmospheric heuristics. This includes calculating an Inversion Index (driven by comparing 850hPa pressure-level temperature with surface temperatures), calculating washout metrics based on precipitation, and utilizing cyclical encoding for model seasonal recognition.
*   **Temporal Memory Construction**: Building temporal momentum loops utilizing lagged historic values (e.g., T-1, T-3, T-7 days) and rolling averages to help algorithms recognize cumulative pollutant stagnation.
*   **Spatio-Temporal Discretization (Ordinary Kriging)**: Applying Ordinary Kriging algorithms to statistically interpolate sparse point data into a continuous high-resolution spatial grid (0.01°). This translates raw numerical severity into continuous geographic "Red Zone" heatmaps.
*   **Multi-Horizon Temporal Forecasting**: Employing parallel eXtreme Gradient Boosting (XGBoost) models tuned specifically to learn the complex, non-linear relationships between meteorological traps and NO₂ accumulations. Independent estimators are trained to predict precisely 24h, 48h, and 72h into the future, coupled with a customized Composite Danger Score logic (evaluating NO₂ levels against stagnation rules).

## 4. Results and Analysis
*   **High-Fidelity Predictability**: The ensemble XGBoost methodology efficiently categorizes conditions leading to stasis-induced pollution, adeptly separating benign baseline variations from critical atmospheric inversion events. The multi-horizon approach secures up to a 72-hour operational lead time for environmental intervention.
*   **Geospatial Actionability**: The integration of Kriging spatial interpolation maps coarse tabular point data into continuous, interactive Leaflet-powered visual web dashboards. This gives decision-makers precise targets for mitigation, such as deploying the algorithmic Green Buffer (Canopy) placement strategy at strictly localized high-danger coordinates.
*   **Automated and Managed Compliance**: The implemented intelligence loop actively scans the inferred grids against regulatory thresholds, utilizing intelligent debouncing (to prevent alert fatigue) and Sirocco flag notifications. 
*   **Conclusion**: Combined, these layers form a robust infrastructure—upgrading legacy, delayed reactionary monitoring into an active, predictive shield for managing atmospheric public health across the Tunisian coast.
