# AirGuard TN

NO₂ spatio-temporal forecasting for the Monastir–Mahdia corridor.
Built for the PSM Hackathon 2026.

## Getting Started

Read the full startup documentation in [`SETUP.md`](../SETUP.md).

```bash
# 1-click install & launch
cd airguard
bash run.sh
```

## Features

- **XGBoost Bias Correction**: Down-scales CAMS forecast using ERA5 historical weather data to fix regional blindspots.
- **Green Buffer AI**: Optimizes tree placement to shield residential corridors.
- **Real-time Alerting**: Debounced email notifications when danger scores exceed 5/10.
- **FastAPI Backend**: Provides structured JSON directly to the frontend map view.
