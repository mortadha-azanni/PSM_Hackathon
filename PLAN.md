That changes everything. 26 person-hours total. Here is the ruthless cut list and the only viable plan.

---

## What gets cut, no negotiation

**Drop Twilio SMS.** API registration, phone number provisioning, and testing eat 2 hours minimum. Replace with email-only alerts via SMTP or mock them visually in the UI.

**Drop Docker.** Running locally for the demo is fine. Docker adds 1–2 hours of debugging with zero demo value.

**Drop WebSocket.** Use `setInterval` polling every 30 seconds. Same visual result, 3 lines of code instead of 30.

**Drop the LSTM entirely.** You do not have time to debug training instability, sequence alignment bugs, or PyTorch dependency hell. Replace with XGBoost — trains in under 90 seconds on 3 years of daily tabular data, still defensible as a gradient-boosted residual corrector, and judges cannot tell the difference from a UI demo.

**Drop harpconvert and GEE entirely.** Open-Meteo Air Quality API gives you CAMS NO₂ for both cities, hourly, 5-day forecast, no API key, 3 lines of Python. This replaces the entire satellite pipeline for demo purposes.

**Drop live Green Buffer scoring.** Pre-compute the JSON offline before the hackathon starts, serve it as a static file.

---

## What remains — the irreducible core

A working map that shows NO₂ forecast heatmaps for Monastir and Mahdia, a danger score per zone, a 3-day forecast chart with confidence band, the weather status badges (inversion / stagnation / sirocco), the green buffer markers, an alert notification panel, and the compliance threshold bar. This is enough to win.

---

## 13-hour plan, 2 people---

## The three things that will kill you if you ignore them

**The data fetch at H1 is the single point of failure.** If Open-Meteo returns unexpected data shapes or the coordinate alignment between weather and NO₂ data is off by an index, everything downstream breaks. Person A must have working DataFrames for both cities before touching the model. Do not proceed to H3 if this is not clean.

**The API/frontend sync at H4 is the second point of failure.** Person B must be able to swap mock JSON for live API with a one-line URL change. Design the mock JSON to have the exact same schema as the real API response. Person A must match that schema exactly. Write it down before H1 starts so you are never blocked on each other.

**Do not let Person B touch the Green Buffer or Episode Replay before H10.** Both require data from Person A. If Person B starts these early using hardcoded placeholder data, the integration at H10 becomes a rewrite. Build left to right: data → API → UI, in that order.

---

## The model in 15 lines

This is what Person A actually trains. No framework instability, no GPU, no 150 epochs.

```python
import openmeteo_requests, pandas as pd
from xgboost import XGBRegressor

# Pull CAMS NO₂ + ERA5 weather for Monastir (repeat for Mahdia)
# openmeteo_requests handles the API call
# Features: cams_no2_lag1, cams_no2_lag7, blh, wind_speed,
#           temp_2m, inversion_flag, sirocco_flag, day_sin, day_cos
# Target: tropomi_no2 - cams_no2  (the local residual)

model = XGBRegressor(n_estimators=300, max_depth=5,
                     learning_rate=0.05, subsample=0.8)
model.fit(X_train, y_train)

# Forecast: CAMS gives +24h/+48h/+72h. Correct each with the model.
# Uncertainty: ±1.5 * std(y_val - model.predict(X_val))
```

This is the entire model. It is scientifically justified as a bias-correction scheme, it trains in 90 seconds, and it lets Person A move to the FastAPI endpoints by Hour 4.