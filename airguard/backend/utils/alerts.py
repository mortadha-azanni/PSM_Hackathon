"""
Alert engine — checks danger scores and sends email notifications.
Debounced: same zone won't trigger more than once per 6 hours.
"""

import os, json, smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ALERT_LOG = Path(__file__).parent.parent / "data" / "alerts.json"
DEBOUNCE_H = 6  # minimum hours between repeat alerts for same zone

LEVEL_MAP = {
    (5, 6): ("ADVISORY",   "Review production schedule; minimize peak emissions 08:00–12:00"),
    (7, 8): ("WARNING",    "Reschedule high-emission processes; activate scrubbers"),
    (9, 10):("EMERGENCY",  "Suspend all non-essential combustion; coordinate with ANPE"),
}


def _get_alert_level(danger_score: int) -> tuple[str, str]:
    for (lo, hi), (level, action) in LEVEL_MAP.items():
        if lo <= danger_score <= hi:
            return level, action
    return "SAFE", "No action required"


def _load_log() -> dict:
    if ALERT_LOG.exists():
        with open(ALERT_LOG) as f:
            return json.load(f)
    return {}


def _save_log(log: dict):
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


def _is_debounced(zone_id: str, log: dict) -> bool:
    last = log.get(zone_id, {}).get("last_sent")
    if not last:
        return False
    delta = datetime.utcnow() - datetime.fromisoformat(last)
    return delta < timedelta(hours=DEBOUNCE_H)


def _send_email(subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    recipient = os.getenv("ALERT_RECIPIENT", smtp_user)

    if not smtp_user or not smtp_pass:
        print(f"  ⚠ Email not configured — alert logged only: {subject}")
        return

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"  ✓ Alert email sent: {subject}")
    except Exception as e:
        print(f"  ✗ Email failed: {e}")


def check_and_trigger(forecast_payload: dict) -> list[dict]:
    """
    Receive a forecast payload from predict_city().
    Fire alerts for any zone with danger_score >= 5.
    Returns list of triggered alerts (empty if none).
    """
    city      = forecast_payload["city"]
    horizon_h = forecast_payload["horizon_hours"]
    wx        = forecast_payload["weather_summary"]
    log       = _load_log()
    triggered = []

    # Find worst zone (highest danger score)
    cells = forecast_payload.get("grid_cells", [])
    if not cells:
        return []

    max_cell = max(cells, key=lambda c: c["danger_score"])
    ds = max_cell["danger_score"]

    if ds < 5:
        return []

    zone_id   = f"{city}_{max_cell['lat']:.3f}_{max_cell['lon']:.3f}"
    level, action = _get_alert_level(ds)

    if _is_debounced(zone_id, log):
        return []

    now = datetime.utcnow()
    alert = {
        "zone_id":        zone_id,
        "city":           city,
        "danger_score":   ds,
        "alert_level":    level,
        "forecast_horizon_h": horizon_h,
        "no2_predicted":  max_cell["no2_predicted"],
        "inversion_flag": max_cell["inversion_flag"],
        "stagnation_flag":max_cell["stagnation_flag"],
        "action":         action,
        "wind_speed_ms":  wx["wind_speed_ms"],
        "blh_m":          wx["boundary_layer_height_m"],
        "timestamp":      now.isoformat(),
    }

    subject = f"🔴 AirGuard TN — {level} | {city.capitalize()} | DS={ds}/10"
    body = f"""
AirGuard TN — {level}
{'=' * 45}
City:           {city.capitalize()}
Zone:           {zone_id}
Danger Score:   {ds}/10
Forecast:       +{horizon_h}h from {now.strftime('%Y-%m-%d %H:%M')} UTC

Predicted NO₂:  {max_cell['no2_predicted']} μg/m³
(WHO Daily Limit: 25 μg/m³)

Conditions:
  Wind speed:          {wx['wind_speed_ms']} m/s
  Boundary layer:      {wx['boundary_layer_height_m']} m
  Thermal inversion:   {'YES' if wx['inversion_detected'] else 'No'}
  Sirocco event:       {'YES' if wx['sirocco_active'] else 'No'}

REQUIRED ACTIONS:
{action}

Contact: ANPE Environmental Desk
{'=' * 45}
    """.strip()

    _send_email(subject, body)

    # Update log
    log[zone_id] = {"last_sent": now.isoformat(), "last_score": ds}
    _save_log(log)

    triggered.append(alert)
    return triggered


def get_recent_alerts(limit: int = 20) -> list[dict]:
    """Return the last N alert records from the log."""
    log = _load_log()
    alerts = [
        {"zone_id": k, **v}
        for k, v in sorted(log.items(),
                            key=lambda x: x[1].get("last_sent", ""),
                            reverse=True)
    ]
    return alerts[:limit]
