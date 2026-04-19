"""
Pre-computes green buffer placement scores for both cities.
Run once before starting the server.

Uses OSM Overpass API to get land use polygons.
Scores each candidate cell using the buffer placement algorithm.
Saves static JSON to frontend/data/ so the API serves it instantly.

Usage:
    python backend/data/compute_buffers.py
"""

import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import requests
import numpy as np
from pathlib import Path
from config.cities import CITIES

OUT_DIR = Path(__file__).parent.parent.parent / "frontend" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

SPECIES_BY_TYPE = {
    "industrial_perimeter": {
        "species": ["Pinus halepensis", "Eucalyptus globulus"],
        "width_m": "20–50",
        "no2_reduction_pct": "25–40",
    },
    "road_buffer": {
        "species": ["Nerium oleander", "Populus nigra"],
        "width_m": "5–10",
        "no2_reduction_pct": "15–25",
    },
    "residential_corridor": {
        "species": ["Ficus nitida", "Mixed shrubs"],
        "width_m": "10–20",
        "no2_reduction_pct": "10–18",
    },
    "coastal_windbreak": {
        "species": ["Tamarix africana", "Acacia saligna"],
        "width_m": "30–80",
        "no2_reduction_pct": "wind + spray reduction",
    },
}


def fetch_osm_land_use(bbox: dict) -> list:
    """Query Overpass for industrial, farmland, and open land in bbox."""
    query = f"""
    [out:json][timeout:25];
    (
      way["landuse"="industrial"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="farmland"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="grass"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["natural"="scrub"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="meadow"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
    );
    out center;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        print(f"    OSM returned {len(elements)} land use elements")
        return elements
    except Exception as e:
        print(f"    ⚠ OSM query failed: {e} — using grid fallback")
        return []


def score_buffer_candidate(lat, lon, land_type, city_center_lat, city_center_lon,
                            prevailing_wind_dir_deg=220) -> int:
    """
    Score a candidate buffer location 0–100.
    Prevailing winter wind in Monastir–Mahdia: SW (~220°) → pollution moves NE.
    Best buffers are placed upwind (SW side) of industrial zones.
    """
    # 1. Upwind position (max 30 pts)
    # Vector from candidate to city center
    dlat = city_center_lat - lat
    dlon = city_center_lon - lon
    bearing = math.degrees(math.atan2(dlon, dlat)) % 360
    wind_alignment = math.cos(math.radians(bearing - prevailing_wind_dir_deg))
    upwind_score = max(0, wind_alignment) * 30

    # 2. Land availability (max 25 pts)
    land_scores = {
        "scrub": 25, "grass": 22, "meadow": 20,
        "farmland": 15, "industrial": 5,
    }
    land_score = land_scores.get(land_type, 12)

    # 3. Proximity to city center / industrial zone (max 25 pts)
    dist_km = math.sqrt(dlat**2 + dlon**2) * 111  # rough km conversion
    proximity_score = max(0, 25 - dist_km * 3)

    # 4. Coastal proximity bonus for Mahdia (max 20 pts)
    coastal_score = max(0, 20 - abs(lon - 11.06) * 100)

    total = upwind_score + land_score + proximity_score + coastal_score
    return min(100, int(total))


def classify_buffer_type(lat, lon, city_center_lat, city_center_lon) -> str:
    dist_km = math.sqrt((lat - city_center_lat)**2 + (lon - city_center_lon)**2) * 111
    if dist_km < 1.5:
        return "industrial_perimeter"
    elif abs(lon - city_center_lon) * 111 < 2:
        return "road_buffer"
    else:
        return "residential_corridor"


def compute_buffers_for_city(city_key: str) -> dict:
    city = CITIES[city_key]
    bbox = city["bbox"]
    print(f"\n[{city_key.upper()}] Computing green buffers...")

    elements = fetch_osm_land_use(bbox)

    candidates = []

    # From OSM elements
    for el in elements:
        center = el.get("center", {})
        if not center:
            continue
        lat = center.get("lat")
        lon = center.get("lon")
        if not lat or not lon:
            continue
        land_type = el.get("tags", {}).get("landuse") or el.get("tags", {}).get("natural", "unknown")
        score = score_buffer_candidate(lat, lon, land_type, city["lat"], city["lon"])
        buf_type = classify_buffer_type(lat, lon, city["lat"], city["lon"])
        spec = SPECIES_BY_TYPE[buf_type]
        candidates.append({
            "lat":           round(lat, 5),
            "lon":           round(lon, 5),
            "buffer_score":  score,
            "buffer_type":   buf_type,
            "land_use":      land_type,
            "recommended_species": spec["species"],
            "buffer_width_m":      spec["width_m"],
            "estimated_no2_reduction_pct": spec["no2_reduction_pct"],
            "area_needed_m2": 5000 if buf_type == "industrial_perimeter" else 2000,
        })

    # Fallback: generate grid candidates if OSM returned nothing
    if not candidates:
        print("    Using fallback grid candidates")
        offsets = [(-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01),
                   (-0.015, -0.01), (0.015, 0.01), (-0.008, 0.012)]
        land_types = ["scrub", "grass", "meadow", "farmland", "scrub", "grass", "meadow"]
        for (dlat, dlon), ltype in zip(offsets, land_types):
            lat = round(city["lat"] + dlat, 5)
            lon = round(city["lon"] + dlon, 5)
            score = score_buffer_candidate(lat, lon, ltype, city["lat"], city["lon"])
            buf_type = classify_buffer_type(lat, lon, city["lat"], city["lon"])
            spec = SPECIES_BY_TYPE[buf_type]
            candidates.append({
                "lat": lat, "lon": lon,
                "buffer_score": score,
                "buffer_type":  buf_type,
                "land_use":     ltype,
                "recommended_species": spec["species"],
                "buffer_width_m": spec["width_m"],
                "estimated_no2_reduction_pct": spec["no2_reduction_pct"],
                "area_needed_m2": 4000,
            })

    # Sort by score descending, take top 10
    candidates.sort(key=lambda x: x["buffer_score"], reverse=True)
    top = candidates[:10]
    print(f"    Top score: {top[0]['buffer_score'] if top else 0}/100")

    return {
        "city":             city_key,
        "city_label":       city["label"],
        "recommendations":  top,
        "total_candidates": len(candidates),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("AirGuard TN — Green Buffer Pre-compute")
    print("=" * 55)
    for city in ["monastir", "mahdia"]:
        result = compute_buffers_for_city(city)
        out = OUT_DIR / f"green_buffers_{city}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Saved → {out}")
    print("\n✓ Green buffer data ready")
