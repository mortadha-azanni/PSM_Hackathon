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

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

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


# Named landmarks for proximity lookup (used when OSM returns real elements)
_LANDMARKS = {
    "monastir": [
        (35.7097, 10.7713, "Ksar Hellal industrial zone"),
        (35.6900, 10.7600, "Moknine industrial area"),
        (35.7200, 10.7500, "Jemmal ceramics cluster"),
        (35.7050, 10.8000, "Zeramdine corridor"),
        (35.7920, 10.8260, "Monastir city center"),
        (35.8200, 10.8400, "Bembla coastal zone"),
        (35.7500, 10.7800, "Sahline agricultural area"),
    ],
    "mahdia": [
        (35.2300, 11.1100, "Chebba port zone"),
        (35.2980, 10.7070, "El Jem industrial"),
        (35.1650, 10.9400, "Ksour Essef corridor"),
        (35.2950, 10.9450, "Bou Merdes zone"),
        (35.5047, 11.0622, "Mahdia coastal"),
        (35.4050, 11.0100, "Sidi Alouane area"),
        (35.2700, 10.9000, "Souassi valley"),
    ],
}


def nearest_location_name(lat: float, lon: float, city_key: str) -> str:
    """Return the name of the closest known landmark."""
    landmarks = _LANDMARKS.get(city_key, [])
    if not landmarks:
        return "Unknown area"
    best_name = "Unknown area"
    best_dist = float("inf")
    for (llt, lln, name) in landmarks:
        d = math.sqrt((lat - llt) ** 2 + (lon - lln) ** 2)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def fetch_osm_land_use(bbox: dict) -> list:
    """Query Overpass for industrial, farmland, and open land in bbox."""
    query = f"""
    [out:json][timeout:30];
    (
      way["landuse"="industrial"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="farmland"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="grass"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["natural"="scrub"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="meadow"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="orchard"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
      way["landuse"="forest"]({bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']});
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


def classify_buffer_type(lat, lon, city_center_lat, city_center_lon, city_key) -> str:
    dist_km = math.sqrt((lat - city_center_lat)**2 + (lon - city_center_lon)**2) * 111
    # Rough coastal coordinate for Mahdia (close to lon 11.06)
    if city_key == "mahdia" and abs(lon - 11.06) * 111 < 1.5:
        return "coastal_windbreak"
    elif dist_km < 1.5:
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
        buf_type = classify_buffer_type(lat, lon, city["lat"], city["lon"], city_key)
        spec = SPECIES_BY_TYPE[buf_type]
        location_name = nearest_location_name(lat, lon, city_key)
        candidates.append({
            "lat":           round(lat, 5),
            "lon":           round(lon, 5),
            "location_name": location_name,
            "buffer_score":  score,
            "buffer_type":   buf_type,
            "land_use":      land_type,
            "recommended_species": spec["species"],
            "buffer_width_m":      spec["width_m"],
            "estimated_no2_reduction_pct": spec["no2_reduction_pct"],
            "area_needed_m2": 5000 if buf_type == "industrial_perimeter" else 2000,
        })

    # Fallback: generate named candidates if OSM returned nothing
    if not candidates:
        print("    Using named fallback candidates")
        # Monastir center: 35.7643, 10.8113
        # All offsets verified against absolute GPS coordinates of each town
        # dlat = target_lat - 35.7643
        # dlon = target_lon - 10.8113

        # Mahdia center: 35.5047, 11.0622
        # dlat = target_lat - 35.5047
        # dlon = target_lon - 11.0622

        NAMED_FALLBACK = {
            "monastir": [
                # Ksar Hellal (35.650, 10.883) — main textile cluster, east of Monastir
                (-0.1143, +0.0717, "scrub",    "industrial_perimeter", "Ksar Hellal industrial zone"),
                (-0.1163, +0.0637, "scrub",    "industrial_perimeter", "Ksar Hellal textile cluster west edge"),

                # Moknine (35.633, 10.900) — south-east, second industrial hub
                (-0.1310, +0.0887, "farmland", "industrial_perimeter", "Moknine industrial perimeter"),
                (-0.1293, +0.0737, "scrub",    "road_buffer",          "Moknine road corridor RN82"),

                # Jemmal (35.621, 10.726) — south, inland
                (-0.1433, -0.0853, "grass",    "road_buffer",          "Jemmal bypass corridor GP1"),
                (-0.1453, -0.0793, "grass",    "road_buffer",          "Jemmal ring road fringe"),

                # Zeramdine (35.690, 10.749) — west corridor
                (-0.0743, -0.0623, "meadow",   "residential_corridor", "Zeramdine residential fringe"),
                (-0.0763, -0.0683, "farmland", "residential_corridor", "Zeramdine south agricultural zone"),

                # Bembla (35.700, 10.800) — north, main industrial satellite per Wikipedia
                (-0.0643, -0.0113, "scrub",    "industrial_perimeter", "Bembla industrial zone"),
                (-0.0623, -0.0053, "grass",    "road_buffer",          "Bembla coastal access road"),

                # Teboulba (35.660, 10.973) — east coast, fishing + small industry
                (-0.1043, +0.1617, "meadow",   "coastal_windbreak",    "Teboulba coastal windbreak"),

                # Sahline (35.742, 10.782) — agricultural fringe NW of Monastir
                (-0.0223, -0.0293, "farmland", "residential_corridor", "Sahline agricultural fringe"),

                # Monastir coastal strip (north of center)
                (+0.0157, +0.0437, "meadow",   "coastal_windbreak",    "Monastir north coastal strip"),

                # RN1 / GP1 highway buffer (between Monastir and Jemmal)
                (-0.0843, -0.0513, "grass",    "road_buffer",          "GP1 highway buffer Monastir–Jemmal"),

                # Ksar Hellal–Moknine inter-zone buffer (between the two clusters)
                (-0.1220, +0.0760, "scrub",    "road_buffer",          "Ksar Hellal–Moknine inter-zone road"),
            ],

            "mahdia": [
                # Bou Merdes (35.456, 11.016) — closest industrial zone north of Mahdia
                (-0.0487, -0.0462, "scrub",    "industrial_perimeter", "Bou Merdes industrial perimeter"),
                (-0.0467, -0.0392, "grass",    "road_buffer",          "Bou Merdes bypass road"),

                # Sidi Alouane (35.481, 10.990) — inland commune
                (-0.0237, -0.0722, "farmland", "residential_corridor", "Sidi Alouane agricultural zone"),
                (-0.0217, -0.0662, "grass",    "road_buffer",          "Sidi Alouane road corridor"),

                # Ksour Essaf (35.418, 10.995) — southern industrial hub
                (-0.0867, -0.0672, "scrub",    "industrial_perimeter", "Ksour Essaf industrial perimeter"),
                (-0.0847, -0.0612, "grass",    "road_buffer",          "GP1 Ksour Essaf buffer"),

                # El Jem (35.296, 10.707) — cement + construction, far south
                (-0.2087, -0.3552, "scrub",    "industrial_perimeter", "El Jem south industrial zone"),
                (-0.2067, -0.3492, "farmland", "residential_corridor", "El Jem residential fringe"),

                # Chebba (35.237, 11.115) — port + fishing, far south-east
                (-0.2677, +0.0528, "scrub",    "industrial_perimeter", "Chebba port industrial zone"),
                (-0.2657, +0.0458, "grass",    "road_buffer",          "Chebba port access road"),

                # Souassi (35.428, 10.979) — olive belt, central governorate
                (-0.0767, -0.0832, "farmland", "residential_corridor", "Souassi agricultural corridor"),

                # Chorbane (35.410, 10.992)
                (-0.0947, -0.0702, "scrub",    "industrial_perimeter", "Chorbane perimeter buffer"),

                # Mahdia coast and Cap Afrique
                (+0.0053, +0.0128, "meadow",   "coastal_windbreak",    "Mahdia coastal windbreak"),
                (+0.0203, +0.0178, "scrub",    "coastal_windbreak",    "Cap Afrique shoreline"),

                # GP1 north of Mahdia city
                (+0.0353, -0.0122, "grass",    "road_buffer",          "GP1 Mahdia north ring road"),
                (-0.0047, -0.0122, "scrub",    "road_buffer",          "GP1 Mahdia city buffer"),

                # Hiboun / Salakta coast (35.468, 11.033)
                (-0.0367, -0.0292, "meadow",   "coastal_windbreak",    "Salakta–Hiboun coastal strip"),

                # Central olive belt
                (-0.0607, -0.0372, "farmland", "residential_corridor", "Mahdia central olive belt"),
                (-0.0307, -0.0822, "meadow",   "residential_corridor", "Mahdia inland olive corridor"),
            ],
        }
        for (dlat, dlon, ltype, forced_type, loc_name) in NAMED_FALLBACK[city_key]:
            lat = round(city["lat"] + dlat, 5)
            lon = round(city["lon"] + dlon, 5)
            score = score_buffer_candidate(lat, lon, ltype, city["lat"], city["lon"])
            spec = SPECIES_BY_TYPE[forced_type]
            # In the fallback loop, change the append to include name:
            candidates.append({
                "lat": lat, "lon": lon,
                "buffer_score": score,
                "buffer_type":  forced_type,
                "land_use":     ltype,
                "location_name": loc_name,          # ADD THIS
                "recommended_species": spec["species"],
                "buffer_width_m": spec["width_m"],
                "estimated_no2_reduction_pct": spec["no2_reduction_pct"],
                "area_needed_m2": 4500 if forced_type == "industrial_perimeter" else 2200,
            })

    # Sort by score descending, take top 10
    candidates.sort(key=lambda x: x["buffer_score"], reverse=True)
    top = candidates[:10]
    print(f"    {len(candidates)} candidates → top {len(top)}, best score: {top[0]['buffer_score'] if top else 0}/100")

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
