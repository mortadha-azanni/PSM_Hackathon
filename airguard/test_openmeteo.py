import requests_cache
import openmeteo_requests
from retry_requests import retry
import pandas as pd

cache = requests_cache.CachedSession('.cache', expire_after=3600)
session = retry(cache, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=session)

params = {
    "latitude": 35.7643,
    "longitude": 10.8113,
    "hourly": "nitrogen_dioxide",
    "start_date": "2024-01-01",
    "end_date": "2024-01-05",
}

try:
    responses = om.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params=params)
    r = responses[0].Hourly()
    
    print("Success:")
    print("Variables:", r.Variables(0).ValuesAsNumpy()[:5])
except Exception as e:
    print("Error:", e)
