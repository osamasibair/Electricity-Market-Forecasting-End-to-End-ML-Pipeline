"""Fetching raw demand and weather data from public APIs"""

import time
from pathlib import Path
import pandas as pd
import requests

raw = Path("data/raw")
elexon = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
weather = "https://archive-api.open-meteo.com/v1/archive"

def fetch_demand(start, end):
    frames = []
    for period_start in pd.date_range(start, end, freq="7D"):
        period_end = min(period_start + pd.offsets.Day(6), pd.Timestamp(end))
        dates = {
            "settlementDateFrom": period_start.strftime("%Y-%m-%d"),
            "settlementDateTo": period_end.strftime("%Y-%m-%d"),
                }
        response = requests.get(elexon, params=dates, timeout=30)
        response.raise_for_status()
        frames.append(pd.DataFrame(response.json()["data"]))
        print(f"demand: {period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}")
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True)

def fetch_weather(start, end, lat=51.5072, lon=-0.1276):
    query = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": "UTC",
    }
    response = requests.get(weather, params=query, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])

if __name__ == "__main__":
    start, end = "2023-01-01", "2026-09-01"
    raw.mkdir(parents=True, exist_ok=True)
    fetch_demand(start, end).to_csv(raw / "demand.csv", index=False)
    fetch_weather(start, end).to_csv(raw / "weather.csv", index=False)
    print("saved to data/raw")





