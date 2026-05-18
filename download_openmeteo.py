"""Download extended OpenMeteo data for the wind farm location.

Adds extra meteorological variables not present in train_dataset.csv that
could help wind power forecasting (humidity, surface pressure, radiation,
soil temperature, etc.).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "openmeteo_extra.csv"

# Wind farm coordinates from data/dataset/README.md
LAT = 46.8268
LON = 38.7179

# Period: 2022-01-01 to 2026-03-31 (covers train + valid)
START = "2022-01-01"
END = "2026-03-31"

# Variables NOT in train_dataset.csv that could be useful for wind forecasting
EXTRA_VARS = [
    "relative_humidity_2m",   # affects air density
    "dew_point_2m",            # humidity proxy
    "surface_pressure",        # different from pressure_msl
    "cloud_cover_mid",         # we only have cloud_cover_low
    "cloud_cover_high",
    "shortwave_radiation",     # thermal effects, atmospheric mixing
    "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm",  # slow thermal mass
    "is_day",                  # day/night indicator
    "weather_code",            # categorical weather state
]


def fetch_archive(start: str, end: str) -> pd.DataFrame:
    """Use historical_forecast endpoint to keep same flavor as the original dataset."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(EXTRA_VARS),
        "timezone": "UTC",
    }
    print(f"GET {url} {start}..{end}")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        print("response keys:", list(data.keys()))
        print("error:", data)
        raise RuntimeError("no hourly data returned")
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def main():
    # Download in two chunks to avoid timeouts (historical_forecast has limits)
    chunks = []
    # Split into yearly chunks to be safe
    cur = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=365), end)
        try:
            df = fetch_archive(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            chunks.append(df)
            print(f"  -> {len(df)} rows")
        except Exception as e:
            print(f"  chunk failed: {e}")
            time.sleep(2)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(1)

    full = pd.concat(chunks, ignore_index=True).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    print(f"total rows: {len(full)}")
    full.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
    print(full.head())
    print(full.tail())


if __name__ == "__main__":
    main()
