import argparse
import os
import time
import zoneinfo
from datetime import datetime

import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"
PARAMS_BASE = {
    "latitude": -6.5924,
    "longitude": 110.671,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain",
    "timezone": "Asia/Jakarta",
}
RAW_DIR = os.path.join("data", "raw")
TZ = zoneinfo.ZoneInfo("Asia/Jakarta")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def fetch_weather(days_back: int) -> dict:
    """Panggil API dengan retry. Return dict 'hourly' dari respons."""
    params = {**PARAMS_BASE, "past_days": days_back, "forecast_days": 1}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()["hourly"]
        except (requests.RequestException, KeyError, ValueError) as err:
            print(f"[Percobaan {attempt}/{MAX_RETRIES}] Gagal: {err}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def ingest_once(days_back: int = 7) -> str:
    """Ambil data sekali, simpan sebagai snapshot baru. Return path file."""
    hourly = fetch_weather(days_back)
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "rain": hourly["rain"],
    })

    now = datetime.now(TZ)
    df = df[pd.to_datetime(df["timestamp"]) <= now.replace(tzinfo=None)]

    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"weather_{now:%Y%m%d_%H%M%S}.csv")
    df.to_csv(path, index=False)
    print(f"[{now:%Y-%m-%d %H:%M:%S}] {len(df)} baris disimpan ke {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Ingestion data cuaca Jepara")
    parser.add_argument("--days", type=int, default=7, help="hari ke belakang")
    parser.add_argument("--runs", type=int, default=1, help="jumlah pengambilan")
    parser.add_argument("--every", type=int, default=3600,
                        help="jeda antar pengambilan (detik)")
    args = parser.parse_args()

    for i in range(args.runs):
        ingest_once(args.days)
        if i < args.runs - 1:
            print(f"Menunggu {args.every} detik...")
            time.sleep(args.every)


if __name__ == "__main__":
    main()