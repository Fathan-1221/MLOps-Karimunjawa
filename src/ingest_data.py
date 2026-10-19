import argparse
import os
import time
import zoneinfo
from datetime import datetime, timedelta

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS_BASE = {
    "latitude": -6.5924,
    "longitude": 110.671,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain",
    "timezone": "Asia/Jakarta",
}
RAW_DIR = os.path.join("data", "raw")
TZ = zoneinfo.ZoneInfo("Asia/Jakarta")
NUM_COLS = ["temperature", "humidity", "pressure", "wind_speed", "rain"]
ARCHIVE_LAG_DAYS = 6  # ERA5 tersedia dengan jeda sekitar 5 hari
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def request_hourly(url: str, params: dict) -> dict:
    """Panggil API dengan retry. Return dict 'hourly' dari respons."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()["hourly"]
        except (requests.RequestException, KeyError, ValueError) as err:
            print(f"[Percobaan {attempt}/{MAX_RETRIES}] Gagal: {err}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def fetch_forecast(days_back: int) -> dict:
    params = {**PARAMS_BASE, "past_days": days_back, "forecast_days": 1}
    return request_hourly(FORECAST_URL, params)


def fetch_archive(start: str, end: str) -> dict:
    params = {**PARAMS_BASE, "start_date": start, "end_date": end}
    return request_hourly(ARCHIVE_URL, params)


def ingest_once(source: str = "forecast", days_back: int = 7,
                start: str = None, end: str = None) -> str:
    """Ambil data sekali, simpan sebagai snapshot baru. Return path file."""
    if source == "archive":
        hourly = fetch_archive(start, end)
    else:
        hourly = fetch_forecast(days_back)

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

    before = len(df)
    df = df.dropna(subset=NUM_COLS, how="all")
    if len(df) < before:
        print(f"{before - len(df)} baris kosong dari API dibuang")
    if df.empty:
        raise SystemExit("API tidak mengembalikan data untuk rentang tersebut.")

    os.makedirs(RAW_DIR, exist_ok=True)
    suffix = "_archive" if source == "archive" else ""
    path = os.path.join(RAW_DIR, f"weather_{now:%Y%m%d_%H%M%S}{suffix}.csv")
    df.to_csv(path, index=False)
    print(f"[{now:%Y-%m-%d %H:%M:%S}] {len(df)} baris ({df['timestamp'].iloc[0]} "
          f"s.d. {df['timestamp'].iloc[-1]}) disimpan ke {path}")
    return path


def resolve_archive_range(start: str, end: str):
    """Validasi rentang tanggal archive. End dibatasi sampai batas ketersediaan ERA5."""
    if not start:
        raise SystemExit("Mode archive butuh --start YYYY-MM-DD (contoh: --start 2025-12-01).")
    latest = (datetime.now(TZ) - timedelta(days=ARCHIVE_LAG_DAYS)).date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else latest
    if end_date > latest:
        print(f"[Peringatan] --end dibatasi ke {latest} (ERA5 punya jeda ~5 hari).")
        end_date = latest
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    if start_date > end_date:
        raise SystemExit("--start harus lebih awal dari --end.")
    return start_date.isoformat(), end_date.isoformat()


def main():
    parser = argparse.ArgumentParser(description="Ingestion data cuaca Jepara")
    parser.add_argument("--source", choices=["forecast", "archive"], default="forecast",
                        help="forecast = data terbaru, archive = data historis ERA5")
    parser.add_argument("--days", type=int, default=7, help="[forecast] hari ke belakang")
    parser.add_argument("--start", help="[archive] tanggal mulai YYYY-MM-DD")
    parser.add_argument("--end", help="[archive] tanggal akhir YYYY-MM-DD (default: batas terbaru)")
    parser.add_argument("--runs", type=int, default=1, help="jumlah pengambilan")
    parser.add_argument("--every", type=int, default=3600,
                        help="jeda antar pengambilan (detik)")
    args = parser.parse_args()

    start, end = None, None
    if args.source == "archive":
        start, end = resolve_archive_range(args.start, args.end)

    for i in range(args.runs):
        ingest_once(args.source, args.days, start, end)
        if i < args.runs - 1:
            print(f"Menunggu {args.every} detik...")
            time.sleep(args.every)


if __name__ == "__main__":
    main()