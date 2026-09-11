"""Modul pengambilan data cuaca maritim dari Open-Meteo API.

Berisi fungsi bootstrap (fetch_historical_data) dan
incremental update per jam (fetch_hourly_data).
"""

from datetime import datetime
import os
import time
import zoneinfo

import pandas as pd
import requests

# Konfigurasi API Open-Meteo
URL = "https://api.open-meteo.com/v1/forecast"
PARAMS_BASE = {
    "latitude": -6.5924,
    "longitude": 110.671,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain",
    "timezone": "Asia/Jakarta",
}

# Lokasi File Penyimpanan
CSV_FILE = os.path.join("data", "raw", "jepara_weather_log.csv")

MAX_LAG_HOURS = 3
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30


def _check_data_freshness(hourly_times, now_jakarta, tz_jakarta):
    """Cek apakah data terakhir dari API terlalu jauh dari waktu sekarang.

    Return True kalau data dianggap segar (lag <= MAX_LAG_HOURS), False kalau
    kemungkinan model cuaca belum update.
    """
    last_time_dt = pd.to_datetime(hourly_times[-1]).tz_localize(tz_jakarta)
    lag_hours = (now_jakarta - last_time_dt).total_seconds() / 3600

    if lag_hours > MAX_LAG_HOURS:
        print(
            f"[PERINGATAN] Data terbaru dari API adalah {hourly_times[-1]}, "
            f"tertinggal {lag_hours:.1f} jam dari waktu sekarang "
            f"({now_jakarta.strftime('%Y-%m-%d %H:%M:%S')}). Kemungkinan model "
            "cuaca upstream belum update."
        )
        return False

    return True


def fetch_historical_data(days_back: int = 30) -> None:
    """Bootstrap data historis ke CSV_FILE."""
    print(f"Mengambil data historis {days_back} hari ke belakang...")

    tz_jakarta = zoneinfo.ZoneInfo("Asia/Jakarta")
    params = PARAMS_BASE.copy()
    params["past_days"] = days_back

    df = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(URL, params=params, timeout=10)
        data = response.json()
        hourly = data["hourly"]

        now_jakarta = datetime.now(tz_jakarta)
        is_fresh = _check_data_freshness(hourly["time"], now_jakarta, tz_jakarta)

        if is_fresh or attempt == MAX_RETRIES:
            df = pd.DataFrame({
                "timestamp": hourly["time"],
                "temperature": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "pressure": hourly["surface_pressure"],
                "wind_speed": hourly["wind_speed_10m"],
                "rain": hourly["rain"],
            })
            if not is_fresh:
                print(
                    f"Percobaan ke-{attempt} tetap tertinggal, lanjut simpan data "
                    "yang ada (bisa di-run ulang nanti untuk melengkapi)."
                )
            break

        print(
            f"Percobaan ke-{attempt}/{MAX_RETRIES} gagal dapat data segar, "
            f"coba lagi dalam {RETRY_DELAY_SECONDS} detik..."
        )
        time.sleep(RETRY_DELAY_SECONDS)

    now_jakarta = datetime.now(tz_jakarta)

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(tz_jakarta)
    df = df[df["timestamp_dt"] <= now_jakarta].copy()
    df.drop(columns=["timestamp_dt"], inplace=True)

    # Simpan ke CSV
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df.to_csv(CSV_FILE, index=False)

    time_str = now_jakarta.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_str}] Berhasil menyimpan {len(df)} baris data historis ke {CSV_FILE}")


def fetch_hourly_data() -> None:
    """Mengambil 1 baris data jam saat ini (Incremental Update)."""
    response = requests.get(URL, params=PARAMS_BASE, timeout=10)
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "rain": hourly["rain"],
    })

    # Format string jam saat ini (YYYY-MM-DDTHH:00)
    tz_jakarta = zoneinfo.ZoneInfo("Asia/Jakarta")
    now_jakarta = datetime.now(tz_jakarta)
    current_time_str = now_jakarta.strftime("%Y-%m-%dT%H:00")

    # Cari baris yang sesuai dengan jam sekarang
    df_current = df[df["timestamp"] == current_time_str]

    if df_current.empty:
        print(
            f"[{now_jakarta}] Data untuk jam {current_time_str} belum tersedia "
            "di API (model kemungkinan belum update), pakai data terakhir yang ada."
        )
        df_current = df.head(1)

    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0

    # Cek duplikasi
    if file_exists:
        existing_df = pd.read_csv(CSV_FILE)
        if (
            not existing_df.empty
            and df_current["timestamp"].iloc[0] in existing_df["timestamp"].values
        ):
            print(f"[{now_jakarta}] Data timestamp tersebut sudah ada di CSV.")
            return

    df_current.to_csv(CSV_FILE, mode="a", header=not file_exists, index=False)
    print(f"[{now_jakarta}] Berhasil menambahkan 1 record baru ke {CSV_FILE}")


if __name__ == "__main__":
    fetch_historical_data(days_back=30)