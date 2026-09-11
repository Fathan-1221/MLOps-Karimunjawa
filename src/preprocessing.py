import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

CSV_RAW = os.path.join("data", "raw", "jepara_weather_log.csv")
CSV_PROCESSED = os.path.join("data", "processed", "jepara_weather_clean.csv")

# Threshold kelayakan 
WIND_SPEED_THRESHOLD = 30      
RAIN_THRESHOLD = 20           
PRESSURE_DROP_THRESHOLD = 5    


def load_raw_data(path: str = CSV_RAW) -> pd.DataFrame:
    """Load CSV mentah dan pastikan tipe data benar."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning: hapus duplikat, urutkan waktu, tangani missing value."""
    df = df.drop_duplicates(subset="timestamp").copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Interpolasi linear untuk missing value numerik (jika ada gap kecil)
    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed", "rain"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)

    # Drop baris yang masih ada NaN setelah interpolasi 
    before = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[Cleaning] {dropped} baris di-drop karena missing value tidak bisa diinterpolasi.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: rolling stats & indikator perubahan cuaca mendadak."""
    df = df.copy()

    # Penurunan tekanan dalam 3 jam terakhir (indikator badai)
    df["pressure_drop_3h"] = (df["pressure"].diff(periods=3) * -1).fillna(0).round(2)

    # Rolling max wind speed & rolling sum rain (window 3 jam)
    df["wind_speed_roll_max_3h"] = df["wind_speed"].rolling(window=3, min_periods=1).max().round(2)
    df["rain_roll_sum_3h"] = df["rain"].rolling(window=3, min_periods=1).sum().round(2)

    df["hour"] = df["timestamp"].dt.hour

    return df


def label_kelayakan(df: pd.DataFrame) -> pd.DataFrame:
    """Beri label biner kelayakan berdasarkan kombinasi threshold.

    1 = Layak, 0 = Tidak Layak.
    Tidak Layak jika salah satu kondisi berikut terpenuhi:
    - wind_speed melebihi ambang batas
    - rain melebihi ambang batas
    - penurunan tekanan tajam dalam 3 jam (indikasi badai)
    """
    df = df.copy()

    tidak_layak = (
        (df["wind_speed"] >= WIND_SPEED_THRESHOLD)
        | (df["rain"] >= RAIN_THRESHOLD)
        | (df["pressure_drop_3h"] >= PRESSURE_DROP_THRESHOLD)
    )

    df["kelayakan"] = np.where(tidak_layak, 0, 1)  # 0 = Tidak Layak, 1 = Layak
    df["kelayakan_label"] = df["kelayakan"].map({1: "Layak", 0: "Tidak Layak"})

    return df


def check_drift(baseline: pd.DataFrame, current: pd.DataFrame, columns=None, alpha: float = 0.05) -> dict:
    """Cek data drift antar dua periode menggunakan Kolmogorov-Smirnov test.

    Return dict berisi p-value & status drift per kolom.
    """
    if columns is None:
        columns = ["temperature", "humidity", "pressure", "wind_speed", "rain"]

    results = {}
    for col in columns:
        stat, p_value = ks_2samp(baseline[col].dropna(), current[col].dropna())
        results[col] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": bool(p_value < alpha),
        }
    return results


def run_pipeline(raw_path: str = CSV_RAW, processed_path: str = CSV_PROCESSED) -> pd.DataFrame:
    """Jalankan seluruh pipeline preprocessing end-to-end."""
    print("Memuat data mentah(raw)")
    df = load_raw_data(raw_path)

    print("Membersihkan data")
    df = clean_data(df)

    print("Feature engineering")
    df = engineer_features(df)

    print("Labeling kelayakan")
    df = label_kelayakan(df)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    n_layak = (df["kelayakan"] == 1).sum()
    n_tidak_layak = (df["kelayakan"] == 0).sum()
    print(
        f"Selesai. {len(df)} baris disimpan ke {processed_path} "
        f"(Layak: {n_layak}, Tidak Layak: {n_tidak_layak})"
    )

    return df


if __name__ == "__main__":
    run_pipeline()