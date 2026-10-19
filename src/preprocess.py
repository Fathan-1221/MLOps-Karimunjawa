import pandas as pd
import os
import glob

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/jepara_weather_clean.csv"

NUMERIC_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rain"
]

WIND_SPEED_THRESHOLD = 18      # km/jam
RAIN_THRESHOLD = 5            # mm/jam
PRESSURE_DROP_THRESHOLD = 3    # hPa dalam 3 jam


def load_data():

    files = sorted(glob.glob(os.path.join(RAW_DIR, "weather_*.csv")))

    if not files:
        raise FileNotFoundError(
            "Tidak ditemukan file data cuaca di data/raw/"
        )

    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    before = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS, how="all")
    if len(df) < before:
        print(f"{before - len(df)} baris kosong dibuang")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def clean_data(df):
    """Membersihkan data cuaca."""

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].interpolate(
        method="linear",
        limit=3
    )
    df = df.dropna().reset_index(drop=True)

    return df


def feature_engineering(df):
    segment = (df["timestamp"].diff() != pd.Timedelta(hours=1)).cumsum()
    by_segment = df.groupby(segment)

    # Perubahan tekanan udara selama 3 jam (positif = tekanan turun)
    df["pressure_drop_3h"] = (
        by_segment["pressure"].shift(3) - df["pressure"]
    ).round(2)

    # Kecepatan angin maksimum dalam 3 jam
    df["wind_speed_roll_max_3h"] = by_segment["wind_speed"].transform(
        lambda s: s.rolling(window=3).max()
    ).round(2)

    # Total curah hujan dalam 3 jam
    df["rain_roll_sum_3h"] = by_segment["rain"].transform(
        lambda s: s.rolling(window=3).sum()
    ).round(2)

    # Jam pengamatan
    df["hour"] = df["timestamp"].dt.hour

    return df


def create_label(df):
    """Membuat label kelayakan penyeberangan."""

    risk_condition = (
        (df["wind_speed"] >= WIND_SPEED_THRESHOLD) |
        (df["rain"] >= RAIN_THRESHOLD) |
        (df["pressure_drop_3h"] >= PRESSURE_DROP_THRESHOLD)
    )

    # 1 = Layak, 0 = Tidak Layak
    df["kelayakan_label"] = (~risk_condition).astype(int)

    df["kelayakan"] = df["kelayakan_label"].map({
        1: "Layak",
        0: "Tidak Layak"
    })

    return df


def save_data(df):
    """Menyimpan data hasil preprocessing."""

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("Data preprocessing berhasil.")
    print(f"Data tersimpan di: {PROCESSED_FILE}")
    print(f"Total data: {len(df)} baris")
    print(df["kelayakan"].value_counts().to_string())


def main():
    print("Memulai preprocessing...")

    # 1. Load seluruh snapshot
    df = load_data()
    print(f"Data setelah digabungkan: {len(df)} baris")

    # 2. Cleaning
    df = clean_data(df)
    print(f"Data setelah cleaning: {len(df)} baris")

    # 3. Feature engineering
    df = feature_engineering(df)

    # 4. Membuat label
    df = create_label(df)

    # 5. Hapus baris awal yang belum memiliki nilai fitur 3 jam
    df = df.dropna()

    # 6. Simpan hasil
    save_data(df)


if __name__ == "__main__":
    main()