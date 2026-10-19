import pandas as pd
import os
import glob

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/jepara_weather_clean.csv"


def load_data():
    
    files = glob.glob(os.path.join(RAW_DIR, "weather_*.csv"))

    if not files:
        raise FileNotFoundError(
            "Tidak ditemukan file data cuaca di data/raw/"
        )

    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def clean_data(df):
    """Membersihkan data cuaca."""

    # Hapus timestamp yang sama
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values("timestamp")

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rain"
    ]

    # Interpolasi missing value maksimal 3 jam berturut-turut
    df[numeric_columns] = df[numeric_columns].interpolate(
        method="linear",
        limit=3
    )
    df = df.dropna()

    return df


def feature_engineering(df):

    # Perubahan tekanan udara selama 3 jam
    df["pressure_drop_3h"] = (
        df["pressure"].shift(3) - df["pressure"]
    )

    # Kecepatan angin maksimum dalam 3 jam
    df["wind_speed_roll_max_3h"] = (
        df["wind_speed"].rolling(window=3).max()
    )

    # Total curah hujan dalam 3 jam
    df["rain_roll_sum_3h"] = (
        df["rain"].rolling(window=3).sum()
    )

    # Jam pengamatan
    df["hour"] = df["timestamp"].dt.hour

    return df


def create_label(df):
    """Membuat label kelayakan penyeberangan."""

    risk_condition = (
        (df["wind_speed"] >= 30) |
        (df["rain"] >= 20) |
        (df["pressure_drop_3h"] >= 5)
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