"""Verifikasi kesiapan inferensi: muat model dari Model Registry, lalu prediksi data baru.

Model dimuat lewat stage Production dan lewat alias production; keduanya
harus menunjuk ke versi yang sama dan menghasilkan prediksi yang sama.
"""

import time

import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlflow.db"
NAME = "karimunjawa-kelayakan"
DATA_PATH = "data/processed/jepara_weather_clean.csv"
FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "rain",
    "pressure_drop_3h", "wind_speed_roll_max_3h", "rain_roll_sum_3h", "hour",
]


def load(uri):
    start = time.perf_counter()
    model = mlflow.pyfunc.load_model(uri)
    print(f"Model dimuat dari {uri} dalam {time.perf_counter() - start:.2f} detik")
    return model


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    by_stage = load(f"models:/{NAME}/Production")
    by_alias = load(f"models:/{NAME}@production")

    # Data "baru": 5 baris terakhir dari data hasil preprocessing
    new_data = pd.read_csv(DATA_PATH).tail(5)
    X = new_data[FEATURES].astype("float64")

    start = time.perf_counter()
    pred = by_stage.predict(X)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\nPrediksi {len(X)} baris dalam {elapsed:.1f} ms")

    hasil = new_data[["timestamp"]].copy()
    hasil["prediksi"] = [int(p) for p in pred]
    hasil["label"] = hasil["prediksi"].map({1: "Layak", 0: "Tidak Layak"})
    print(hasil.to_string(index=False))

    same = list(pred) == list(by_alias.predict(X))
    print(f"\nPrediksi via stage == via alias: {same}")


if __name__ == "__main__":
    main()
