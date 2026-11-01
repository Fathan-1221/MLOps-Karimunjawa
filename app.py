"""Layanan inferensi REST API: prediksi kelayakan penyeberangan Karimunjawa.

Endpoint:
  GET  /health   -> cek layanan hidup
  POST /predict  -> prediksi Layak / Tidak Layak dari data cuaca
Dokumentasi interaktif otomatis ada di /docs.
"""

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.pkl"))

# Urutan fitur harus sama dengan FEATURES di src/train.py
FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "rain",
    "pressure_drop_3h", "wind_speed_roll_max_3h", "rain_roll_sum_3h", "hour",
]

# Dimuat sekali saat server start. Kalau file model hilang, server langsung gagal start.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Prediksi Kelayakan Penyeberangan Karimunjawa",
    description="Memprediksi kelayakan penyeberangan 3 jam ke depan dari data cuaca maritim.",
    version="1.0.0",
)


class WeatherInput(BaseModel):
    temperature: float = Field(..., description="Suhu udara 2 m (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Kelembapan relatif (%)")
    pressure: float = Field(..., gt=0, description="Tekanan permukaan (hPa)")
    wind_speed: float = Field(..., ge=0, description="Kecepatan angin 10 m (km/jam)")
    rain: float = Field(..., ge=0, description="Curah hujan (mm/jam)")
    pressure_drop_3h: float = Field(..., description="Penurunan tekanan 3 jam terakhir (hPa)")
    wind_speed_roll_max_3h: float = Field(..., ge=0, description="Angin maksimum 3 jam terakhir (km/jam)")
    rain_roll_sum_3h: float = Field(..., ge=0, description="Total hujan 3 jam terakhir (mm)")
    hour: int = Field(..., ge=0, le=23, description="Jam pengamatan (0-23)")

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "temperature": 29.5,
            "humidity": 70,
            "pressure": 1009.3,
            "wind_speed": 21.5,
            "rain": 0.0,
            "pressure_drop_3h": 1.2,
            "wind_speed_roll_max_3h": 22.0,
            "rain_roll_sum_3h": 0.0,
            "hour": 14,
        }]
    })


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="1 = Layak, 0 = Tidak Layak")
    label: str
    probability_tidak_layak: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(data: WeatherInput):
    X = pd.DataFrame([data.model_dump()], columns=FEATURES).astype("float64")
    pred = int(model.predict(X)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        idx = list(model.classes_).index(0)  # kolom kelas 0 = Tidak Layak
        proba = round(float(model.predict_proba(X)[0][idx]), 4)

    return PredictionOut(
        prediction=pred,
        label="Layak" if pred == 1 else "Tidak Layak",
        probability_tidak_layak=proba,
    )
