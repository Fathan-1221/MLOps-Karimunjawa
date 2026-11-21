"""Layanan inferensi REST API: prediksi kelayakan penyeberangan Karimunjawa.

Sumber model:
  - MODEL_URI diisi (mis. models:/karimunjawa-kelayakan@production): model ditarik dari
    MLflow Model Registry lewat MLFLOW_TRACKING_URI (mode Docker Compose).
  - MODEL_URI kosong: model dibaca dari file MODEL_PATH (mode satu kontainer, LK-07).

Endpoint:
  GET  /health      -> cek layanan hidup
  GET  /model-info  -> dari mana model dimuat (bukti komunikasi dengan tracking server)
  POST /predict     -> prediksi Layak / Tidak Layak dari data cuaca
Dokumentasi interaktif otomatis ada di /docs.
"""

import os
import re
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.pkl"))
MODEL_URI = os.getenv("MODEL_URI")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Urutan fitur harus sama dengan FEATURES di src/train.py
FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "rain",
    "pressure_drop_3h", "wind_speed_roll_max_3h", "rain_roll_sum_3h", "hour",
]


def load_model():
    """Muat model dari MLflow server (jika MODEL_URI diisi) atau dari file lokal."""
    if not MODEL_URI:
        return joblib.load(MODEL_PATH), {"source": "file", "path": MODEL_PATH}

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    last_error = None
    for attempt in range(1, 6):
        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
            break
        except Exception as err:  # server baru bangun / model belum terdaftar
            last_error = err
            print(f"[Percobaan {attempt}/5] Gagal memuat {MODEL_URI}: {err}", flush=True)
            time.sleep(3)
    else:
        raise RuntimeError(f"Tidak bisa memuat model dari {MODEL_URI}") from last_error

    info = {"source": "mlflow-server", "tracking_uri": TRACKING_URI, "model_uri": MODEL_URI}
    match = re.fullmatch(r"models:/([^@/]+)@(.+)", MODEL_URI)
    if match:
        name, alias = match.groups()
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        info.update(model_name=name, alias=alias, version=str(mv.version), run_id=mv.run_id)
    print(f"Model dimuat: {info}", flush=True)
    return model, info


# Dimuat sekali saat server start. Kalau gagal, kontainer langsung berhenti (tidak diam-diam).
model, MODEL_INFO = load_model()

app = FastAPI(
    title="Prediksi Kelayakan Penyeberangan Karimunjawa",
    description="Memprediksi kelayakan penyeberangan 3 jam ke depan dari data cuaca maritim.",
    version="1.1.0",
)


class WeatherInput(BaseModel):
    temperature: float = Field(..., description="Suhu udara 2 m (derajat C)")
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


@app.get("/model-info")
def model_info():
    return MODEL_INFO


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
