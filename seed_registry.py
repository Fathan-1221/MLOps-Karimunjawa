"""Seed Model Registry: daftarkan model produksi ke MLflow server (sekali saja).

Dijalankan otomatis oleh Docker Compose setelah mlflow-server sehat. Model dibaca dari
models/model.pkl (hasil ekspor model Production LK-07/LK-08), didaftarkan sebagai
'karimunjawa-kelayakan' dan diberi alias 'production'. Idempotent: kalau model sudah ada,
tidak melakukan apa-apa, sehingga data di volume tidak digandakan setiap compose up.
"""

import os
import time

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
NAME = os.getenv("MODEL_NAME", "karimunjawa-kelayakan")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.pkl"))
EXPERIMENT = "karimunjawa-kelayakan"


def wait_for_server(client, tries=10):
    for attempt in range(1, tries + 1):
        try:
            client.search_registered_models(max_results=1)
            return
        except Exception as err:
            print(f"[Percobaan {attempt}/{tries}] Tracking server belum siap: {err}", flush=True)
            time.sleep(3)
    raise SystemExit("Tracking server tidak dapat dihubungi.")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    wait_for_server(client)

    try:
        mv = client.get_model_version_by_alias(NAME, "production")
        print(f"Model '{NAME}' versi {mv.version} sudah terdaftar (alias production). Seed dilewati.")
        return
    except Exception:
        pass  # belum terdaftar

    print(f"Mendaftarkan {MODEL_PATH} ke {TRACKING_URI} ...")
    model = joblib.load(MODEL_PATH)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="seed_from_model_pkl"):
        mlflow.set_tag("source", "models/model.pkl")
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=NAME,
            serialization_format="cloudpickle",
        )

    latest = max(int(v.version) for v in client.search_model_versions(f"name='{NAME}'"))
    client.update_model_version(
        NAME, str(latest),
        description="Model produksi (models/model.pkl) didaftarkan otomatis oleh seed-registry.",
    )
    client.set_registered_model_alias(NAME, "production", str(latest))
    print(f"Selesai: '{NAME}' versi {latest} terdaftar dengan alias production.")


if __name__ == "__main__":
    main()
