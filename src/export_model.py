"""Ekspor model terbaik dari MLflow ke file joblib (models/model.pkl).

File inilah yang disalin ke Docker image oleh Dockerfile.
Jalankan di laptop (tempat mlflow.db berada), bukan di dalam Docker.
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_NAME = "karimunjawa-kelayakan"
OUT_PATH = os.path.join("models", "model.pkl")


def resolve_uri(args) -> str:
    """Tentukan URI model: dari run ID, atau dari Model Registry."""
    if args.run_id:
        return f"runs:/{args.run_id}/model"

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{args.name}'")
    if not versions:
        raise SystemExit(
            f"Model '{args.name}' belum terdaftar di Model Registry. "
            "Daftarkan dulu lewat MLflow UI, atau pakai --run-id <ID_RUN>."
        )
    version = args.version or max(int(v.version) for v in versions)
    return f"models:/{args.name}/{version}"


def main():
    parser = argparse.ArgumentParser(description="Ekspor model MLflow ke models/model.pkl")
    parser.add_argument("--name", default=DEFAULT_NAME, help="nama model di Model Registry")
    parser.add_argument("--version", help="versi model (default: terbaru)")
    parser.add_argument("--run-id", help="ambil langsung dari run ID (tanpa registry)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    uri = resolve_uri(args)
    print(f"Memuat model dari {uri} ...")
    model = mlflow.sklearn.load_model(uri)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(model, OUT_PATH, compress=3)
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"Model disimpan ke {OUT_PATH} ({size_mb:.2f} MB)")
    print(f"Tipe model: {type(model).__name__}")


if __name__ == "__main__":
    main()
