"""Transisi stage model di MLflow Model Registry + catatan riwayat.

Contoh:
  python src/promote.py 1 Staging --note "kandidat awal"
  python src/promote.py 1 Production --note "F1 tertinggi"

Selain mengubah stage (None -> Staging -> Production -> Archived), script ini
memasang alias dengan nama yang sama (staging / production) dan mencatat
setiap transisi ke models/registry_log.md sebagai dokumentasi.
"""

import argparse
import os
import warnings
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
NAME = "karimunjawa-kelayakan"
LOG_PATH = os.path.join("models", "registry_log.md")
STAGES = ["Staging", "Production", "Archived", "None"]


def main():
    parser = argparse.ArgumentParser(description="Transisi stage model")
    parser.add_argument("version", help="nomor versi model")
    parser.add_argument("stage", choices=STAGES, help="stage tujuan")
    parser.add_argument("--note", default="", help="alasan transisi (masuk ke log)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    before = client.get_model_version(NAME, args.version).current_stage

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # API stage ditandai deprecated di MLflow baru
        client.transition_model_version_stage(
            NAME, args.version, args.stage,
            archive_existing_versions=(args.stage == "Production"),
        )

    if args.stage in ("Staging", "Production"):
        client.set_registered_model_alias(NAME, args.stage.lower(), args.version)

    # Versi yang naik ke Production tidak lagi berstatus staging: lepas alias staging-nya
    if args.stage == "Production":
        try:
            if str(client.get_model_version_by_alias(NAME, "staging").version) == str(args.version):
                client.delete_registered_model_alias(NAME, "staging")
        except Exception:
            pass  # alias staging memang belum ada

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        if new_file:
            f.write("# Riwayat Transisi Model\n\n| Waktu | Model | Versi | Dari | Ke | Catatan |\n|---|---|---|---|---|---|\n")
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {NAME} | {args.version} | {before} | {args.stage} | {args.note} |\n")

    print(f"Versi {args.version}: {before} -> {args.stage}")
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        full = client.get_model_version(NAME, v.version)  # get_model_version memuat alias
        print(f"  v{v.version}: stage={full.current_stage}, alias={sorted(full.aliases)}")


if __name__ == "__main__":
    main()
