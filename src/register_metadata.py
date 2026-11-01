"""Isi deskripsi dan tag metadata untuk model terdaftar di MLflow Model Registry.

Deskripsi tiap versi dibuat otomatis dari parameter dan metrik run asalnya,
jadi isinya selalu sesuai dengan data eksperimen. Aman dijalankan berulang.
"""

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
NAME = "karimunjawa-kelayakan"

MODEL_DESCRIPTION = (
    "Klasifikasi biner kelayakan penyeberangan Karimunjawa (1 = Layak, 0 = Tidak Layak) "
    "untuk 3 jam ke depan, berdasarkan data cuaca maritim Open-Meteo wilayah Jepara "
    "(suhu, kelembapan, tekanan, angin, hujan, dan fitur rolling 3 jam). "
    "Kelas positif untuk metrik adalah Tidak Layak."
)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    client.update_registered_model(NAME, description=MODEL_DESCRIPTION)

    for mv in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        run = client.get_run(mv.run_id)
        p, m = run.data.params, run.data.metrics
        desc = (
            f"{p.get('model', '?')} | n_estimators={p.get('n_estimators', '-')}, "
            f"max_depth={p.get('max_depth', '-')}, horizon={p.get('horizon', '-')} jam. "
            f"Metrik data uji: accuracy={m.get('accuracy', 0):.4f}, "
            f"precision={m.get('precision', 0):.4f}, recall={m.get('recall', 0):.4f}, "
            f"f1={m.get('f1', 0):.4f}. Run asal: {run.info.run_name}."
        )
        client.update_model_version(NAME, mv.version, description=desc)
        tags = {
            "run_name": run.info.run_name,
            "data_version_md5": run.data.tags.get("data_version_md5", "unknown"),
            "f1": f"{m.get('f1', 0):.4f}",
            "recall": f"{m.get('recall', 0):.4f}",
        }
        for key, value in tags.items():
            client.set_model_version_tag(NAME, mv.version, key, value)
        print(f"Versi {mv.version} diperbarui: {desc}")


if __name__ == "__main__":
    main()
