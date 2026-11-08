"""Tahap 3 - Evaluasi dan validasi: bandingkan metrik run terbaru dengan ambang batas.

Ambang batas ada di config/thresholds.json. Kalau ada metrik di bawah ambang,
script keluar dengan kode 1 sehingga job CI gagal dan tahap registry tidak jalan.
"""

import json
import math
import os

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "karimunjawa-kelayakan"
THRESHOLDS_PATH = os.path.join("config", "thresholds.json")


def check_thresholds(metrics: dict, thresholds: dict):
    """Bandingkan metrik dengan ambang batas minimum.

    Return (lolos, rincian). Metrik yang hilang atau NaN dianggap tidak lolos.
    """
    rows = []
    for name, minimum in thresholds.items():
        value = metrics.get(name)
        ok = value is not None and not math.isnan(value) and value >= minimum
        rows.append({"metric": name, "value": value, "minimum": minimum, "pass": ok})
    return all(r["pass"] for r in rows), rows


def get_latest_run():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("Tidak ada run di MLflow. Tahap training belum menghasilkan apa pun.")
    return runs.iloc[0]


def write_summary(text: str):
    """Tulis ke halaman ringkasan GitHub Actions kalau berjalan di CI."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main():
    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        thresholds = json.load(f)

    run = get_latest_run()
    metrics = {k: run.get(f"metrics.{k}") for k in thresholds}
    ok, rows = check_thresholds(metrics, thresholds)

    lines = [f"## Evaluasi model (run `{run['tags.mlflow.runName']}`)", "",
             "| Metrik | Nilai | Minimum | Status |", "|---|---|---|---|"]
    print(f"Run dievaluasi: {run['tags.mlflow.runName']} ({run['run_id']})")
    for r in rows:
        status = "LOLOS" if r["pass"] else "GAGAL"
        value = "-" if r["value"] is None else f"{r['value']:.4f}"
        print(f"  {r['metric']:<10} nilai={value:<8} minimum={r['minimum']:<6} {status}")
        lines.append(f"| {r['metric']} | {value} | {r['minimum']} | {status} |")
    lines += ["", "**Hasil: model LOLOS validasi.**" if ok else "**Hasil: model GAGAL validasi, tidak didaftarkan.**"]
    write_summary("\n".join(lines))

    if not ok:
        raise SystemExit(1)
    print("Model lolos validasi.")


if __name__ == "__main__":
    main()
