import argparse
import os
import re

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "processed", "jepara_weather_clean.csv")
DVC_FILE = os.path.join("data", "raw.dvc")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "karimunjawa-kelayakan"

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "rain",
    "pressure_drop_3h", "wind_speed_roll_max_3h", "rain_roll_sum_3h", "hour",
]


def get_data_version() -> str:
    """Ambil hash DVC dataset mentah (data/raw.dvc) supaya run tercatat versi datanya."""
    try:
        with open(DVC_FILE, encoding="utf-8") as f:
            match = re.search(r"md5:\s*(\S+)", f.read())
        return match.group(1) if match else "unknown"
    except FileNotFoundError:
        return "unknown"


def pick_label_column(df: pd.DataFrame) -> str:
    """Cari kolom label yang bertipe angka (0/1), apa pun namanya."""
    for col in ("kelayakan", "kelayakan_label"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError("Kolom label numerik (kelayakan / kelayakan_label) tidak ditemukan.")


def load_dataset(horizon: int, test_size: float):
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    label_col = pick_label_column(df)

    # Target = kelayakan `horizon` jam ke depan (hindari kebocoran label)
    df["target"] = df[label_col].shift(-horizon)
    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    counts = df["target"].value_counts().to_dict()
    print(f"Distribusi target (1=Layak, 0=Tidak Layak): {counts}")
    if df["target"].nunique() < 2:
        raise SystemExit(
            "Target hanya punya satu kelas, model tidak bisa dilatih.\n"
            "Ambil data lebih banyak (python src/ingest_data.py --days 90) "
            "dan/atau turunkan threshold di preprocess.py, lalu jalankan preprocess ulang."
        )

    split = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split], df.iloc[split:]
    n_bad_train = int((train["target"] == 0).sum())
    n_bad_test = int((test["target"] == 0).sum())
    print(f"Tidak Layak di data latih: {n_bad_train}, di data uji: {n_bad_test}")
    if n_bad_test < 5:
        print("[Peringatan] Data uji hampir tidak punya kejadian Tidak Layak, "
              "recall/F1 tidak bisa dipercaya. Coba --test-size 0.3 atau tambah data.")

    X = df[FEATURES].astype("float64")  # float semua, hindari warning skema MLflow
    return X.iloc[:split], train["target"], X.iloc[split:], test["target"]


def build_model(args):
    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=42,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000),
    )


def main():
    p = argparse.ArgumentParser(description="Training + MLflow tracking")
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=3, help="prediksi N jam ke depan")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    X_train, y_train, X_test, y_test = load_dataset(args.horizon, args.test_size)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name):
        params = {"model": args.model, "horizon": args.horizon, "test_size": args.test_size}
        if args.model == "rf":
            params.update(n_estimators=args.n_estimators, max_depth=args.max_depth)
        else:
            params.update(C=args.C)
        mlflow.log_params(params)
        mlflow.log_params({
            "n_train": len(X_train), "n_test": len(X_test),
            "n_train_tidak_layak": int((y_train == 0).sum()),
            "n_test_tidak_layak": int((y_test == 0).sum()),
        })
        mlflow.set_tag("data_version_md5", get_data_version())

        model = build_model(args)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Kelas positif = Tidak Layak (0)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, pos_label=0, zero_division=0),
            "recall": recall_score(y_test, pred, pos_label=0, zero_division=0),
            "f1": f1_score(y_test, pred, pos_label=0, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, "model", input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

        print(f"Run selesai: {params}")
        print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()