Prediksi kelayakan penyeberangan & wisata Karimunjawa berbasis data cuaca maritim (Open-Meteo API), dibangun dengan pipeline MLOps end-to-end.

## Tujuan Proyek

Memprediksi status **Layak** / **Tidak Layak** untuk aktivitas penyeberangan laut ke Karimunjawa berdasarkan parameter cuaca maritim (angin, curah hujan, tekanan udara, suhu, kelembapan) yang diambil secara real-time dan berkala dari Open-Meteo Forecast API. Proyek ini bertujuan membantu wisatawan dan operator kapal mengambil keputusan lebih awal (H-1/H-2) untuk menghindari kerugian akibat pembatalan mendadak.

## Struktur Direktori

MLOps-Karimunjawa/
├── .devcontainer/       # Konfigurasi GitHub Codespaces (devcontainer.json)
├── .github/workflows/   # CI/CD pipeline (GitHub Actions)
├── data/
│   ├── raw/              # Data mentah hasil fetch dari Open-Meteo API
│   └── processed/        # Data hasil preprocessing/feature engineering
├── models/               # Model hasil training (artifact, tidak di-commit ke git)
├── notebooks/            # Notebook eksplorasi & eksperimen (EDA, dsb.)
├── src/                  # Source code utama (data pipeline, training, API)
├── config/               # File konfigurasi (parameter, threshold, dsb.)
├── tests/                # Unit test
├── requirements.txt      # Daftar dependency Python
├── .gitignore
├── LICENSE
└── README.md
```

## Cara Menjalankan Codespaces

1. Buka repository ini di GitHub.
2. Klik tombol **Code** → tab **Codespaces** → **Create codespace on main**.
3. Tunggu environment selesai di-build sesuai konfigurasi pada `.devcontainer/devcontainer.json` (Python 3.11 + dependency otomatis ter-install lewat `requirements.txt`).
4. Setelah Codespace aktif, environment sudah siap dipakai.

## Branching Strategy

Proyek ini menerapkan **GitHub Flow**:
- Branch `main` selalu dalam kondisi stabil/deploy-ready.
- Setiap eksperimen/fitur baru dikerjakan di branch terpisah, misalnya `feat/initial-eda`.
- Perubahan di-merge ke `main` melalui Pull Request setelah divalidasi.

## Menjalankan Pipeline Data

1. **Fetch data cuaca** (bootstrap historis + update per jam):
```bash
   python src/data_acquisition.py
```
2. **Preprocessing & feature engineering** (cleaning, labeling kelayakan):
```bash
   python src/preprocessing.py
```
   Output tersimpan di `data/processed/jepara_weather_clean.csv`.