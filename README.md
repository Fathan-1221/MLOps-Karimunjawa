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
5. Bisa tuliskan perintah python src/ingest_data.py untuk mencoba mengambil data dari openmateo

## Menjalankan Pipeline Data
 1. Fetch Data Cuaca
 Data cuaca diambil dari Open-Meteo Forecast API menggunakan:

python src/ingest_data.py

Script mengambil data cuaca dalam interval per jam untuk wilayah Jepara yang digunakan sebagai representasi kondisi cuaca menuju Karimunjawa.

Parameter cuaca yang diambil meliputi:

- temperature
- humidity
- pressure
- wind_speed
- rain

Secara default, script mengambil data 7 hari ke belakang dan data forecast 1 hari dari Open-Meteo. Data forecast yang waktunya belum terjadi akan dibuang.

Data hasil ingestion disimpan di:
data/raw/

Setiap kali proses ingestion dijalankan, data disimpan sebagai file snapshot baru dengan timestamp, contohnya:

data/raw/
├── weather_20260928_200001.csv
├── weather_20260928_210001.csv
└── weather_20260928_220001.csv

Dengan mekanisme tersebut, data hasil ingestion sebelumnya tidak tertimpa.
Untuk mengambil data historis dengan jumlah hari tertentu, dapat menggunakan:

python src/ingest_data.py --days 30

Perintah tersebut mengambil data 30 hari ke belakang.


2. Preprocessing & Feature Engineering

Setelah data berhasil diambil, jalankan:

python src/preprocess.py

Script akan membaca seluruh file snapshot weather_*.csv yang terdapat di data/raw/, kemudian menggabungkan data tersebut sebelum dilakukan preprocessing.

Tahapan preprocessing meliputi:

- Menggabungkan seluruh snapshot data cuaca
- Menghapus data duplikat berdasarkan timestamp
- Mengurutkan data berdasarkan waktu
- Menangani missing value
- Melakukan interpolasi data numerik
- Membuat fitur berbasis kondisi cuaca 3 jam
- Membuat label kelayakan

Feature engineering yang digunakan:

- pressure_drop_3h
- wind_speed_roll_max_3h
- rain_roll_sum_3h
- hour

Data hasil preprocessing disimpan di:

data/processed/jepara_weather_clean.csv
Aturan Label Kelayakan

Label kelayakan dibuat berdasarkan kondisi cuaca.
Status Tidak Layak (0) diberikan apabila memenuhi salah satu kondisi berikut:

wind_speed >= 30 km/h
ATAU
rain >= 20 mm
ATAU
pressure_drop_3h >= 5 hPa

Jika tidak memenuhi kondisi tersebut, maka diberikan status:

Layak (1)

Kolom hasil labeling:

kelayakan_label → 1 = Layak, 0 = Tidak Layak
kelayakan → "Layak" atau "Tidak Layak"


## Simulasi Ingestion Berkala

Script ingestion juga dapat digunakan untuk melakukan simulasi pengambilan data secara berkala.

Contoh:
python src/ingest_data.py --runs 3 --every 60

Perintah tersebut akan melakukan pengambilan data sebanyak 3 kali dengan jeda 60 detik antar pengambilan.
Setiap pengambilan menghasilkan file snapshot baru sehingga data sebelumnya tetap tersimpan.