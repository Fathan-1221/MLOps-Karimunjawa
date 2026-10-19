# Model Terbaik - Prediksi Kelayakan Penyeberangan Karimunjawa

- Model terdaftar: karimunjawa-kelayakan (versi 1)
- Run name / Run ID: rf_200_depth10 / [isi Run ID dari MLflow]
- Algoritma: Random Forest Classifier
- Parameter terbaik: n_estimators=200, max_depth=10, class_weight=balanced
- Horizon prediksi: 3 jam
- Metrik data uji: accuracy=0.9038, precision=0.4040, recall=0.2632, f1=0.3187
- Versi data: data-v3 (md5 [isi MD5 dari data/raw.dvc])
- Threshold label: angin >= [isi sesuai preprocess.py] km/jam, hujan >= [isi sesuai preprocess.py] mm/jam, penurunan tekanan >= [isi sesuai preprocess.py] hPa/3 jam
- Alasan dipilih: F1-score tertinggi untuk kelas Tidak Layak, yaitu 0.3187