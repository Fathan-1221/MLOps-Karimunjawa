FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependensi dipasang lebih dulu supaya layer ini ter-cache
# selama requirements-api.txt tidak berubah
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Jalankan sebagai user biasa, bukan root
RUN useradd --create-home appuser

# Salin kode dan artefak model
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser models/model.pkl models/model.pkl
USER appuser

# Port harus sama dengan port uvicorn di CMD
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# host 0.0.0.0 wajib supaya bisa diakses dari luar container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
