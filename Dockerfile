FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependensi dipasang lebih dulu supaya layer ini ter-cache
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Jalankan sebagai user biasa, bukan root
RUN useradd --create-home appuser

# Kode, skrip seed registry, dan model cadangan
COPY --chown=appuser:appuser app.py seed_registry.py ./
COPY --chown=appuser:appuser models/model.pkl models/model.pkl
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
