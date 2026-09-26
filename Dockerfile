# Reproducible runtime for the GBM drug-recommender pipeline (CPU only).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small and the resolver away from CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies before copying the source so code edits don't bust the layer cache.
COPY pyproject.toml README.md LICENSE ./
COPY gbm_drug ./gbm_drug
RUN pip install ".[dashboard]"

COPY . .

EXPOSE 8501

CMD ["python", "main.py"]
