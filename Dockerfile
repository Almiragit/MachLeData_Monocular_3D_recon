# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Install dependencies using the separate Docker requirements file
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ─── Runtime Stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Group 15 — Monocular 3D Reconstruction"
LABEL description="FastAPI inference server — Depth-Anything-V2"

WORKDIR /app

# Copy python packages and binaries from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# We use opencv-python-headless, so no heavy GL libs needed, just standard system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy source — DaV2 checkpoint dir is mounted at runtime via docker-compose volume
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/__init__.py ./app/__init__.py
COPY app/api/ ./app/api/
COPY app/monitoring/ ./app/monitoring/

RUN mkdir -p artifacts/checkpoints artifacts/logs outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
