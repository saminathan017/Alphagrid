# ── AlphaGrid v7 — Dashboard Server ──────────────────────────────────────────
# Lightweight image: no CUDA, no PyTorch (models are pre-trained .pkl files)
# Build: docker build -t alphagrid .
# Run:   docker run -p 8080:8080 alphagrid
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (excludes torch — pre-trained models are .pkl, not .pt)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy pandas scipy yfinance alpaca-py \
    fastapi "uvicorn[standard]" pydantic python-multipart \
    sqlalchemy "passlib[bcrypt]" "python-jose[cryptography]" \
    lightgbm scikit-learn imbalanced-learn \
    aiohttp httpx requests websockets \
    loguru pyyaml python-dotenv python-dateutil pytz

# Copy project
COPY . .

# Port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start server
CMD ["python", "-m", "dashboard.app"]
