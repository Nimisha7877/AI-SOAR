# syntax=docker/dockerfile:1
# ============================================================================
# AI SOAR - inference API image
# ----------------------------------------------------------------------------
# Build (from the project root):
#     docker build -t ai-soar:latest .
# Run:
#     docker run --rm -p 8000:8000 ai-soar:latest
# Or use docker-compose.yml (preferred):  docker compose up --build
#
# The trained models (~8 MB) and the committed demo artifacts are baked into
# the image, so a container can serve /predict immediately - no dataset, no
# retraining. The CICIDS2017 dataset is deliberately NOT in the image.
# ============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=2 \
    AI_SOAR_API_HOST=0.0.0.0 \
    AI_SOAR_API_PORT=8000

# libgomp1 = the OpenMP runtime LightGBM links against; python:slim lacks it
# and the failure surfaces only at predict time, not at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first -> this layer is cached until requirements.txt changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# application code + config + committed artifacts
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY knowledge_base ./knowledge_base
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY tests ./tests

# IMPORTANT: we do NOT `pip install .` the package.
# ai_soar/config.py derives PROJECT_ROOT from its own file location
# (Path(__file__).resolve().parents[2]). Installing into site-packages would
# move that file and make PROJECT_ROOT point at /usr/local/lib/python3.12/...,
# so config/settings.yaml, artifacts/ and knowledge_base/ would never be found.
# Keeping the source tree at /app/src (via PYTHONPATH) makes PROJECT_ROOT=/app.

RUN useradd --create-home --uid 10001 soar \
    && mkdir -p /app/artifacts/incidents /app/artifacts/reports \
    && chown -R soar:soar /app
USER soar

EXPOSE 8000

# no curl/wget in slim images -> probe /health with the stdlib instead
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "scripts/serve_api.py"]