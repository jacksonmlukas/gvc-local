# ---------------------------------------------------------------------------
# GVC-Local API  --  multi-stage Docker build
#
# This image contains ONLY the FastAPI serving layer.  vLLM runs as a
# separate service (see docker-compose.yml) and this container talks to
# it over the internal Docker network.
#
# Build:
#   docker build -t gvc-local-api .
#
# Run:
#   docker run -p 8080:8080 \
#       -e VLLM_BASE_URL=http://vllm:8000/v1 \
#       -e VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct \
#       gvc-local-api
# ---------------------------------------------------------------------------

# === Stage 1: build dependencies ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (required for some transitive C extensions).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy only the files needed to resolve dependencies first so Docker
# can cache this layer independently of source changes.
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install .

# === Stage 2: runtime =====================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Jackson Lukas <jlukas3313@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/jacksonlukas/gvc-local"
LABEL org.opencontainers.image.description="GVC-Local API serving layer"

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source (needed for the app factory import path).
COPY src/ src/

# FastAPI + uvicorn must be available at runtime.
RUN pip install --no-cache-dir "fastapi>=0.111" "uvicorn[standard]>=0.29"

# Non-root user for production hardening.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --no-create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_BASE_URL=http://vllm:8000/v1 \
    VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

EXPOSE 8080

CMD ["uvicorn", "gvc_local.serving.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
