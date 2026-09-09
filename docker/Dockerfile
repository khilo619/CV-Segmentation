# ==============================================================================
# Multi-Stage Production Dockerfile for CV Stream Segmentation Engine (v1.0)
# CPU-Optimized, Minimal Footprint (<350MB)
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libqpdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
# Install production dependencies into a virtualenv or system site-packages
RUN uv pip install --system --no-cache-dir .

# Stage 2: Minimal Production Runtime
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/placeholder-org/cv-stream-segmenter"
LABEL org.opencontainers.image.description="CV Stream Segmentation Engine"

WORKDIR /app

# Install runtime libraries only (libqpdf for pikepdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqpdf29 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/

# Install the app itself in editable mode without re-downloading dependencies
RUN pip install --no-deps -e /app

# Run as non-root user for enterprise container security
USER 10001:10001

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "cv_segment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
