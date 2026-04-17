# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install uv==0.5.11 \
 && uv pip install --system --no-cache . \
 && uv pip install --system --no-cache \
      "fastapi>=0.110.0" \
      "uvicorn[standard]>=0.29.0" \
      "pydantic>=2.6.0" \
      "python-multipart>=0.0.9" \
      "prometheus-fastapi-instrumentator>=7.0.0" \
      "huggingface-hub>=0.22.0"

FROM python:3.13-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src /app/src
COPY artifacts /app/artifacts
COPY data/widget /app/data/widget
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "cardio_risk_rf.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
