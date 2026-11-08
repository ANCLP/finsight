# FinSight — AI Financial News Intelligence Platform
# Production Dockerfile optimized for Google Cloud Run (scale-to-zero, < 160MB)

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir --user -r requirements-server.txt

# Final minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy application code, pre-trained baseline model, and static webapp
COPY fin_sentiment/ /app/fin_sentiment/
COPY finsight/ /app/finsight/
COPY checkpoints/ /app/checkpoints/
COPY webapp/ /app/webapp/

# Default GCP Cloud Run port
EXPOSE 8080

# Execute serverless FastAPI app with dynamic port binding
CMD ["sh", "-c", "uvicorn fin_sentiment.web.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
