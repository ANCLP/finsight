# fisight: Production-grade financial news sentiment analysis engine
# Lightweight container (<150MB) for CLI and batch inference on CPU

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy application source code and reference datasets
COPY fin_sentiment/ /app/fin_sentiment/
COPY pyproject.toml README.md all-data.csv* /app/
COPY data/ /app/data/
COPY FinancialPhraseBank/ /app/FinancialPhraseBank/

# Install library CLI in container
RUN pip install --no-deps -e /app

# Default to CLI entrypoint
ENTRYPOINT ["fisight"]
CMD ["--help"]
