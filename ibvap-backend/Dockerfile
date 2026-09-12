# ==============================================================================
# Production FastAPI & WebSocket Gateway Container
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATABASE_PATH=/app/data/ibvap.db

WORKDIR /app

# Install system dependencies for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create persistent storage directory for SQLite ledger and data
RUN mkdir -p /app/data && \
    useradd -m -u 1000 ibvapuser && \
    chown -R ibvapuser:ibvapuser /app

USER ibvapuser

EXPOSE 8000

VOLUME ["/app/data"]

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
