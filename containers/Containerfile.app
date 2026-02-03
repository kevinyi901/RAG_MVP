# Containerfile for RAG Application
# Base: Python 3.12.11 with table extraction support

FROM python:3.12.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and pre-downloaded wheels for air-gapped install
COPY requirements.txt .
COPY wheels/ /wheels/

# Install Python dependencies from local wheels (air-gapped)
RUN pip install --no-cache-dir --no-index --find-links /wheels/ -r requirements.txt

# Copy application code
COPY app/ .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Default command (overridden by compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
