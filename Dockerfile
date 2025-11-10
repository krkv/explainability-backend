# Use slim Python image for smaller size
FROM python:3.12-slim as builder

# Set build arguments
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY instances/ ./instances/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Cloud Run uses PORT env var, but 8080 is default)
EXPOSE 8080

# Health check (optional - Cloud Run has its own health checks)
# Using Python's built-in http.client for lightweight check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8080); conn.request('GET', '/ready'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Run FastAPI with uvicorn
# Cloud Run sets PORT env var dynamically, default to 8080 if not set
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
