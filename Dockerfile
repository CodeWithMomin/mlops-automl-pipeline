# Use an official lightweight Python image
FROM python:3.12-slim as builder

# Set build-time env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for compiling certain ML packages if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Final stage
FROM python:3.12-slim

# Install runtime system dependencies (libgomp1 is required by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set runtime env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/mlops/.local/bin:$PATH
ENV PYTHONPATH=/app

# Create a non-root user for security
RUN useradd -m -u 1000 mlops
WORKDIR /app

# Copy installed libraries from the builder stage
COPY --from=builder --chown=mlops:mlops /root/.local /home/mlops/.local

# Copy application source code
COPY --chown=mlops:mlops src/ /app/src/

# Create directory for local data store and change ownership
RUN mkdir -p /app/data_store && chown -R mlops:mlops /app/data_store

USER mlops

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
