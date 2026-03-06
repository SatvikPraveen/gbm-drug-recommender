# Multi-stage Docker image for GBM Drug Analysis and Recommendation
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/smiles results/models results/figures results/similarity results/pathways

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for dashboard (if needed)
EXPOSE 8501

# Default command runs the main analysis
CMD ["python", "main.py"]
