# Multi-stage Dockerfile for TEN-GNN Spoofing Detection
# Supports both CPU and GPU inference

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/data /app/alerts

# ============================================
# CPU Stage
# ============================================
FROM base as cpu

ENV DEVICE=cpu
EXPOSE 8000

CMD ["python", "deployment/api/server.py", "--device", "cpu"]

# ============================================
# GPU Stage  
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEVICE=cuda

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install PyTorch with CUDA support
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/data /app/alerts

EXPOSE 8000

CMD ["python", "deployment/api/server.py", "--device", "cuda"]
