# ==============================================================================
# GraFlag Method Dockerfile Template
# ==============================================================================
# This Dockerfile builds the container for your method.
# The container must include all dependencies and the graflag_runner library.
#
# Build context: The build runs from graflag-shared/ directory, so paths are:
#   - methods/your_method/  -> Your method files
#   - libs/                 -> GraFlag libraries (graflag_runner)
#   - datasets/             -> Not included in build (mounted at runtime)
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Choose a base image
# ------------------------------------------------------------------------------
# Options:
#   - python:3.9-slim          -> Lightweight Python (CPU only)
#   - pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime -> PyTorch with GPU
#   - nvidia/cuda:12.1.0-runtime-ubuntu22.04        -> CUDA base for custom setup
#   - tensorflow/tensorflow:2.12.0-gpu              -> TensorFlow with GPU

FROM python:3.9-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Step 2: Install system dependencies
# ------------------------------------------------------------------------------
# Add any system packages your method requires (compilers, libraries, etc.)

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Step 3: Upgrade pip
# ------------------------------------------------------------------------------

RUN pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------------------------
# Step 4: Install Python dependencies
# ------------------------------------------------------------------------------
# Option A: Install packages directly
# Option B: Use requirements.txt (COPY methods/your_method/requirements.txt .)

# Install PyTorch separately (--index-url must be in its own command)
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    scikit-learn \
    scipy \
    networkx \
    tqdm \
    psutil

# ------------------------------------------------------------------------------
# Step 5: Set working directory
# ------------------------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------------------------
# Step 6: (Optional) Clone external source code
# ------------------------------------------------------------------------------
# If your method wraps an existing implementation:
#
# RUN git clone https://github.com/author/method.git src
# RUN cd src && pip install -e .

# ------------------------------------------------------------------------------
# Step 7: Copy your method files
# ------------------------------------------------------------------------------
# Path is relative to graflag-shared/ (the build context)

COPY methods/example/train_graflag.py ./train_graflag.py

# Optional: Copy additional files if needed
# COPY methods/example/model.py ./model.py
# COPY methods/example/utils.py ./utils.py

# ------------------------------------------------------------------------------
# Step 8: Install GraFlag libraries (REQUIRED)
# ------------------------------------------------------------------------------
# The graflag_runner library provides ResultWriter and execution framework

COPY libs/ ./libs/
RUN pip install --no-cache-dir ./libs/graflag_runner

# ------------------------------------------------------------------------------
# Step 9: Set the entry point (REQUIRED)
# ------------------------------------------------------------------------------
# This runs graflag_runner which:
#   1. Sets up environment (DATA, EXP paths)
#   2. Converts _PARAM env vars to --param CLI args (--pass-env-args)
#   3. Executes your COMMAND from .env
#   4. Monitors resources and captures output

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]

# ==============================================================================
# NOTES:
# ==============================================================================
# - Build command: ./graflag_cli.py benchmark -m your_method -d dataset --build
# - The --build flag triggers image rebuild
# - Images are pushed to local registry at MANAGER_IP:5000
#
# - At runtime, these volumes are mounted:
#   - /shared/datasets/   -> Read-only dataset access
#   - /shared/experiments -> Write experiment results
#
# - GPU access is automatic if nvidia-docker is configured
# ==============================================================================
