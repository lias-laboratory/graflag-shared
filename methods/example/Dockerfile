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
    tqdm

# ------------------------------------------------------------------------------
# Step 5: Set working directory
# ------------------------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------------------------
# Step 6: (Optional) Clone external source code
# ------------------------------------------------------------------------------
# If your method wraps an existing implementation, pin it. A floating clone
# means today's image and next month's image are different experiments, and
# nothing in the results records which one ran. Put the SHA in .env as
# SOURCE_REF; graflag forwards it and SOURCE_CODE as --build-arg, so the
# Dockerfile names neither and the two cannot drift apart:
#
# ARG SOURCE_CODE
# ARG SOURCE_REF
# RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
#     && git clone ${SOURCE_CODE} src \
#     && git -C src checkout --detach ${SOURCE_REF}
#
# The `test -n` is what makes an unset SOURCE_REF a failed build: without it,
# `git checkout --detach` with no argument exits 0 and the pin is silently gone.
#
# Local fixes to the clone go in methods/<name>/patches/ and are applied with
# `git apply`, never `sed -i`. sed exits 0 when it matches nothing, so a patch
# that stops applying after a SOURCE_REF bump disappears without a word; git
# apply refuses the patch and stops the build:
#
# COPY methods/example/patches/ /patches/
# RUN git -C src apply --verbose /patches/*.patch
#
# Then reach it from train_graflag.py with `upstream()`, which anchors on the
# script's own directory instead of a hardcoded /app:
#
#     from graflag_runner import upstream
#     upstream("src")            # adds /app/src to sys.path, or raises

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

# Where the GraFlag libraries come from. `local` (the default) installs the
# copy on the share, `pypi` the published wheel; graflag passes the choice as
# --build-arg from GRAFLAG_LIBS in its config. Until this existed, testing a
# change to graflag_runner meant hand-editing this file on the manager -- which
# is what happened, three different ways across eight methods, and `graflag
# sync --lib` meanwhile reported success while changing nothing that ran.
#
# The unknown arm exits non-zero on purpose: an if/else would read a typo as
# "pypi" and build an image whose libraries are not the ones that were asked
# for, which is the silent success this file's `test -n` guards also exist to
# prevent.
ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

# ------------------------------------------------------------------------------
# Step 9: Set the entry point (REQUIRED)
# ------------------------------------------------------------------------------
# This runs graflag_runner which:
#   1. Sets up the environment (DATA, EXP, and every _PARAM from .env)
#   2. Executes your COMMAND from .env
#   3. Monitors CPU, memory and GPU, and captures output
#
# Add --pass-env-args only if the command you run parses CLI arguments --
# typically when you are driving an upstream argparse entry point you cannot
# change. It converts _BATCH_SIZE=128 into --batch_size 128 (lowercased). A
# script written against the SDK reads them with params() and does not need it.

CMD ["python3", "-m", "graflag_runner"]

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
