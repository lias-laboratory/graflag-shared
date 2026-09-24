# RARE was developed and evaluated natively, not in the Dockerfile it ships
# (Python 3.11, torch 2.4.0, PyG 2.6.1). This image reproduces the native
# environment instead: Python 3.12.4 with pip-installed torch 2.6.0+cu124 and
# PyG 2.7.0, every package RARE imports at the version installed there.
FROM python:3.12.4-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The CUDA runtime, cuDNN and cuBLAS come with the torch wheel as the
# nvidia-*-cu12 packages it pins, the same set the native install has. The
# driver comes from the node. A CUDA base image would set these two for the
# nvidia runtime; without them it hands this container no GPU.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install --no-cache-dir torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Natively, PyG runs with torch-scatter and torch-sparse (both CUDA builds) and
# without pyg-lib, torch-cluster or torch-spline-conv. PyG picks its
# implementation of some operations by which of these are installed, so the
# image carries exactly these two.
RUN pip install --no-cache-dir torch-geometric==2.7.0 \
    torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Everything else RARE imports, at the native versions; joblib and
# threadpoolctl because they decide how scikit-learn's IsolationForest runs.
# pygod is installed natively but nothing under rare/ imports it.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.17.0 \
    scikit-learn==1.7.2 \
    joblib==1.5.2 \
    threadpoolctl==3.6.0 \
    networkx==3.5 \
    matplotlib==3.8.4 \
    tqdm==4.67.1 \
    pyyaml==6.0.1 \
    dacite==1.9.2 \
    requests==2.32.3

WORKDIR /app

# Clone the pinned upstream. SOURCE_CODE and SOURCE_REF are declared in this
# method's .env and forwarded by `docker build --build-arg`. `test -n` turns an
# empty SOURCE_REF into a failed build: `git checkout --detach` with no
# argument exits 0 and leaves the clone on the default branch.
ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

COPY methods/rare/train_graflag.py ./train_graflag.py

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

# Upstream's Dockerfile also sets CUBLAS_WORKSPACE_CONFIG and
# PYTORCH_CUDA_ALLOC_CONF. The native environment sets neither, and the first
# can change which cuBLAS kernels run, so neither is set here. Unbuffered
# output only makes the log stream as the run goes.
ENV PYTHONUNBUFFERED=1

# train_graflag.py reads its parameters with graflag_runner.params(),
# so the command needs no --pass-env-args rewrite.
CMD ["python3", "-m", "graflag_runner"]
