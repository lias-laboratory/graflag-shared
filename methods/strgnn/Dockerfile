FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    "networkx<3.0" \
    tqdm \
    scikit-learn \
    gensim==4.3.0 \
    node2vec \
    psutil

WORKDIR /app

# Clone the pinned upstream. SOURCE_CODE and SOURCE_REF are declared in this
# method's .env and forwarded by `docker build --build-arg` (BUILD_ARG_KEYS in
# docker_ops.py), so the repository and the commit are named in one place.
#
# `test -n` is not ceremony. With SOURCE_REF empty -- a build run by hand, a
# typo in the key -- `git checkout --detach` still exits 0 and leaves the clone
# on whatever the default branch holds that day, which is the silent float the
# pin exists to remove.
ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Build the C++ library for DGCNN
WORKDIR /app/src/pytorch_DGCNN/lib
RUN make clean && make -j4

WORKDIR /app

# Copy GraFlag integration files
COPY methods/strgnn/train_graflag.py ./train_graflag.py

# Copy custom libraries
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
      local) pip3 install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip3 install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

# train_graflag.py reads its parameters with graflag_runner.params(),
# so the command needs no --pass-env-args rewrite.
CMD ["python3", "-m", "graflag_runner"]
