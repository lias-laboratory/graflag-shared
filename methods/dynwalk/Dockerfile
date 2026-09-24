FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "numpy<2" \
    scipy \
    networkx \
    pandas \
    scikit-learn \
    tqdm
# psutil is gone with the sampling loop it fed: graflag_runner measures
# exec time, peak memory and peak GPU from outside the method.

WORKDIR /app

# Copy GraFlag integration files
COPY methods/dynwalk/train_graflag.py ./train_graflag.py

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
      local) pip install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

# train_graflag.py reads its parameters with graflag_runner.params(),
# so the command needs no --pass-env-args rewrite.
CMD ["python3", "-m", "graflag_runner"]
