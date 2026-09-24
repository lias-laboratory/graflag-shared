## Stage 1: Build StreamSpot with old GCC
FROM ubuntu:16.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

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
    && git clone ${SOURCE_CODE} sbustreamspot-core \
    && git -C sbustreamspot-core checkout --detach ${SOURCE_REF}

# Local fixes to the cloned source live in patches/ and are applied with
# `git apply`, which checks the context it is editing and exits non-zero when
# upstream no longer matches. A `sed` in its place exits 0 either way. Each
# patch header says what it changes, why, and the SOURCE_REF it was cut from.
COPY methods/streamspot/patches/ /patches/
RUN git -C sbustreamspot-core apply --verbose /patches/*.patch

# `test -x` catches a `make` that printed errors and still exited 0.
RUN cd sbustreamspot-core \
    && make clean \
    && make optimized \
    && test -x streamspot

## Stage 2: Runtime with modern Python
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python packages. psutil is gone with the sampling loop it fed:
# graflag_runner measures exec time and peak memory from outside the method.
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas

WORKDIR /app

# Copy compiled binary from builder stage
COPY --from=builder /build/sbustreamspot-core/streamspot /app/sbustreamspot-core/streamspot

# Copy GraFlag integration files
COPY methods/streamspot/train_graflag.py ./train_graflag.py

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
