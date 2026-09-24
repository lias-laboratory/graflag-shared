# MIDAS (AAAI 2020) -- see methods/midas/README.md.
#
# No CUDA base and no torch: MIDAS is header-only C++ with a count-min sketch,
# it does not train anything, and it runs on CPU by design -- the paper's claim
# is constant time and memory per edge. The image is small for the same reason
# anograph's is, and the two share this prologue.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy pandas scikit-learn

WORKDIR /app

# Clone the pinned upstream. `test -n` is not ceremony: with SOURCE_REF empty,
# `git checkout --detach` still exits 0 and leaves the clone on whatever the
# default branch holds that day.
ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Parameterise the driver. `git apply --verbose` exits non-zero when the
# context no longer matches, so upstream drift fails the build here rather
# than producing a binary that silently scores the repository's own DARPA copy
# instead of the mounted dataset.
COPY methods/midas/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/graflag-driver.patch

# TBB and OpenMP are optional in upstream's CMakeLists (FIND_PACKAGE(... QUIET))
# and only parallelise the Experiment target's sweep, not the scoring loop, so
# neither is installed.
RUN cmake -DCMAKE_BUILD_TYPE=Release -S src -B src/build \
    && cmake --build src/build --target Demo -j"$(nproc)" \
    && test -x src/build/Demo

COPY methods/midas/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip3 install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip3 install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
