# DiffGAD (ICLR 2025) -- see methods/diffgad/README.md.
#
# The dependency prologue below is byte-identical across the unsupervised
# methods added in this batch. That is deliberate and load-bearing: docker
# derives a layer's identity from the parent image plus the instruction, so
# identical prologues over the same base share one set of layers on disk
# instead of each method paying for its own ~13 GB. Diverge a line and the
# cluster silently pays again. If a version has to move, move it in all of
# them.

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    numpy scipy scikit-learn networkx pandas pyyaml tqdm tensorboardX

RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --break-system-packages \
    torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

WORKDIR /app

# Clone the pinned upstream. SOURCE_CODE and SOURCE_REF come from this
# method's .env via --build-arg. `test -n` is not ceremony: with SOURCE_REF
# empty, `git checkout --detach` still exits 0 and leaves the clone on
# whatever the default branch holds that day -- the silent float the pin
# exists to remove.
ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Expose the score vector main() already computes. `git apply --verbose` exits
# non-zero when the context no longer matches, so upstream drift fails the
# build here instead of producing an image whose main() returns None.
COPY methods/diffgad/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/expose-scores.patch

# DiffGAD imports `pygod.utils.load_data` and `pygod.metric.metric`, the
# current API, so the released PyGOD is the right one -- pinned to the same
# commit images/bond_base uses, so the two never disagree about what a
# detector or a metric means.
RUN pip install --no-cache-dir --break-system-packages \
    git+https://github.com/pygod-team/pygod.git@c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8

COPY methods/diffgad/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir --break-system-packages /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir --break-system-packages graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
