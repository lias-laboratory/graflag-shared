# ADA-GAD (AAAI 2024) -- see methods/ada_gad/README.md.
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
COPY methods/ada_gad/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/pyg-compat.patch \
 && git -C src apply --verbose /app/patches/expose-scores.patch

# ADA-GAD vendors its own PyGOD 0.3.1 under src/pygod and imports
# `pygod.metrics`, which the released PyGOD renamed to `pygod.metric`.
# Installing the published wheel instead would import cleanly and then fail at
# the first metric call, so the vendored copy is the one that has to win.
#
# It cannot be pip-installed: its setup.py reads README.rst and
# requirements.txt from its own directory and neither file is in the commit,
# so `pip install ./src/pygod` dies reading them. The package itself is
# perfectly importable, so it goes on the path instead of through pip.
#
# The path is src/pygod (the directory *containing* the package), not
# src/pygod/pygod. train_graflag.py then appends src rather than inserting it,
# because src/pygod has no __init__.py: put src first and `import pygod`
# resolves to that as a namespace package and `pygod.utils` stops existing.
ENV PYTHONPATH=/app/src/pygod

# Pin scikit-learn below 1.6 for this image only.
#
# The vendored PyGOD's BaseDetector.predict() calls sklearn's
# check_is_fitted(), and sklearn 1.6 made that go through get_tags(), which
# requires every estimator to implement __sklearn_tags__. PyGOD 0.3.1 predates
# that contract, so predict() raises
#   AttributeError: 'ADANET' object has no attribute '__sklearn_tags__'
# *after* the model has trained -- three encoders in, at the evaluation call.
#
# ADA-GAD's own requirements.txt pins scikit_learn==1.2.2, so this restores the
# environment the authors declared rather than inventing one. It sits after the
# shared prologue deliberately: the prologue's layers stay byte-identical with
# the other methods and keep sharing, and only this image carries the older
# sklearn.
RUN pip install --no-cache-dir --break-system-packages "scikit-learn<1.6"

COPY methods/ada_gad/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir --break-system-packages /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir --break-system-packages graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
