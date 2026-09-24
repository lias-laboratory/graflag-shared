FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages from requirements
RUN pip install --no-cache-dir \
    h5py==3.7.0 \
    imbalanced-learn==0.12.3 \
    matplotlib==3.10.0 \
    networkx==2.8.7 \
    numpy==1.23.3 \
    pandas==1.4.4 \
    scikit-learn==1.6.0 \
    scipy==1.8.1 \
    tqdm==4.65.2

# Install PyTorch with CUDA support (2.1.2+cu121)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric (2.2.0)
RUN pip install --no-cache-dir torch-geometric==2.2.0

# Install PyTorch Geometric extensions (need torch to be installed first)
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2+pt21cu121 \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

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

# 11 of the cloned .py files are CRLF at this pin. Python 3 opens those without
# complaint, so this is belt and braces rather than a fix for a known failure --
# but it is bracketed like any other rewrite, because `find -exec sed` exits 0
# when it changes nothing: the first grep asserts there is something to strip,
# the second that nothing survived. Drop the brackets and a SOURCE_REF bump
# could turn this into a no-op with no sign that it had.
RUN CR=$(printf '\r') \
    && grep -rlU "$CR" src --include='*.py' > /dev/null \
    && find src -name '*.py' -exec sed -i 's/\r$//' {} + \
    && ! grep -rlU "$CR" src --include='*.py' > /dev/null

# Copy GraFlag integration files
COPY methods/generaldyg/train_graflag.py ./train_graflag.py

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

# train_graflag.py applies its parameters onto upstream's argparse
# namespace with graflag_runner.apply_params(), so the command needs no
# --pass-env-args rewrite.
CMD ["python3", "-m", "graflag_runner"]
