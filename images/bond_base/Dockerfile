# The image all seventeen bond_* methods run in.
#
# It lives here rather than under methods/ because it belongs to no one of
# them. Seventeen byte-identical Dockerfiles used to produce seventeen copies
# of the same ~9 GB image, which is why building the whole method tree did not
# fit on the cluster's disk at all. A method opts in with IMAGE=bond_base in
# its .env; graflag resolves the name in one place (DockerManager._image_names)
# and the build and the service agree by construction.
#
# The methods stay distinct at run time: graflag sets METHOD_NAME per service
# and graflag_bond.train selects the PyGOD detector from it, so this image
# contains no method-specific step and must not grow one. Anything that would
# differ between the seventeen belongs in their .env files.
#
# The build context is still SHARED_DIR, the same as for a method Dockerfile.

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from PyPI
RUN pip install --no-cache-dir --break-system-packages \
    numpy scipy scikit-learn networkx
# psutil is gone with the sampling loop it fed in graflag_bond.train:
# graflag_runner measures exec time, peak memory and peak GPU from outside
# the method.

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and extensions
RUN pip install --no-cache-dir --break-system-packages \
    torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install PyGOD, pinned.
#
# This was `--upgrade git+...pygod.git` with no ref: every rebuild picked up
# whatever master held, and a detector whose signature had changed since the
# .env was written would start dropping parameters (get_all_parameters keeps
# only what the constructor accepts) or disappear from pygod.detector
# altogether. Both failures look like a GraFlag bug and are really an
# unannounced upstream release.
#
# Bump deliberately, and re-read methods/BOND.md's parameter tables when you
# do -- they are generated from the installed signatures.
RUN pip install --no-cache-dir --break-system-packages \
    git+https://github.com/pygod-team/pygod.git@c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8

WORKDIR /app

# Install your custom libraries
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
COPY libs/graflag_bond /tmp/graflag_libs/graflag_bond
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir --break-system-packages /tmp/graflag_libs/graflag_runner /tmp/graflag_libs/graflag_bond ;; \
      pypi)  pip install --no-cache-dir --break-system-packages graflag-runner graflag-bond ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]