# AD-GCL (AAAI 2025) -- see methods/ad_gcl/README.md.
#
# Built on the stack upstream's README states: Python 3.8.13, PyTorch 1.12.1,
# dgl 0.4.3.post1. AD-GCL is written against DGL 0.4's API -- it samples
# subgraphs with dgl.contrib.sampling.random_walk_with_restart, which DGL
# removed in 0.5, and it treats the graph as a host object, which is how DGL
# 0.4 worked. On DGL 2.4 it could not run as written, however many device
# errors were patched (README, "Verification").
#
# It runs on the CPU. DGL 0.4.3.post1's CUDA builds carry GPU machine code for
# sm_30 to sm_70 and no PTX, so on this cluster's NVIDIA A16 (sm_86) every DGL
# kernel launch failed without an error and left its output at zero
# (exp__ad_gcl__gad_cora__20261118_115058). This image therefore installs the
# CPU builds of torch, torch_scatter and DGL, and patches/cpu-device.patch lets
# `--gpu_id -1` select the CPU, which upstream's run.py cannot express.
#
# Python is Ubuntu 20.04's 3.8.10, not 3.8.13.

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# libgomp1: libdgl.so links libgomp.so.1, which the base image lacks.
RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
 && python3 -c "import sys; assert sys.version_info[:2] == (3, 8), sys.version; print('[OK] python', sys.version.split()[0])"

RUN python3 -m pip install --no-cache-dir "pip==24.0"

# PyTorch 1.12.1, CPU build, by URL so pip checks the pinned sha256.
RUN python3 -m pip install --no-cache-dir \
    "https://download.pytorch.org/whl/cpu/torch-1.12.1%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=42b7d500629c9e99d875cb204a1a1f5b3f65f6743823b21de9b4cba5962300d8" \
 && python3 -c "import torch; assert torch.__version__ == '1.12.1+cpu', torch.__version__; print('[OK] torch', torch.__version__)"

# torch_scatter, which AD-GCL's model imports, built for exactly this torch.
RUN python3 -m pip install --no-cache-dir \
    "torch-scatter==2.1.0+pt112cpu" -f https://data.pyg.org/whl/torch-1.12.1+cpu.html \
 && python3 -c "import torch_scatter; print('[OK] torch_scatter', torch_scatter.__version__)"

# numpy is held below 1.24, which removed the np.int/np.float aliases code of
# DGL 0.4's age uses; networkx below 3.0, because upstream's adj_to_dgl_graph()
# calls nx.from_scipy_sparse_matrix(), which 3.0 removed. Pinning restores the
# API the authors wrote against rather than rewriting their calls.
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.23.5" "scipy==1.10.1" "scikit-learn==1.3.2" "networkx==2.8.8" \
    "tqdm==4.66.4" "requests==2.32.3" "psutil==5.9.8"

# DGL 0.4 asks which framework to use, on stdin, unless told: with no
# DGLBACKEND and no ~/.dgl/config.json, `import dgl` calls input(), which in a
# container reads EOF and raises (exp__ad_gcl__gad_cora__20261118_114249).
ENV DGLBACKEND=pytorch

# DGL 0.4.3.post1, CPU build (PyPI's `dgl`, the package name upstream's README
# gives), by URL so pip checks the pinned sha256. The CPU build links nothing
# a docker build lacks, so the version and the sampler AD-GCL needs are
# checked here; train_graflag.py checks message passing again at run time.
RUN python3 -m pip install --no-cache-dir --no-deps \
    "https://files.pythonhosted.org/packages/9d/97/67873d394b47cb9db8b3583ae72a419fc0100680d205997cd47fd204b1e7/dgl-0.4.3.post1-cp38-cp38-manylinux1_x86_64.whl#sha256=fa68bbcc7254519e2e86a14365e48430162148a89a1f4722d02538fb7ad07e00" \
 && python3 -c "import dgl; assert dgl.__version__ == '0.4.3post1', dgl.__version__; from dgl.contrib.sampling import random_walk_with_restart; print('[OK] dgl', dgl.__version__, 'with dgl.contrib.sampling')"

WORKDIR /app

ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Two changes to the clone, both in run.py: the CPU device, and the export of
# the score vector run.py computes. `git apply --verbose` exits non-zero on
# drift, so upstream moving either line fails the build rather than producing
# an image that runs on the wrong device or writes no scores.
COPY methods/ad_gcl/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/cpu-device.patch \
 && git -C src apply --verbose /app/patches/export-scores.patch

# train_graflag.py runs `python3 AD-GCL/run.py` with cwd set into the clone
# rather than importing from it, so nothing statically references src. Assert
# the entry point and its loader are where we expect at build time: upstream
# moving run.py would otherwise surface as a subprocess failure at run time,
# after the image had shipped.
RUN test -f src/AD-GCL/run.py && test -f src/AD-GCL/utils.py \
    && test -d src/Data

COPY methods/ad_gcl/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) python3 -m pip install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  python3 -m pip install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
