# HUGE-GAD (AAAI 2025) -- see methods/huge_gad/README.md.

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# networkx is pinned below 3.0 for the same reason as ad_gcl: this corner of
# the literature calls nx.from_scipy_sparse_matrix(), which networkx 3.0
# removed in favour of from_scipy_sparse_array().
RUN pip install --no-cache-dir --break-system-packages \
    numpy scipy scikit-learn pandas tqdm "networkx<3"

# torch is pinned to 2.4.0 to match DGL, and torchaudio is not installed.
#
# DGL loads libgraphbolt_pytorch_<torch version>.so, so the two have to agree
# exactly; a mismatch imports the dgl package cleanly and then raises at the
# first graph operation, inside the method, long after the image has shipped.
# torchaudio is unused by this method and only forces a conflicting torch pin.
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# DGL 2.4.0, not 2.5.0, and from conda rather than pip.
#
# data.dgl.ai lists dgl-2.5.0+cu121-cp312 in its index and then serves it with
# HTTP 403 -- the wheel is advertised but not downloadable. 2.4.0 was the
# newest that could be fetched; PyPI's dgl stops at 2.2.1.
#
# On 2026-11-17 data.dgl.ai stopped completing the TLS handshake from every
# network tried (SSL_ERROR_SYSCALL), and the S3 bucket behind it serves the
# index but answers 403 for every wheel, so pip has no source at all. The DGL
# team publishes the same version for the same torch and CUDA on its conda
# channel; install_dgl.py fetches it, checks its pinned SHA-256 and extracts
# it next to torch. It is not the wheel's binary: its sparse library links the
# driver directly (see below). Its
# dependencies, from the package's requires.txt, are installed here by name
# because nothing resolves them.
#
# Both checks below are load-bearing. The import alone is not enough: DGL's
# C++ half is built per torch version, so a mismatched pair imports fine and
# fails at the first graph operation.
COPY methods/huge_gad/install_dgl.py /tmp/install_dgl.py
RUN pip install --no-cache-dir --break-system-packages \
    packaging pandas psutil "pydantic>=2.0" pyyaml requests tqdm \
 && python3 /tmp/install_dgl.py \
 && python3 -c "import torch, dgl; print('[OK] torch', torch.__version__, 'dgl', dgl.__version__)" \
 && python3 -c "import torch, dgl; g = dgl.graph((torch.tensor([0,1]), torch.tensor([1,2]))); assert g.num_nodes() == 3; print('[OK] dgl graph ops')"

# DGL's sparse library needs CUDA 12, and this base image is CUDA 13.
#
# libdgl_sparse_pytorch_2.4.0.so links against libnvrtc.so.12, which
# nvidia/cuda:13.0.2 does not provide -- it ships libnvrtc.so.13. torch is
# unaffected because it bundles its own CUDA 12 runtime under
# dist-packages/nvidia/, so the library is present in the image and simply not
# on the loader path. Putting it there is smaller than moving the whole image
# to a CUDA 12 base.
#
# This surfaced only at run time and only in dgl.sparse: `import dgl` and
# ordinary graph ops both succeed without it.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}

# The conda build of that library also links libcuda.so.1, the driver,
# which no image ships: the NVIDIA runtime mounts it into a running container,
# and a docker build has no GPU. So `import dgl.sparse` cannot run here, and
# train_graflag.py imports it before anything else at run time instead. What
# can be checked here is the part this image is responsible for: the library
# is present and the loader finds libnvrtc.so.12 through the path above.
RUN f=/usr/local/lib/python3.12/dist-packages/dgl/dgl_sparse/libdgl_sparse_pytorch_2.4.0.so \
 && test -f "$f" \
 && ldd "$f" | grep -q 'libnvrtc.so.12 => /' \
 && echo "[OK] dgl.sparse library present; libnvrtc.so.12 resolves"

# torchmetrics, which HUGE-GAD imports in its modules. Pure Python, so PyPI
# is fine and no versioned index is needed.
RUN pip install --no-cache-dir --break-system-packages torchmetrics \
 && python3 -c "import torchmetrics; print('[OK] torchmetrics', torchmetrics.__version__)"

# torch_geometric, for the datasets rather than for the method. A bond_* graph
# is a pickled PyG Data, so stage_dataset() -- graflag_runner's
# load_attributed_graph, then write_mat -- cannot unpickle it without PyG
# installed: the first run that got past DGL died there, `No module named
# 'torch_geometric'`. 2.6.1 is the release built against torch 2.4, which this
# image is pinned to for DGL; PyG's core is pure Python and declares no torch
# requirement, so installing it moves nothing. pyg-lib and the scatter/sparse
# extensions are not needed to load a Data and are not installed.
RUN pip install --no-cache-dir --break-system-packages torch-geometric==2.6.1 \
 && python3 -c "import torch, torch_geometric; from torch_geometric.data import Data; \
assert torch.__version__.startswith('2.4.0'), torch.__version__; \
print('[OK] torch_geometric', torch_geometric.__version__, 'with torch', torch.__version__)"

WORKDIR /app

ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Export the score vector run.py computes. `git apply --verbose` exits
# non-zero on drift, so upstream moving that block fails the build rather than
# producing an image whose runs write no scores.
COPY methods/huge_gad/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/export-scores.patch

# train_graflag.py runs `python3 main.py` with cwd set into the clone
# rather than importing from it, so nothing statically references src. Assert
# the entry point and its loader are where we expect at build time: upstream
# moving main.py would otherwise surface as a subprocess failure at run time,
# after the image had shipped.
RUN test -f src/main.py && test -f src/modules/utils.py \
    && test -d src/datasets

COPY methods/huge_gad/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir --break-system-packages /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir --break-system-packages graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
