# AD-GCL (AAAI 2025) -- see methods/ad_gcl/README.md.

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# networkx is pinned below 3.0 on purpose. AD-GCL's adj_to_dgl_graph() calls
# nx.from_scipy_sparse_matrix(), which networkx 3.0 removed in favour of
# from_scipy_sparse_array(). Pinning restores the API the authors wrote
# against; rewriting the call would be a change to their code for no gain.
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

# DGL 2.4.0, not 2.5.0.
#
# data.dgl.ai lists dgl-2.5.0+cu121-cp312 in its index and then serves it with
# HTTP 403 -- the wheel is advertised but not downloadable, which is upstream's
# fault and not something a version specifier fixes. 2.4.0 and 2.3.0 both
# return 200 for the same interpreter; PyPI's dgl 2.2.1 ships Windows wheels
# only. 2.4.0 is the newest that can actually be fetched.
#
# Both checks below are load-bearing. The import alone is not enough: DGL's
# C++ half is built per torch version, so a mismatched pair imports fine and
# fails at the first graph operation.
RUN pip install --no-cache-dir --break-system-packages \
    "dgl==2.4.0+cu121" -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
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
# ordinary graph ops both succeed without it, which is why the build check
# above now imports dgl.sparse specifically.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}

RUN python3 -c "import dgl.sparse; print('[OK] dgl.sparse C++ library loads')"

# torch_scatter, from the PyG index matching torch 2.4.0+cu121.
#
# AD-GCL imports it in its model code. It is a compiled extension built per
# torch version, so it comes from the versioned index rather than PyPI -- and
# the import is checked here because a mismatched build imports as a
# ModuleNotFoundError only at run time, inside the method.
RUN pip install --no-cache-dir --break-system-packages \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html \
 && python3 -c "import torch_scatter; print('[OK] torch_scatter', torch_scatter.__version__)"

WORKDIR /app

ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# Export the score vector run.py computes. `git apply --verbose` exits
# non-zero on drift, so upstream moving that block fails the build rather than
# producing an image whose runs write no scores.
COPY methods/ad_gcl/patches/ /app/patches/
RUN git -C src apply --verbose /app/patches/gpu-compat.patch \
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
      local) pip install --no-cache-dir --break-system-packages /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir --break-system-packages graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
