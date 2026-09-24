# F-FADE (WSDM 2021) -- see methods/f_fade/README.md.
#
# CPU-sized on purpose. F-FADE trains a small embedding online rather than a
# GPU model, and upstream pins torch==1.4.0; this installs a current torch
# because 1.4.0 has no wheel for a modern Python, and the operations used are
# stable across that gap. README.md records the substitution.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# numpy is pinned below 1.24 on purpose. F-FADE's model.py calls np.int,
# which numpy 1.24 removed -- it had been a deprecated alias for the builtin
# since 1.20. Upstream pins numpy==1.19.1, so this restores the era the code
# was written against rather than rewriting the authors' arrays.
RUN pip3 install --no-cache-dir "numpy<1.24" scipy scikit-learn pandas

RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

ARG SOURCE_CODE
ARG SOURCE_REF
RUN test -n "${SOURCE_CODE}" && test -n "${SOURCE_REF}" \
    && git clone ${SOURCE_CODE} src \
    && git -C src checkout --detach ${SOURCE_REF}

# train_graflag.py runs `python3 main.py` with cwd set into the clone rather
# than importing from it, so nothing statically references src. Assert the
# entry point and its modules are where we expect at build time: upstream
# renaming or moving main.py would otherwise surface as a subprocess failing
# at run time, after the image had shipped.
RUN test -f src/main.py && test -f src/model.py && test -f src/utils.py \
    && test -f src/arguments.py

COPY methods/f_fade/train_graflag.py ./train_graflag.py

ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip3 install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip3 install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

CMD ["python3", "-m", "graflag_runner"]
