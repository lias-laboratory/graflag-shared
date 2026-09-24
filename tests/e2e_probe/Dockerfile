# The probe's image. Deliberately tiny: no CUDA, no upstream clone, no
# dependency beyond numpy, so a failed run points at the pipeline rather than
# at anything the method did.

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy

# Same stanza as every method image, for the same reason: what the container
# runs has to be the libraries that were asked for, and a typo must stop the
# build rather than quietly select the other source. A probe that tested a
# different graflag_runner from the one the methods use would answer the wrong
# question.
ARG GRAFLAG_LIBS=local
COPY libs/graflag_runner /tmp/graflag_libs/graflag_runner
RUN case "${GRAFLAG_LIBS}" in \
      local) pip install --no-cache-dir /tmp/graflag_libs/graflag_runner ;; \
      pypi)  pip install --no-cache-dir graflag-runner ;; \
      *) echo "[ERROR] GRAFLAG_LIBS must be local or pypi, got '${GRAFLAG_LIBS}'" >&2; exit 1 ;; \
    esac && rm -rf /tmp/graflag_libs

COPY methods/e2e_probe/train_graflag.py /app/train_graflag.py

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
