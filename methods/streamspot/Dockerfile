## Stage 1: Build StreamSpot with old GCC
FROM ubuntu:16.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone and build StreamSpot Core
RUN git clone https://github.com/sbustreamspot/sbustreamspot-core.git && \
    cd sbustreamspot-core && \
    sed -i 's/-march=native -mtune=native/-O3/g' Makefile && \
    make clean && \
    make optimized

## Stage 2: Runtime with modern Python
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    psutil

WORKDIR /app

# Copy compiled binary from builder stage
COPY --from=builder /build/sbustreamspot-core/streamspot /app/sbustreamspot-core/streamspot

# Copy GraFlag integration files
COPY methods/streamspot/train_graflag.py ./train_graflag.py

# Copy custom libraries
COPY libs/ ./libs/
RUN pip3 install --no-cache-dir ./libs/graflag_runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
