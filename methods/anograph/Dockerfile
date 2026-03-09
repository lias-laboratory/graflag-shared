FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    psutil

WORKDIR /app

# Clone AnoGraph repository
RUN git clone https://github.com/Stream-AD/AnoGraph.git src

# Fix missing includes in source files
RUN sed -i '1i #include <algorithm>' /app/src/code/anograph.cpp && \
    sed -i '1i #include <algorithm>' /app/src/code/anographk.cpp && \
    sed -i '1i #include <algorithm>\n#include <limits>' /app/src/code/hcms.cpp

# Build the C++ code
WORKDIR /app/src/code
RUN make clean && make

WORKDIR /app

# Copy GraFlag integration files
COPY methods/anograph/train_graflag.py ./train_graflag.py

# Copy custom libraries
COPY libs/ ./libs/
RUN pip3 install --no-cache-dir ./libs/graflag_runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
