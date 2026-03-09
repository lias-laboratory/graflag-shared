FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    "networkx<3.0" \
    tqdm \
    scikit-learn \
    gensim==4.3.0 \
    node2vec \
    psutil

WORKDIR /app

# Clone StrGNN repository
RUN git clone https://github.com/KnowledgeDiscovery/StrGNN.git src

# Build the C++ library for DGCNN
WORKDIR /app/src/pytorch_DGCNN/lib
RUN make clean && make -j4

WORKDIR /app

# Copy GraFlag integration files
COPY methods/strgnn/train_graflag.py ./train_graflag.py

# Copy custom libraries
COPY libs/ ./libs/
RUN pip3 install --no-cache-dir ./libs/graflag_runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
