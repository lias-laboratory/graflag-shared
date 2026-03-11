FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# Install Python dependencies (numpy<2 required for PyTorch 1.13.1 compatibility)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    scikit-learn \
    tqdm \
    pandas \
    psutil

WORKDIR /app

# Clone AddGraph repository
RUN git clone https://github.com/Ljiajie/Addgraph.git src

# Copy GraFlag integration files
COPY methods/addgraph/train_graflag.py ./train_graflag.py

# Copy custom libraries
RUN pip3 install --no-cache-dir graflag-runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
