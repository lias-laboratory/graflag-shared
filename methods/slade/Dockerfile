FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages matching SLADE requirements
RUN pip install --no-cache-dir \
    pandas==1.5.2 \
    numpy==1.23.5 \
    scikit-learn==1.1.3 \
    tqdm==4.65.0 \
    psutil

# Install PyTorch with CUDA support (1.13.1+cu117)
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# Install torch-scatter matching torch version
RUN pip install --no-cache-dir \
    torch-scatter==2.1.0+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

WORKDIR /app

# Clone the source code from GitHub repository
ENV SOURCE_CODE=https://github.com/jhsk777/SLADE
RUN git clone ${SOURCE_CODE} src

# Copy GraFlag integration files
COPY methods/slade/train_graflag.py ./train_graflag.py

# Copy custom libraries
RUN pip install --no-cache-dir graflag-runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
