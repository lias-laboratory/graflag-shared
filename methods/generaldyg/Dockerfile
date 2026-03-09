FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages from requirements
RUN pip install --no-cache-dir \
    h5py==3.7.0 \
    imbalanced-learn==0.12.3 \
    matplotlib==3.10.0 \
    networkx==2.8.7 \
    numpy==1.23.3 \
    pandas==1.4.4 \
    scikit-learn==1.6.0 \
    scipy==1.8.1 \
    tqdm==4.65.2

# Install PyTorch with CUDA support (2.1.2+cu121)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric (2.2.0)
RUN pip install --no-cache-dir torch-geometric==2.2.0

# Install PyTorch Geometric extensions (need torch to be installed first)
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2+pt21cu121 \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

WORKDIR /app

# Clone the source code from GitHub repository
ENV SOURCE_CODE=https://github.com/YXNTU/GeneralDyG
RUN git clone ${SOURCE_CODE} src \
    && find src -name '*.py' -exec sed -i 's/\r$//' {} +

# Copy GraFlag integration files
COPY methods/generaldyg/train_graflag.py ./train_graflag.py
COPY methods/generaldyg/dataset_all.py ./dataset_all.py

# Copy custom libraries
COPY libs/ ./libs/
RUN pip install --no-cache-dir ./libs/graflag_runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
