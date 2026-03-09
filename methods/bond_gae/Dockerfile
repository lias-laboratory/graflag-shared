FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from PyPI
RUN pip install --no-cache-dir --break-system-packages \
    numpy scipy scikit-learn networkx psutil

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and extensions
RUN pip install --no-cache-dir --break-system-packages \
    torch-geometric pyg-lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install PyGOD
RUN pip install --no-cache-dir --break-system-packages --upgrade git+https://github.com/pygod-team/pygod.git

WORKDIR /app

# Install your custom libraries
COPY libs/ ./libs/
RUN pip install --no-cache-dir --break-system-packages ./libs/graflag_runner ./libs/graflag_bond

CMD ["python3", "-m", "graflag_runner"]