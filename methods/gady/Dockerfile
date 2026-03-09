FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.19.5 \
    pandas==1.1.5 \
    scipy==1.5.4 \
    scikit-learn==0.24.2 \
    tqdm==4.62.3 \
    psutil

# Install PyTorch with CUDA 11.1 support
RUN pip install --no-cache-dir \
    torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

# Clone GADY source code from GitHub
ENV SOURCE_CODE=https://github.com/mufeng-74/GADY
RUN git clone ${SOURCE_CODE} src

# Create directories for GADY data processing
RUN mkdir -p data pos_features results log

# Copy GraFlag integration files
COPY methods/gady/train_graflag.py ./train_graflag.py
COPY methods/gady/data_loader.py ./data_loader.py

# Copy and install graflag_runner library
COPY libs/ ./libs/
RUN pip install --no-cache-dir ./libs/graflag_runner

# Entry point with parameter extraction
CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
