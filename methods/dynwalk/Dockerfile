FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "numpy<2" \
    scipy \
    networkx \
    pandas \
    scikit-learn \
    tqdm \
    psutil

WORKDIR /app

# Copy GraFlag integration files
COPY methods/dynwalk/train_graflag.py ./train_graflag.py

# Copy custom libraries
RUN pip install --no-cache-dir graflag-runner

CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
