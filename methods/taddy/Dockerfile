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
    numpy==1.19.2 \
    scipy==1.5.2 \
    scikit-learn==0.23.2 \
    networkx==2.5 \
    six \
    protobuf==3.20.0 \
    transformers==3.5.1

# Install PyTorch with CUDA 11.1 support
RUN pip install --no-cache-dir \
    torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

# Clone TADDY source code from GitHub
RUN git clone https://github.com/yixingwang0517/Temporal-Anomaly-Detection-in-Dynamic-Graphs-via-Transformer src \
    && find src -name '*.py' -exec sed -i 's/\r$//' {} +

# Copy GraFlag integration script
COPY methods/taddy/train_graflag.py ./train_graflag.py

# Copy custom libraries (graflag_runner will be added later)
RUN pip install --no-cache-dir graflag-runner

# Run with graflag_runner wrapper
CMD ["python3", "-m", "graflag_runner", "--pass-env-args"]
