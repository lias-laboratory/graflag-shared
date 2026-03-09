FROM python:3.10

# Create working directory
WORKDIR /workspace

# Copy requirements first (for better layer caching)
COPY requirements.txt /workspace/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Copy the evaluator package
COPY . /workspace/graflag_evaluator/

# Set Python path
ENV PYTHONPATH=/workspace

# Set entrypoint
ENTRYPOINT ["python3", "/workspace/graflag_evaluator/run_evaluation.py"]
