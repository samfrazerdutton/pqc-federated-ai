# Use the official NVIDIA PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for CuPy and compilation if necessary)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir cupy-cuda12x

# Copy the entire project into the container
COPY . /app/

# Set the Python path so modules can find each other
ENV PYTHONPATH=/app

# Default command (can be overridden to run the client instead)
CMD ["python", "server/aggregator.py"]
