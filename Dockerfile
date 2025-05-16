# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all project files to the Docker image
COPY . .

# Install necessary Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default working directory
ENV PYTHONPATH=/app

# Default command when the container starts
CMD ["python", "train.py"]
