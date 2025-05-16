# Optimizing CNNs: A Study of Software and Hardware Strategies

This project demonstrates how to optimize Convolutional Neural Networks (CNNs) using a combination of software and hardware strategies, including pruning, quantization, and deployment with TensorRT.

## Project Structure
- **`train.py`** - Train a baseline ResNet-18 model on CIFAR-10 using One-Cycle Learning Rate.
- **`prune.py`** - Prune the baseline model (30% structured pruning) and fine-tune.
- **`quantize_qat.py`** - Apply Quantization-Aware Training (QAT) to the pruned model.
- **`export_onnx.py`** - Export the quantized model to ONNX for TensorRT.
- **`infer_bench.py`** - Benchmark the model performance in various modes (PyTorch, ONNX, TensorRT).
- **`Dockerfile`** - Containerized setup for easy deployment.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <your-github-repo-url>
   cd Optimizing-CNNs
2. Install required Python packages:
   pip install torch torchvision onnx onnxruntime numpy
3. Use Docker for a reproducible environment:
   docker build -t optimizing-cnns .
   docker run --gpus all -it optimizing-cnns

## Usage
1. Train:
   python train.py
2. Prune (30%):
   python prune.py
3. Apply Quantization:
   python quantize.py
4. Export model to ONNX:
   python export_onnx.py
5. Inference and Benchmarking:
   python infer_bench.py
Result is inference_benchmark_results.txt

## Docker setup
# Build the Docker image
docker build -t optimizing-cnns .

# Run the Docker container (GPU support)
docker run --gpus all -it optimizing-cnns
