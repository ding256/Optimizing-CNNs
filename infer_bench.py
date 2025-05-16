import torch
import torch.nn as nn
import time
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms, models

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 Dataset (Test Set)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Load the Quantized Model (PyTorch)
model = models.resnet18()
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_pruned_quantized.pth"))
model.to(device)
model.eval()
print("Quantized PyTorch model loaded.")

# Evaluation Function
def evaluate_pytorch(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Benchmark Function (PyTorch)
def benchmark_pytorch(model, device, mode="FP32"):
    model.to(device)
    model.eval()
    inputs = torch.randn(1, 3, 32, 32).to(device)
    if mode == "INT8":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    # Warm-up
    for _ in range(10):
        _ = model(inputs)
    
    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = model(inputs)
    end = time.time()
    avg_latency = (end - start) / 1000 * 1000  # in milliseconds
    return avg_latency

# ONNX Inference (CPU)
onnx_model_path = "resnet18_int8.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("ONNX model loaded for CPU inference.")

def benchmark_onnx(session):
    inputs = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        session.run([output_name], {input_name: inputs})

    # Timing
    start = time.time()
    for _ in range(1000):
        session.run([output_name], {input_name: inputs})
    end = time.time()
    avg_latency = (end - start) / 1000 * 1000  # in milliseconds
    return avg_latency

# Evaluating and Benchmarking
print("\nStarting Inference Benchmarks...")
results = []

# PyTorch CPU FP32
cpu_accuracy = evaluate_pytorch(model, testloader, torch.device("cpu"))
cpu_latency = benchmark_pytorch(model, torch.device("cpu"), mode="FP32")
results.append(["PyTorch CPU (FP32)", cpu_accuracy, cpu_latency])

# PyTorch GPU FP32
if device.type == "cuda":
    gpu_accuracy = evaluate_pytorch(model, testloader, device)
    gpu_latency = benchmark_pytorch(model, device, mode="FP32")
    results.append(["PyTorch GPU (FP32)", gpu_accuracy, gpu_latency])

# PyTorch GPU INT8
if device.type == "cuda":
    int8_latency = benchmark_pytorch(model, device, mode="INT8")
    results.append(["PyTorch GPU (INT8)", gpu_accuracy, int8_latency])

# ONNX Runtime CPU INT8
onnx_latency = benchmark_onnx(session)
results.append(["ONNX CPU (INT8)", cpu_accuracy, onnx_latency])

# Displaying Results
print("\nBenchmark Results:")
print(f"{'Mode':<25} {'Accuracy (%)':<15} {'Latency (ms)'}")
for mode, acc, lat in results:
    print(f"{mode:<25} {acc:<15.2f} {lat:.3f} ms")

# Save results to file
with open("inference_benchmark_results.txt", "w") as f:
    f.write("Mode,Accuracy (%),Latency (ms)\n")
    for mode, acc, lat in results:
        f.write(f"{mode},{acc:.2f},{lat:.3f}\n")
    print("\nResults saved to inference_benchmark_results.txt")
