import os
import torch

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Full Quantized Model
quantized_model = torch.load("models/resnet18_pruned_quantized_full.pth")
quantized_model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "models/resnet18_int8.onnx"
torch.onnx.export(quantized_model, dummy_input, onnx_path, opset_version=13)
print(f"ONNX model saved to {onnx_path}")
