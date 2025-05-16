import os
import torch
from torchvision import models

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Full Quantized Model Directly
with torch.serialization.safe_globals([models.resnet.ResNet]):
    quantized_model = torch.load("models/resnet18_pruned_quantized_full.pth", weights_only=False)

# Export the model directly without any mode changes
dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "models/resnet18_int8.onnx"
torch.onnx.export(quantized_model, dummy_input, onnx_path, opset_version=13, do_constant_folding=True)
print(f"ONNX model saved to {onnx_path}")
