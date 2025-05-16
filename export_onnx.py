import os
import torch
import torch.nn as nn
from torchvision import models

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the quantized model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned_quantized.pth"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "models/resnet18_int8.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)
print(f"ONNX model saved to {onnx_path}")
