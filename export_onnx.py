import os
import torch
from torchvision import models

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Quantized Model (State Dict Only)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.eval()

# Load the Quantized State Dict
state_dict = torch.load("models/resnet18_pruned_quantized_state.pth")
model.load_state_dict(state_dict)

# Export the Quantized Model to ONNX
dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "models/resnet18_int8.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)
print(f"ONNX model saved to {onnx_path}")
