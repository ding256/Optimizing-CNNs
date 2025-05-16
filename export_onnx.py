import torch
import torch.nn as nn
from torchvision import models
import torch.quantization as quantization

# Set device (CPU for ONNX export)
device = torch.device("cpu")

# Load the quantized model
model = models.resnet18()
num_classes = 10  # CIFAR-10 has 10 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_pruned_quantized.pth", map_location=device))
model.to(device)
model.eval()
print("Quantized model loaded.")

# Verify that the model is in eval mode and quantized
print("Model is quantized:", model.qconfig is not None)

# Create a dummy input for the ONNX export (CIFAR-10 image size)
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# Exporting the quantized model to ONNX
onnx_model_path = "resnet18_int8.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,  # Store model weights
    opset_version=13,    # Latest compatible ONNX opset
    do_constant_folding=True,  # Optimize the model
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Quantized model exported to ONNX format as {onnx_model_path}.")

# Verifying the ONNX model
import onnx

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully.")
