import torch
import torch.nn as nn
import torch.quantization
import torch.onnx

# Load the quantized model (ensure the state dict is for quantized model)
model = torch.load("models/resnet18_pruned_quantized_full.pth", map_location='cpu')
model.to('cpu')  # Ensure model is on CPU
model.eval()  # Set to evaluation mode

# Dummy input for ONNX export (CPU tensor)
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

# Export to ONNX
onnx_path = "models/resnet18_pruned_quantized.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    opset_version=13, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output']
)

print(f"ONNX model exported to {onnx_path}")
