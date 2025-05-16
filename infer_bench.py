import torch
import torch.nn as nn
import torch.quantization

# Load the quantized model (ensure it is for quantized CPU)
model = torch.load("models/resnet18_pruned_quantized_full.pth", map_location='cpu')
model.to('cpu')  # Ensure model is on CPU
model.eval()  # Set to evaluation mode

# Dummy input for testing (CPU tensor)
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

# Inference (CPU)
with torch.no_grad():
    outputs = model(dummy_input)

print(f"Inference completed. Output shape: {outputs.shape}")
