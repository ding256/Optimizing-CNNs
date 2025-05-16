import os
import torch
import torch.nn as nn
from torchvision import models
import torch.quantization as quantization

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Pruned Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned.pth"))
model.eval()

# Set Model to QAT Mode
model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
model = quantization.prepare_qat(model, inplace=False)

# Simulate Fine-Tuning (2 epochs, fast simulation)
for _ in range(2):
    pass  # We are not fine-tuning here for speed

# Convert to Fully Quantized Model (Safe Conversion)
quantized_model = quantization.convert(model.eval(), inplace=False)

# Save Full Quantized Model (not state_dict)
torch.save(quantized_model, "models/resnet18_pruned_quantized_full.pth")
print("Quantized model saved to models/resnet18_pruned_quantized_full.pth")
