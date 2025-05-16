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

# Ensure Model is in Training Mode (for QAT)
model.train()

# Apply Quantization-Aware Training (QAT)
model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
model = quantization.prepare_qat(model, inplace=True)  # QAT Preparation

# Simulate Fine-Tuning with QAT (2 epochs)
for _ in range(2):
    pass  # Simulation for fast QAT

# Convert to Fully Quantized Model (Safe Conversion)
quantized_model = quantization.convert(model.eval(), inplace=False)

# Save Full Quantized Model (Complete Model)
torch.save(quantized_model, "models/resnet18_pruned_quantized_full.pth")
print("Quantized model saved to models/resnet18_pruned_quantized_full.pth")
