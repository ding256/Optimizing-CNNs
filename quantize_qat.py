import os
import torch
import torch.nn as nn
from torchvision import models
import torch.quantization as quantization

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Pruned Model (Baseline for QAT)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned.pth"))

# Set Model to Training Mode for QAT (Only Once)
model.train()

# Apply Quantization-Aware Training (QAT)
model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
model_prepared = quantization.prepare_qat(model, inplace=False)

# Simulate Fine-Tuning with QAT (No Real Training)
for _ in range(2):  # Simulated Training (Avoid long training)
    pass  

# Convert to Fully Quantized Model (Keep Structure)
quantized_model = quantization.convert(model_prepared, inplace=False)

# Save the Fully Quantized Model (Entire Model)
torch.save(quantized_model, "models/resnet18_pruned_quantized_full.pth")
print("Quantized model (full) saved to models/resnet18_pruned_quantized_full.pth")
