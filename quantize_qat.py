import os
import torch
import torch.nn as nn
from torchvision import models
import torch.quantization as quantization

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the pruned model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned.pth"))
model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
quantization.prepare_qat(model, inplace=True)
model.train()

# Simulate QAT Training (2 Epochs)
for _ in range(2):
    pass  # No actual training for simplicity

# Convert to Quantized Model
quantized_model = quantization.convert(model.eval(), inplace=False)
torch.save(quantized_model.state_dict(), "models/resnet18_pruned_quantized.pth")
print("Quantized model saved to models/resnet18_pruned_quantized.pth")
