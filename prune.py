import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

# Ensure directories exist
os.makedirs("models", exist_ok=True)

# Load the Baseline Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_baseline.pth"))
model.eval()

# Apply Pruning (30% Filters)
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
        prune.remove(module, "weight")

# Save the Pruned Model
torch.save(model.state_dict(), "models/resnet18_pruned.pth")
print("Pruned model saved to models/resnet18_pruned.pth")
