import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_baseline.pth"))

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
        prune.remove(module, "weight")

torch.save(model.state_dict(), "models/resnet18_pruned.pth")
