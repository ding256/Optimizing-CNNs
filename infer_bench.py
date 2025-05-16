import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import get_default_qconfig, prepare, convert

# Load the quantized model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10

# Prepare the model for quantization
model.qconfig = get_default_qconfig("fbgemm")
model = prepare(model)
model = convert(model)

# Load the quantized state dictionary with strict=False
quantized_state_dict = torch.load("models/resnet18_pruned_quantized_state.pth")
model.load_state_dict(quantized_state_dict, strict=False)  # strict=False allows unmatched keys

# Set to evaluation mode
model.eval()

# Sample inference
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    outputs = model(dummy_input)
print("Inference complete with quantized model.")
