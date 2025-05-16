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

# Exporting to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "models/resnet18_pruned_quantized_full.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=True)

print("ONNX model exported successfully.")
