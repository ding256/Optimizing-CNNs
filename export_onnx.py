import torch
from torchvision import models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned_quantized_state.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "models/resnet18_pruned_quantized.onnx"

model = model.dequantize()  # Convert to standard model for ONNX export
torch.onnx.export(model, dummy_input, onnx_path, opset_version=13)
