import torch
import torch.nn as nn
from torchvision import models
import torch.quantization as quantization

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/resnet18_pruned.pth"))

model.train()
model.qconfig = quantization.get_default_qat_qconfig("fbgemm")
model_prepared = quantization.prepare_qat(model, inplace=False)

quantized_model = quantization.convert(model_prepared, inplace=False)
torch.save(quantized_model.state_dict(), "models/resnet18_pruned_quantized_state.pth")
