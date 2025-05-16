import os
import torch
from torchvision import datasets, transforms, models

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Load the Full Quantized Model (Directly)
with torch.serialization.safe_globals([models.resnet.ResNet]):
    model = torch.load("models/resnet18_pruned_quantized_full.pth", weights_only=False)

# Ensure Model is in Evaluation Mode
model.eval()

# Evaluate Accuracy
correct, total = 0, 0
for inputs, labels in testloader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Quantized Model Accuracy: {100 * correct / total:.2f}%")
