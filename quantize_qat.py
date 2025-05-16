import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.quantization as quantization

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Load the pruned model
model = models.resnet18()
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_pruned.pth"))
model = model.to(device)
model.eval()

# Preparing the model for QAT
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
quantization.prepare_qat(model, inplace=True)
print("Model prepared for Quantization-Aware Training (QAT).")

# Fine-tuning QAT model (2 epochs)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

model.train()
for epoch in range(2):  # Fine-tuning for QAT (2 epochs)
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"QAT Fine-tuning Epoch [{epoch+1}/2], Loss: {running_loss/len(trainloader):.4f}")

# Converting the model to a fully quantized version
model.eval()
quantized_model = quantization.convert(model, inplace=False)
print("Model converted to fully quantized (int8).")

# Evaluating the quantized model
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy of Quantized Model: {accuracy:.2f}%")
    return accuracy

# Testing the quantized model
accuracy = evaluate(quantized_model, testloader)

# Saving the quantized model
quantized_model_path = "resnet18_pruned_quantized.pth"
torch.save(quantized_model.state_dict(), quantized_model_path)
print(f"Quantized model saved as {quantized_model_path}")
