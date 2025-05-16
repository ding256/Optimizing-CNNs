import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torchvision import datasets, transforms, models

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Load the baseline ResNet-18 model
model = models.resnet18()
num_classes = 10  # CIFAR-10 has 10 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet18_baseline.pth"))
model = model.to(device)

# Function to apply structured pruning (filter pruning)
def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    print("Pruning applied with {}% of filters removed.".format(amount * 100))
    return model

# Applying pruning (30% of filters removed)
pruned_model = apply_pruning(model, amount=0.3)

# Fine-tuning the pruned model for 1 epoch
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)

pruned_model.train()
for epoch in range(1):  # Fine-tuning for 1 epoch
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = pruned_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Fine-tuning Epoch [{epoch+1}/1], Loss: {running_loss/len(trainloader):.4f}")

# Evaluating pruned model on test set
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
    print(f"Test Accuracy of Pruned Model: {accuracy:.2f}%")
    return accuracy

# Testing the pruned model
accuracy = evaluate(pruned_model, testloader)

# Saving the pruned model
pruned_model_path = "models/resnet18_pruned.pth"
torch.save(pruned_model.state_dict(), pruned_model_path)
print(f"Pruned model saved as {pruned_model_path}")
