import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Set directories for data and models
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (easily adjustable)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5

# Data transforms (standard for CIFAR-10)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 Dataset and DataLoader
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes (CIFAR-10)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print("\nStarting Training...\n")
model.train()  # Make sure the model is in training mode
for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track loss for this batch
        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(trainloader)}], Loss: {running_loss / 50:.4f}")
            running_loss = 0.0

# Save model after successful training
model_save_path = "models/resnet18_baseline.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nTraining Completed. Model saved to {model_save_path}\n")
