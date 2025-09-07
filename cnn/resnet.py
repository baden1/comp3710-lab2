import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int)
args = parser.parse_args()

# print(args.epochs)

def load_from_disk(path):
    loaded_model = resnet18(pretrained=False)
    loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 10)
    loaded_model.load_state_dict(torch.load(path, map_location=device))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Transformations: resize to 224x224, convert to tensor, normalize
transform = transforms.Compose(
    [
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

data_path = str(Path.home() / "scikit_learn_data")

# Load dataset
trainset = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

# Load ResNet-18
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)

# model = load_from_disk(Path.home() / "comp3710/lab2/cnn/resnet18.pth")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_start = time.time()

# Training loop
for epoch in range(args.epochs):
    start = time.time()
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

    end = time.time()
    print(f"epoch {epoch + 1} took {end - start:.3f}s")

train_end = time.time()
print(f"Training took {train_end - train_start:.3f}s")
print("Finished Training")

# Save model to disk
save_path = str(Path.home() / "comp3710/lab2/cnn/resnet18.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
