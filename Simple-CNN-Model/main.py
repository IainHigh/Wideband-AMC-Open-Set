import torch
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from ModulationDataset import ModulationDataset
from ModulationClassifier import ModulationClassifier


def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")

    return model


# Parameters
data_dir = "../data"
batch_size = 64
epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_dataset = ModulationDataset(os.path.join(data_dir, "training"))
val_dataset = ModulationDataset(os.path.join(data_dir, "validation"))

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
num_classes = len(train_dataset.label_to_idx)
model = ModulationClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def main():
    # Train the model
    trained_model = train_model(
        train_loader, val_loader, model, criterion, optimizer, epochs, device
    )

    # Save the model
    torch.save(trained_model.state_dict(), "modulation_classifier.pth")


if __name__ == "__main__":
    main()
