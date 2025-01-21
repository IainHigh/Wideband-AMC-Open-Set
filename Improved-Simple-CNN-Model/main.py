import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
import glob
import time

from ModulationDataset import ModulationDataset
from ModulationClassifier import ModulationClassifier

##############################################
########### MODIFIABLE PARAMETERS ############
##############################################
create_new_dataset = True
data_dir = "/exports/eddie/scratch/s2062378/data"
batch_size = 64
epochs = 100
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Available devices: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device'}")
print("\n")


def create_dataset():
    # If the dirs exist remove them
    if os.path.exists(f"{data_dir}/training"):
        for f in glob.glob(f"{data_dir}/training/*"):
                os.remove(f)
        os.removedirs(f"{data_dir}/training")
    if os.path.exists(f"{data_dir}/validation"):
        for f in glob.glob(f"{data_dir}/validation/*"):
                os.remove(f)
        os.removedirs(f"{data_dir}/validation")

    # Generate the new dataset using the generator.py script.
    os.system("python3 generator.py ./configs/training_set.json")
    os.system("python3 generator.py ./configs/validation_set.json")


def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
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


def main():

    # Create new dataset
    if create_new_dataset:
        create_dataset()

    # Datasets
    train_dataset = ModulationDataset(os.path.join(data_dir, "training"))
    val_dataset = ModulationDataset(os.path.join(data_dir, "validation"))

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    num_classes = len(train_dataset.label_to_idx)
    model = ModulationClassifier(num_classes, input_len=1024) # Number of classes and length of each IQ sample.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(
        train_loader, val_loader, model, criterion, optimizer, epochs, device
    )

    # Save the model
    torch.save(trained_model.state_dict(), "modulation_classifier.pth")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    time_diff = end_time - start_time
    hours = time_diff // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60
    print(
        f"\n\nTraining took {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds."
    )
