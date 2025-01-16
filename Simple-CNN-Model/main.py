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
#################### TODO ####################
##############################################
# 1) Get it functional on the GPU cluster.
# 2) Run the full model on dataset with 0 SNR in the dataset.
# 3) Improve the model.
# 4) All randomness should be seeded (for data generation too) - for reproducibility.

##############################################
########### MODIFIABLE PARAMETERS ############
##############################################
create_new_dataset = False
data_dir = "../data"
batch_size = 64
epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataset():
    print(
        "You are about to create a new dataset. This will delete the previous dataset."
    )
    print("Do you want to continue? (y/n)")
    choice = input()
    if choice != "y":
        print("Exiting...")
        return

    # Wipe out the previous data.
    for f in glob.glob("../data/training/*") + glob.glob("../data/validation/*"):
        os.remove(f)

    # If the dirs exist remove them
    if os.path.exists("../data/training"):
        os.removedirs("../data/training")
    if os.path.exists("../data/validation"):
        os.removedirs("../data/validation")

    # Generate the new dataset using the generator.py script.
    os.system("cd ..; python3 generator.py ./configs/training_set.json")
    os.system("cd ..; python3 generator.py ./configs/validation_set.json")


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
    model = ModulationClassifier(num_classes)
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
    print("STARTING TRAINING. TIMESTAMP: ", start_time)
    main()
    end_time = time.time()
    print("TRAINING FINISHED. TIMESTAMP: ", end_time)
    time_diff = end_time - start_time
    hours = time_diff // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60
    print(
        f"Training took {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds."
    )
