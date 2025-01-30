import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
import glob
import time
from collections import defaultdict

from ModulationDataset import ModulationDataset
from CNNs.LiteratureCNN import ModulationClassifier # CHANGE THIS TO THE MODEL YOU WANT TO USE

##############################################
########### MODIFIABLE PARAMETERS ############
##############################################
create_new_dataset = True
save_model = False
data_dir = "/exports/eddie/scratch/s2062378/data"
batch_size = 128
epochs = 30
learning_rate = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##############################################
########## END OF MODIFIABLE PARAMETERS ######
##############################################

print(f"\n\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Available devices: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device'}")
print("\n")


def create_dataset():
    # Remove any existing datasets
    for set in ["training", "validation", "testing"]:
        if os.path.exists(f"{data_dir}/{set}"):
            for f in glob.glob(f"{data_dir}/{set}/*"):
                os.remove(f)
            os.removedirs(f"{data_dir}/{set}")

    # Generate the new dataset using the generator.py script.
    os.system(f"python3 generator.py ./configs/training_set.json")
    os.system(f"python3 generator.py ./configs/validation_set.json")
    os.system(f"python3 generator.py ./configs/testing_set.json")


def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, _ in tqdm(
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
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")

    return model

def test_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels, snrs in tqdm(test_loader, desc="Testing Model"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Update overall accuracy
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update accuracy per SNR
            for i in range(len(snrs)):
                snr_value = snrs[i]  # Extract SNR
                snr_correct[snr_value] += (predicted[i] == labels[i]).item()
                snr_total[snr_value] += 1

    # Compute accuracy per SNR
    overall_accuracy = total_correct / total_samples * 100
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")

    for snr in sorted(snr_correct.keys()):
        snr_acc = (snr_correct[snr] / snr_total[snr]) * 100
        print(f"SNR {snr} dB -> Accuracy: {snr_acc:.2f}%")
    
    

def main():

    # Create new dataset
    if create_new_dataset:
        create_dataset()

    # Datasets
    train_dataset = ModulationDataset(os.path.join(data_dir, "training"))
    val_dataset = ModulationDataset(os.path.join(data_dir, "validation"))
    test_dataset = ModulationDataset(os.path.join(data_dir, "testing"))

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    num_classes = len(train_dataset.label_to_idx)
    model = ModulationClassifier(num_classes, input_len=1024) # Number of classes and length of each IQ sample.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train the model
    trained_model = train_model(
        train_loader, val_loader, model, criterion, optimizer, epochs, device
    )

    # Test the model
    test_model(trained_model, test_loader, device)

    # Save the model
    if save_model:
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
