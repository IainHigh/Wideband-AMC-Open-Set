import torch
import torch.optim as optim
import torch.nn as nn
import os
import glob
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import filtfilt, firwin
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader

##############################################
########### MODIFIABLE PARAMETERS ############
##############################################

from CNNs.LiteratureCNN import (
    ModulationClassifier,
)  # Change this to the model you want to use.

# Change this between ModulationDataset and WidebandModulationDataset
from ModulationDataset import WidebandModulationDataset as ModulationDataset

create_new_dataset = True
save_model = False
batch_size = 8
epochs = 30
learning_rate = 0.002

##############################################
########## END OF MODIFIABLE PARAMETERS ######
##############################################

# Read the configs/system_parameters.json file.
with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]

data_dir = system_parameters["Dataset_Directory"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(rng_seed)

if torch.cuda.is_available():
    print("\n\n CUDA is available.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("\n\nCUDA is not available. Using CPU.")
print("\n")


##############################################
#         DATASET CREATION SCRIPT
##############################################
def create_dataset():
    """
    Removes existing training/validation/testing sets, then invokes generator.py
    to create new sets using the specified JSON config files and seeds.
    """
    for set_name in ["training", "validation", "testing"]:
        full_path = os.path.join(data_dir, set_name)
        if os.path.exists(full_path):
            for f in glob.glob(f"{full_path}/*"):
                os.remove(f)
            os.removedirs(full_path)

    # Generate the new dataset using the generator.py script.
    # Different rng seeds so each set is unique.
    os.system(f"python3 generator.py ./configs/training_set.json {rng_seed + 1}")
    os.system(f"python3 generator.py ./configs/validation_set.json {rng_seed + 2}")
    os.system(f"python3 generator.py ./configs/testing_set.json {rng_seed + 3}")


def multi_signal_collate_fn(batch):
    """
    Flatten sub-samples from each wideband file into a single batch of shape [N, 2, L].
    batch is a list of length B, each element is (sub_samples, sub_labels, sub_snrs).
    """
    all_samples = []
    all_labels = []
    all_snrs = []

    for sub_samps, sub_labs, sub_snrs_ in batch:
        all_samples.extend(sub_samps)  # each is shape [2, L]
        all_labels.extend(sub_labs)
        all_snrs.extend(sub_snrs_)

    if len(all_samples) == 0:
        # If no signals found in the entire batch
        # Return dummy tensors so train loop can skip
        return (
            torch.empty(0, 2, 1),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )

    # Stack
    x_out = torch.stack(all_samples, dim=0)  # shape [N, 2, L]
    y_out = torch.tensor(all_labels, dtype=torch.long)
    # Convert SNR to float tensor
    snr_out = torch.tensor(all_snrs, dtype=torch.float32)

    return x_out, y_out, snr_out


##############################################
#         TRAIN / TEST FUNCTIONS
##############################################
def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device):
    """
    Training function that loops over each epoch, does forward/backward passes,
    and evaluates on the validation set after each epoch.
    """
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, snrs in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            # Skip empty batch
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, snrs in val_loader:
                if inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.2f}%")

    return model


def test_model(model, test_loader, device):
    """
    Computes overall accuracy and per-SNR accuracy. Also saves confusion matrices.
    """
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model.eval()
    total_correct = 0
    total_samples = 0

    # Dictionaries for SNR-based results
    snr_correct = {}
    snr_total = {}

    # For confusion matrix
    overall_true = []
    overall_pred = {}

    # We will just store predicted and label in lists
    overall_pred_list = []
    overall_true_list = []

    # Dictionaries for storing predictions/labels per SNR
    snr_trues = {}
    snr_preds = {}

    with torch.no_grad():
        for inputs, labels, snrs in tqdm(test_loader, desc="Testing Model"):
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # For confusion matrix
            overall_true_list.extend(labels.cpu().numpy())
            overall_pred_list.extend(predicted.cpu().numpy())

            # Per-SNR tracking
            snrs = snrs.cpu().numpy()
            for i, snr_val in enumerate(snrs):
                if snr_val not in snr_correct:
                    snr_correct[snr_val] = 0
                    snr_total[snr_val] = 0
                    snr_trues[snr_val] = []
                    snr_preds[snr_val] = []
                if predicted[i] == labels[i]:
                    snr_correct[snr_val] += 1
                snr_total[snr_val] += 1

                snr_trues[snr_val].append(labels[i].item())
                snr_preds[snr_val].append(predicted[i].item())

    # Overall accuracy
    overall_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")

    # Per-SNR accuracy
    for snr_val in sorted(snr_correct.keys()):
        snr_acc = 100.0 * snr_correct[snr_val] / snr_total[snr_val]
        print(f"SNR {snr_val:.1f} dB -> Accuracy: {snr_acc:.2f}%")

    # Confusion matrix (overall)
    cm_overall = confusion_matrix(overall_true_list, overall_pred_list)
    cm_overall_perc = np.zeros_like(cm_overall, dtype=float)
    for i in range(cm_overall.shape[0]):
        row_sum = cm_overall[i].sum()
        if row_sum > 0:
            cm_overall_perc[i] = (cm_overall[i] / row_sum) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_overall_perc,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=range(cm_overall.shape[0]),
        yticklabels=range(cm_overall.shape[0]),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Overall Confusion Matrix (%)")
    plt.savefig(os.path.join(plot_dir, "confusion_matrix_overall.png"))
    plt.close()

    # Per-SNR confusion matrices
    for snr_val in sorted(snr_trues.keys()):
        true_labels_snr = snr_trues[snr_val]
        pred_labels_snr = snr_preds[snr_val]
        cm_snr = confusion_matrix(true_labels_snr, pred_labels_snr)
        cm_snr_perc = np.zeros_like(cm_snr, dtype=float)
        for i in range(cm_snr.shape[0]):
            row_sum = cm_snr[i].sum()
            if row_sum > 0:
                cm_snr_perc[i] = (cm_snr[i] / row_sum) * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_snr_perc,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=range(cm_snr.shape[0]),
            yticklabels=range(cm_snr.shape[0]),
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix for SNR {snr_val} dB (%)")
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix_snr_{snr_val}.png"))
        plt.close()


##############################################
#                    MAIN
##############################################
def main():
    # (Optional) Create new dataset via generator.py
    if create_new_dataset:
        create_dataset()

    # Build the wideband dataset for training/val/test
    train_dataset = ModulationDataset(
        os.path.join(data_dir, "training"), transform=None
    )
    val_dataset = ModulationDataset(
        os.path.join(data_dir, "validation"), transform=None
    )
    test_dataset = ModulationDataset(os.path.join(data_dir, "testing"), transform=None)

    # DataLoaders with our custom collate_fn that flattens sub-samples
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multi_signal_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multi_signal_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multi_signal_collate_fn,
    )

    # Model, Loss, Optimizer
    num_classes = len(train_dataset.label_to_idx)
    model = ModulationClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train
    trained_model = train_model(
        train_loader, val_loader, model, criterion, optimizer, epochs, device
    )

    # Test
    test_model(trained_model, test_loader, device)

    # Optional: save the trained model
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
