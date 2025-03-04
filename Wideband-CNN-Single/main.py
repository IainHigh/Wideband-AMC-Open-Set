import os
import glob
import time
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from CNN import ModulationClassifier
from ModulationDataset import WidebandModulationDataset

np.Inf = np.inf

# ============================================================
# USER CONFIGURATIONS
# ============================================================
create_new_dataset = False
save_model = True
test_only = False
save_model_path = "modulation_classifier.pth"

# REINFORCE training hyperparams
epochs = 15
learning_rate = 0.001
batch_size = 256

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


# ============================================================
# DATASET CREATION
# ============================================================
def create_dataset():
    """
    Removes existing training/validation/testing sets, then calls generator.py
    to create new sets using specified JSON config files and seeds.
    """
    for set_name in ["training", "validation", "testing"]:
        full_path = os.path.join(data_dir, set_name)
        if os.path.exists(full_path):
            for f in glob.glob(f"{full_path}/*"):
                os.remove(f)
            os.removedirs(full_path)

    # Example usage with seeds
    os.system(f"python3 generator.py ./configs/training_set.json {rng_seed + 1}")
    os.system(f"python3 generator.py ./configs/validation_set.json {rng_seed + 2}")
    os.system(f"python3 generator.py ./configs/testing_set.json {rng_seed + 3}")


# ============================================================
# CUSTOM COLLATE FOR THE WIDEBAND DATASET
# ============================================================
def wideband_collate_fn(batch):
    """
    batch is list of (wideband_tensor, metadata_dict, freq_start, freq_end).
    We'll combine them into Tensors for the training step.
    """
    wideband_list, meta_list, fstart_list, fend_list = [], [], [], []
    for xb, meta, fs, fe in batch:
        wideband_list.append(xb)
        meta_list.append(meta)
        fstart_list.append(fs)
        fend_list.append(fe)

    x_out = torch.stack(wideband_list, dim=0)
    f_s = torch.tensor(fstart_list, dtype=torch.float32)
    f_e = torch.tensor(fend_list, dtype=torch.float32)
    return x_out, meta_list, f_s, f_e


# ============================================================
# COST FUNCTION: multi_signal_cost
#  - includes frequency scaling to avoid huge costs
# ============================================================
def multi_signal_cost(predictions, gt_freqs, gt_mods, device):
    # TODO: REMOVE HARDCODED MAGIC NUMBERS (2, 5, 1e-6)

    """
    predictions: list[(freq_pred, mod_pred)]
    gt_freqs:    list of ground-truth freq floats
    gt_mods:     list of ground-truth mod indices
    returns a torch scalar cost
    """
    G = len(gt_freqs)
    P = len(predictions)
    # If no GT & no preds => cost=0
    if G == 0 and P == 0:
        return torch.tensor(0.0, device=device)
    # If no GT but some preds => false alarms
    if G == 0 and P > 0:
        return 2.0 * P * torch.tensor(1.0, device=device)
    # If GT but no preds => missed signals
    if G > 0 and P == 0:
        return 5.0 * G * torch.tensor(1.0, device=device)

    # Normalise freq for big values (MHz => scale)
    # We'll treat frequencies as "in Hz" => if your data is ~1e7 Hz, do:
    # scale them by 1e-6 to convert to MHz so that cost doesn't blow up
    # Then do a standard cost matrix approach
    freq_min = min(gt_freqs)
    freq_max = max(gt_freqs)
    if freq_max < freq_min:  # weird edge
        freq_max = freq_min + 1.0

    # convert to MHz
    scaled_gt_freqs = [(f * 1e-6) for f in gt_freqs]
    scaled_pred = []
    for freq_pred, mod_pred in predictions:
        scaled_pred.append((freq_pred * 1e-6, mod_pred))

    freq_range = (freq_max - freq_min) * 1e-6
    if freq_range < 1e-9:
        freq_range = 1.0

    cost_mat = torch.zeros((G, P), device=device)
    for g in range(G):
        freq_gt_mhz = scaled_gt_freqs[g]
        mod_gt = gt_mods[g]
        for p in range(P):
            freq_pred_mhz, mod_pred = scaled_pred[p]
            freq_err = abs(freq_pred_mhz - freq_gt_mhz) / freq_range
            mod_err = 0.0 if (mod_pred == mod_gt) else 1.0
            cost_mat[g, p] = freq_err + mod_err

    cost_np = cost_mat.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    cost_val = cost_mat[row_idx, col_idx].sum()

    unmatched_g = G - len(row_idx)
    unmatched_p = P - len(col_idx)
    # Missed signals => 5 each
    cost_val += 5.0 * unmatched_g
    # Extra signals => 2 each
    cost_val += 2.0 * unmatched_p

    return cost_val


# ============================================================
# REINFORCE TRAINING WITH RECURSIVE MODEL
# ============================================================
def train_reinforce(train_loader, val_loader, model, optimizer, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()

        for x_wide, meta_list, f_s, f_e in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            x_wide = x_wide.to(device)
            batch_size = x_wide.size(0)

            batch_loss = torch.tensor(0.0, device=device)
            for i in range(batch_size):
                meta_i = meta_list[i]
                gt_freqs = meta_i["center_frequencies"]
                gt_mods = meta_i["modulation_classes"]
                startf = f_s[i].item()
                endf = f_e[i].item()

                log_probs_accum = []
                # Recursively gather predictions
                predictions = model.stochastic_recursive_inference(
                    x_wide[i : i + 1], startf, endf, log_probs_accum
                )

                # Convert final predictions to cost => cost_i
                cost_i = multi_signal_cost(predictions, gt_freqs, gt_mods, device)

                # sum of log-probs from recursion
                sum_log_p = torch.stack(log_probs_accum).sum()

                # REINFORCE update => loss = - reward_i * sum_log_p
                sample_loss = cost_i * sum_log_p
                batch_loss += sample_loss

            batch_loss = batch_loss / max(1, batch_size)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Average loss: {batch_loss.item():.4f}")

        ########## VALIDATION STEP ##########
        # We'll measure a naive modulation accuracy approach
        val_mod_acc = validate_mod_accuracy(val_loader, model)
        print(f"Validation Accuracy: {val_mod_acc:.2f}%")

    return model


@torch.no_grad()
def validate_mod_accuracy(val_loader, model):
    """
    Evaluate mod accuracy with recursion on the val_loader.
    Returns a float [0..100] for naive mod accuracy.
    """
    model.eval()
    total_preds = 0
    total_correct = 0

    for x_wide, meta_list, f_s, f_e in val_loader:
        x_wide = x_wide.to(device)
        batch_size = x_wide.size(0)
        for i in range(batch_size):
            meta_i = meta_list[i]
            gt_mods = meta_i["modulation_classes"]
            preds = model.stochastic_recursive_inference(
                x_wide[i : i + 1], f_s[i].item(), f_e[i].item(), log_probs_accum=[]
            )
            for _, pred_mod in preds:
                total_preds += 1
                if pred_mod in gt_mods:
                    total_correct += 1

    if total_preds > 0:
        return 100.0 * (total_correct / total_preds)
    else:
        return 0.0


# ============================================================
# TESTING: PLOT CONFUSION MATRIX + SNR STATS
#  We do a naive approach: each predicted mod, if in GT => correct
# ============================================================
def test_model(model, test_loader):
    model.eval()
    total_preds = 0
    total_correct = 0

    # We'll store predicted and actual mod classes for confusion matrix
    # We don't know how many signals => we flatten them
    all_preds_mod = []
    all_true_mod = []

    # For SNR-based results, we track per sample
    # But each sample might produce multiple signals => we need a scheme
    # We'll do "for each predicted mod => see if it's in GT => if yes => correct"
    # Then can't define a single SNR for multiple signals. We'll just track the
    # sample's SNR array as a single "snr" if you wish. If you have multiple signals => not well-defined
    # We'll skip SNR breakdown or do a naive approach
    with torch.no_grad():
        for x_wide, meta_list, f_s, f_e in tqdm(test_loader, desc="Testing"):
            bsz = x_wide.size(0)
            x_wide = x_wide.to(device)
            for i in range(bsz):
                meta_i = meta_list[i]
                gt_mods = meta_i["modulation_classes"]
                # produce predictions
                preds = model.stochastic_recursive_inference(
                    x_wide[i : i + 1], f_s[i].item(), f_e[i].item(), []
                )
                # for each predicted mod => check if in GT
                for _, m_pred in preds:
                    total_preds += 1
                    all_preds_mod.append(m_pred)
                    # pick if it's correct if m_pred in gt_mods
                    if m_pred in gt_mods:
                        total_correct += 1
                        # We'll pick the first match in gt_mods as "the correct label"
                        # Or we can just do "good" for mod
                        # For confusion matrix we want a single label => let's pick m_pred as predicted,
                        # pick the first GT? or pick random? We'll do first index:
                        all_true_mod.append(gt_mods[0])  # naive
                    else:
                        # predicted mod was not in GT => for confusion matrix, pick the first GT anyway
                        # or we can pick random. We'll do first GT for demonstration
                        if len(gt_mods) > 0:
                            all_true_mod.append(gt_mods[0])
                        else:
                            # no GT => unify => pick -1?
                            all_true_mod.append(-1)

    # final stats
    if total_preds > 0:
        mod_acc = 100.0 * total_correct / total_preds
        print(f"\nTest mod accuracy => {mod_acc:.2f}%")
    else:
        print("No predictions made.")
        return

    # Let's also do a confusion matrix across the actual mod classes that appear
    # We assume all predicted mods are in [0, num_classes-1]
    # Some 'true' might be -1 if no GT. We'll filter those out
    valid_idx = [i for i, (t) in enumerate(all_true_mod) if t >= 0]
    pred_mods_filtered = [all_preds_mod[i] for i in valid_idx]
    true_mods_filtered = [all_true_mod[i] for i in valid_idx]
    if not valid_idx:
        print("No valid predictions for confusion matrix.")
        return

    # map them to a common label set
    unique_mods = sorted(list(set(true_mods_filtered + pred_mods_filtered)))
    # build index map
    label_map = {}
    for i, um in enumerate(unique_mods):
        label_map[um] = i
    # build arrays
    preds_idx = [label_map[m] for m in pred_mods_filtered]
    true_idx = [label_map[m] for m in true_mods_filtered]

    cm = confusion_matrix(true_idx, preds_idx)
    # percentage
    cm_perc = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_perc[i] = (cm[i] / row_sum) * 100.0

    class_labels = [f"mod{m}" for m in unique_mods]  # or do a better mapping

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_perc,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted mod")
    plt.ylabel("True mod")
    plt.title("Confusion Matrix (%)")
    plt.savefig("plots/recursive_confusion_matrix.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    # Possibly create dataset
    if create_new_dataset:
        create_dataset()

    # Load the wideband dataset
    train_data = WidebandModulationDataset(os.path.join(data_dir, "training"))
    val_dataset = WidebandModulationDataset(os.path.join(data_dir, "validation"))
    test_data = WidebandModulationDataset(os.path.join(data_dir, "testing"))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=wideband_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=wideband_collate_fn,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=wideband_collate_fn
    )

    model = ModulationClassifier(num_classes=len(train_data.label_to_idx))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train with REINFORCE
    model = train_reinforce(train_loader, val_loader, model, optimizer, epochs=epochs)

    # Evaluate final model
    test_model(model, test_loader)

    # Optionally save
    if save_model:
        torch.save(model.state_dict(), save_model_path)


if __name__ == "__main__":
    # If we have a saved model and we only want to test it
    if test_only:
        if not os.path.exists(save_model_path):
            print("Error: Model file not found.")
            sys.exit(1)
        if create_new_dataset:
            print(
                "Warning: create_new_dataset is set to True, but we are only testing a model. No new dataset will be created."
            )
        if save_model:
            print(
                "Warning: save_model is set to True, but we are only testing a model. A new model won't be created and hence will not be saved."
            )

        # Load the model and test it
        test_dataset = WidebandModulationDataset(
            os.path.join(data_dir, "testing"), transform=None
        )

        model = ModulationClassifier(len(test_dataset.label_to_idx))
        model.load_state_dict(torch.load(save_model_path))
        model.to(device)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=wideband_collate_fn,
        )

        test_model(model, test_loader, device)
        sys.exit(0)

    # Otherwise, train a model.
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
