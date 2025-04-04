#############################################
# main.py
#############################################

import torch
import json
import time
import sys
import os
import glob
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_wideband_yolo import WidebandYoloDataset
from model_and_loss_wideband_yolo import WidebandYoloModel, WidebandYoloLoss
from config_wideband_yolo import (
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VAL_PRINT_SAMPLES,
    SAMPLING_FREQUENCY,
    S,
    PRINT_CONFIG_FILE,
    print_config_file,
)

MIN_CONFIDENCE = 0.25

with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]
data_dir = system_parameters["Dataset_Directory"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(rng_seed)
random.seed(rng_seed)

if torch.cuda.is_available():
    print("\n\nCUDA is available.")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("\n\nCUDA is not available. Using CPU.")
print("\n")

def main():        
    # Print the configuration file
    if PRINT_CONFIG_FILE:
        print_config_file()

    # 1) Build dataset and loaders
    train_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "training"), transform=None
    )
    val_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "validation"), transform=None
    )
    test_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "testing"), transform=None
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2) Create model & loss
    num_samples = train_dataset.get_num_samples()
    model = WidebandYoloModel(num_samples).to(device)
    criterion = WidebandYoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) Training loop
    for epoch in range(EPOCHS):
        # Training
        model, avg_train_loss, train_mean_freq_err, train_cls_accuracy = train_model(
            model, train_loader, device, optimizer, criterion, epoch)

        # Validation
        avg_val_loss, val_mean_freq_err, val_cls_accuracy, val_frames = validate_model(
            model, val_loader, device, criterion, epoch)

        # Print metrics for this epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train: Loss={avg_train_loss:.4f},"
              f"  MeanFreqErr={train_mean_freq_err:.4f},"
              f"  ClsAcc={train_cls_accuracy:.2f}%")
        print(f"  Valid: Loss={avg_val_loss:.4f},"
              f"  MeanFreqErr={val_mean_freq_err:.4f},"
              f"  ClsAcc={val_cls_accuracy:.2f}%")

        # Print a random subset of "frames"
        random.shuffle(val_frames)
        to_print = val_frames[:VAL_PRINT_SAMPLES]  # up to VAL_PRINT_SAMPLES frames
        print(f"\n  Some random frames from validation (only {VAL_PRINT_SAMPLES} shown):")
        print(f"  Prediction format: (freq_offset [0,1], class, conf)")
        print(f"  GroundTruth format: (freq_offset [0,1], class)")
        for idx, frame_dict in enumerate(to_print, 1):
            pred_list = frame_dict["pred_list"]
            gt_list   = frame_dict["gt_list"]

            print(f"    Frame {idx}:")
            print(f"      Predicted => {pred_list}")
            print(f"      GroundTruth=> {gt_list}")
        print("")

    print("Training complete.")
    
    # 4) Test the model
    test_model(model, test_loader, device)

def train_model(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    total_train_loss = 0.0

    # For metrics:
    tp_count = 0    # true positives (both prediction and ground truth > threshold)
    fp_count = 0    # false positives (predicted > threshold but no object in ground truth)
    fn_count = 0    # false negatives (object in ground truth but predicted <= threshold)
    sum_freq_err_tp = 0.0  # frequency error for true positives
    correct_cls_tp = 0     # classification correct count for true positives

    for time_data, freq_data, label_tensor, _ in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{EPOCHS}"):
        time_data = time_data.to(device)
        freq_data = freq_data.to(device)
        label_tensor = label_tensor.to(device)

        optimizer.zero_grad()
        pred = model(time_data, freq_data)
        loss = criterion(pred, label_tensor)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Reshape prediction.
        bsize = pred.shape[0]
        pred_reshape = pred.view(bsize, pred.shape[1], -1, (1 + 1 + NUM_CLASSES))
        x_pred     = pred_reshape[..., 0]  # predicted frequency (normalized)
        conf_pred  = pred_reshape[..., 1]  # predicted confidence
        class_pred = pred_reshape[..., 2:]

        x_tgt    = label_tensor[..., 0]    # ground truth frequency (normalized)
        conf_tgt = label_tensor[..., 1]    # ground truth confidence (0 or 1 typically)
        class_tgt= label_tensor[..., 2:]

        # Define masks for evaluation:
        # TP: both predicted and gt confidence above MIN_CONFIDENCE.
        # FP: predicted above threshold, but gt not (i.e. false alarm).
        # FN: ground truth above threshold but prediction is below threshold.
        tp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)
        fp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt <= MIN_CONFIDENCE)
        fn_mask = (conf_pred <= MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)

        # Count the number of objects for each.
        tp_batch = tp_mask.sum().item()
        fp_batch = fp_mask.sum().item()
        fn_batch = fn_mask.sum().item()

        tp_count += tp_batch
        fp_count += fp_batch
        fn_count += fn_batch

        # Frequency error only for true positives.
        freq_err = torch.abs(x_pred - x_tgt)
        sum_freq_err_tp += freq_err[tp_mask].sum().item()

        # For classification: only count the ones that are TP.
        pred_class_idx = torch.argmax(class_pred, dim=-1)
        true_class_idx = torch.argmax(class_tgt, dim=-1)
        correct_tp = ((pred_class_idx == true_class_idx) & tp_mask).sum().item()
        correct_cls_tp += correct_tp

    avg_train_loss = total_train_loss / len(train_loader)
    train_mean_freq_err = (sum_freq_err_tp / tp_count) if tp_count > 0 else 0.0
    train_cls_accuracy = 100.0 * (correct_cls_tp / tp_count) if tp_count > 0 else 0.0
    fp_percentage = 100.0 * (fp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    fn_percentage = 100.0 * (fn_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0.0

    print(f"Train TP count: {tp_count}, FP count: {fp_count} ({fp_percentage:.2f}%), FN count: {fn_count} ({fn_percentage:.2f}%)")

    return model, avg_train_loss, train_mean_freq_err, train_cls_accuracy

def validate_model(model, val_loader, device, criterion, epoch):
    model.eval()
    total_val_loss = 0.0

    tp_count = 0
    fp_count = 0
    fn_count = 0
    sum_freq_err_tp = 0.0
    correct_cls_tp = 0

    val_frames = []

    with torch.no_grad():
        for time_data, freq_data, label_tensor, _ in tqdm(val_loader, desc=f"Validation epoch {epoch+1}/{EPOCHS}"):
            time_data = time_data.to(device)
            freq_data = freq_data.to(device)
            label_tensor = label_tensor.to(device)

            pred = model(time_data, freq_data)
            loss = criterion(pred, label_tensor)
            total_val_loss += loss.item()

            bsize = pred.shape[0]
            pred_reshape = pred.view(bsize, pred.shape[1], -1, (1 + 1 + NUM_CLASSES))
            x_pred     = pred_reshape[..., 0]
            conf_pred  = pred_reshape[..., 1]
            class_pred = pred_reshape[..., 2:]

            x_tgt    = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt= label_tensor[..., 2:]

            tp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)
            fp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt <= MIN_CONFIDENCE)
            fn_mask = (conf_pred <= MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)

            tp_batch = tp_mask.sum().item()
            fp_batch = fp_mask.sum().item()
            fn_batch = fn_mask.sum().item()

            tp_count += tp_batch
            fp_count += fp_batch
            fn_count += fn_batch

            freq_err = torch.abs(x_pred - x_tgt)
            sum_freq_err_tp += freq_err[tp_mask].sum().item()

            pred_class_idx = torch.argmax(class_pred, dim=-1)
            true_class_idx = torch.argmax(class_tgt, dim=-1)
            correct_tp = ((pred_class_idx == true_class_idx) & tp_mask).sum().item()
            correct_cls_tp += correct_tp

            # For printing frames (only include boxes with predicted conf > MIN_CONFIDENCE)
            for i in range(bsize):
                pred_list = []
                gt_list   = []
                for s_idx in range(pred_reshape.shape[1]):
                    for b_idx in range(pred_reshape.shape[2]):
                        x_p = x_pred[i, s_idx, b_idx].item()
                        # convert normalized offset to raw frequency value.
                        x_p = (s_idx * SAMPLING_FREQUENCY / S) + x_p * (SAMPLING_FREQUENCY / S)
                        conf_val = conf_pred[i, s_idx, b_idx].item()
                        cls_p = pred_class_idx[i, s_idx, b_idx].item()

                        if conf_val > MIN_CONFIDENCE:
                            pred_list.append((x_p, cls_p, conf_val))
                        if conf_tgt[i, s_idx, b_idx] > 0:
                            x_g = x_tgt[i, s_idx, b_idx].item()
                            x_g = (s_idx * SAMPLING_FREQUENCY / S) + x_g * (SAMPLING_FREQUENCY / S)
                            cls_g = true_class_idx[i, s_idx, b_idx].item()
                            gt_list.append((x_g, cls_g))
                pred_list.sort(key=lambda tup: tup[2], reverse=True)
                frame_dict = {"pred_list": pred_list, "gt_list": gt_list}
                val_frames.append(frame_dict)

    avg_val_loss = total_val_loss / len(val_loader)
    val_mean_freq_err = (sum_freq_err_tp / tp_count) if tp_count > 0 else 0.0
    val_cls_accuracy  = 100.0 * (correct_cls_tp / tp_count) if tp_count > 0 else 0.0
    fp_percentage = 100.0 * (fp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    fn_percentage = 100.0 * (fn_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0.0

    print(f"Val TP count: {tp_count}, FP count: {fp_count} ({fp_percentage:.2f}%), FN count: {fn_count} ({fn_percentage:.2f}%)")

    return avg_val_loss, val_mean_freq_err, val_cls_accuracy, val_frames

def test_model(model, test_loader, device):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_freq_err_tp = 0.0
    total_correct_cls = 0

    overall_true_classes = []
    overall_pred_classes = []

    snr_obj_count = {}
    snr_correct_cls = {}
    snr_freq_err = {}

    with torch.no_grad():
        for time_data, freq_data, label_tensor, snr_tensor in tqdm(test_loader, desc=f"Testing on test set"):
            time_data = time_data.to(device)
            freq_data = freq_data.to(device)
            label_tensor = label_tensor.to(device)
            pred = model(time_data, freq_data)

            bsize = pred.shape[0]
            Sdim = pred.shape[1]
            pred_reshape = pred.view(bsize, Sdim, -1, (1+1+NUM_CLASSES))
            x_pred     = pred_reshape[..., 0]
            conf_pred  = pred_reshape[..., 1]
            class_pred = pred_reshape[..., 2:]

            x_tgt      = label_tensor[..., 0]
            conf_tgt   = label_tensor[..., 1]
            class_tgt  = label_tensor[..., 2:]

            tp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)
            fp_mask = (conf_pred > MIN_CONFIDENCE) & (conf_tgt <= MIN_CONFIDENCE)
            fn_mask = (conf_pred <= MIN_CONFIDENCE) & (conf_tgt > MIN_CONFIDENCE)

            tp_batch = tp_mask.sum().item()
            fp_batch = fp_mask.sum().item()
            fn_batch = fn_mask.sum().item()

            total_tp += tp_batch
            total_fp += fp_batch
            total_fn += fn_batch

            freq_err = torch.abs(x_pred - x_tgt)
            total_freq_err_tp += freq_err[tp_mask].sum().item()

            pred_class_idx = torch.argmax(class_pred, dim=-1)
            true_class_idx = torch.argmax(class_tgt, dim=-1)
            correct_tp = ((pred_class_idx == true_class_idx) & tp_mask).sum().item()
            total_correct_cls += correct_tp

            # Confusion matrix collection (only for boxes above threshold)
            pred_class_flat = pred_class_idx[tp_mask].cpu().numpy()
            true_class_flat = true_class_idx[tp_mask].cpu().numpy()
            overall_true_classes.extend(true_class_flat.tolist())
            overall_pred_classes.extend(pred_class_flat.tolist())

            # Per-SNR evaluation.
            snrs = snr_tensor.numpy()
            for i in range(bsize):
                sample_snr = snrs[i]
                sample_mask = tp_mask[i]  # using only TP for classification metrics
                sample_tp = sample_mask.sum().item()
                if sample_tp > 0:
                    sample_freq_err = freq_err[i][sample_mask].sum().item()
                    sample_correct = ((pred_class_idx[i] == true_class_idx[i]) & sample_mask).sum().item()
                    if sample_snr not in snr_obj_count:
                        snr_obj_count[sample_snr] = 0
                        snr_correct_cls[sample_snr] = 0
                        snr_freq_err[sample_snr] = 0.0
                    snr_obj_count[sample_snr] += sample_tp
                    snr_correct_cls[sample_snr] += sample_correct
                    snr_freq_err[sample_snr] += sample_freq_err

    overall_cls_acc = 100.0 * (total_correct_cls / total_tp) if total_tp > 0 else 0.0
    overall_freq_err = total_freq_err_tp / total_tp if total_tp > 0 else 0.0
    fp_percentage = 100.0 * (total_fp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    fn_percentage = 100.0 * (total_fn / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

    print("\n=== TEST SET RESULTS ===")
    print(f"Overall TP count: {total_tp}")
    print(f"Overall Classification Accuracy (TP only): {overall_cls_acc:.2f}%")
    print(f"Overall Mean Frequency Error (TP only): {overall_freq_err:.4f}")
    print(f"False Positives: {total_fp} ({fp_percentage:.2f}%)")
    print(f"False Negatives: {total_fn} ({fn_percentage:.2f}%)")

    snr_keys_sorted = sorted(snr_obj_count.keys())
    for snr_val in snr_keys_sorted:
        if snr_obj_count[snr_val] > 0:
            cls_acc_snr = 100.0 * snr_correct_cls[snr_val] / snr_obj_count[snr_val]
            freq_err_snr = snr_freq_err[snr_val] / snr_obj_count[snr_val]
        else:
            cls_acc_snr = 0.0
            freq_err_snr = 0.0
        print(f"SNR {snr_val:.1f}: Accuracy={cls_acc_snr:.2f}%, FreqErr={freq_err_snr:.4f}")

    # Confusion Matrix
    class_list = test_loader.dataset.class_list
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(overall_true_classes, overall_pred_classes, labels=range(len(class_list)))
    cm_percent = cm.astype(float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = (cm[i]/row_sum)*100.0

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_list, yticklabels=class_list)
    plt.title("Test Set Confusion Matrix (%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png")
    plt.close()
    print("\nTest confusion matrix saved to test_confusion_matrix.png.\n")
    print("=== END OF TESTING ===")
    

if __name__ == "__main__":
    start_time = time.time()
    main()
    time_diff = time.time() - start_time
    print(
        f"\nCode Execution took {time_diff // 3600:.0f} hours, "
        f"{(time_diff % 3600) // 60:.0f} minutes, {time_diff % 60:.0f} seconds."
    )