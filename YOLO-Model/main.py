#############################################
# main.py
#############################################

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import shutil
import numpy as np
import sys
import os
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from adjustText import adjust_text
from tqdm import tqdm
from dataset_wideband_yolo import WidebandYoloDataset
from model_and_loss_wideband_yolo import WidebandYoloModel, WidebandYoloLoss
import config_wideband_yolo as cfg

SAVE_MODEL_NAME = "yolo_model"
UNKNOWN_CLASS_IDX = -1

with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]
data_dir = system_parameters["Dataset_Directory"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(rng_seed)
random.seed(rng_seed)


def calibrate_open_set_threshold(model, train_loader, device):
    model.eval()
    all_maxprobs = []
    with torch.no_grad():
        for time_data, freq_data, label_tensor, _ in train_loader:
            time_data, freq_data = time_data.to(device), freq_data.to(device)
            pred = model(time_data, freq_data)
            # reshape to [B, S, B_, 1+1+NUM_CLASSES]
            Bsz = pred.shape[0]
            pred = pred.view(Bsz, cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES)
            # pull out only classification logits (no objectness)
            logits = pred[..., 2:]  # shape [B, S, B_, C]
            probs = F.softmax(logits, dim=-1)  # softmax over known classes
            maxp, _ = probs.max(dim=-1)  # [B, S, B_]
            # mask to only the *true* signals (anchors where target_confidence>0)
            obj_mask = label_tensor[..., 1] > 0
            all_maxprobs.append(maxp[obj_mask].cpu().numpy())
    all_maxprobs = np.concatenate(all_maxprobs)
    # find the (1-coverage) percentile
    tau = np.percentile(all_maxprobs, (1.0 - cfg.OPENSET_COVERAGE) * 100)
    return float(tau)


def convert_to_readable(frequency, modclass, class_list):
    # Convert frequency to MHz and modclass to string

    if frequency > 1000:
        size_map = {
            1: "Hz",
            1000: "KHz",
            1000000: "MHz",
            1000000000: "GHz",
            1000000000000: "THz",
        }
        for size in size_map.keys():
            if frequency < size:
                frequency /= size / 1000
                break
        frequency = round(frequency, 4)
        frequency_string = f"{frequency} {size_map[size / 1000]}"
    else:
        frequency_string = f"{frequency} Hz"

    modclass_str = class_list[modclass]
    return frequency_string, modclass_str


def main():
    # Print the configuration file
    if cfg.PRINT_CONFIG_FILE:
        cfg.print_config_file()

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

    cfg.MODULATION_CLASSES = train_dataset.class_list
    # reserve one more slot for “unknown”
    cfg.MODULATION_CLASSES = cfg.MODULATION_CLASSES + ["unknown"]
    NUM_TOTAL_CLASSES = len(cfg.MODULATION_CLASSES)
    UNKNOWN_CLASS_IDX = NUM_TOTAL_CLASSES - 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 2) Create model & loss
    num_samples = train_dataset.get_num_samples()
    model = WidebandYoloModel(num_samples).to(device)
    criterion = WidebandYoloLoss()

    start_epoch = 0
    # If the model is already partially trained, load the model and get the epoch from which to continue training.
    if cfg.MULTIPLE_JOBS_PER_TRAINING:
        for i in range(cfg.EPOCHS):
            if os.path.exists(f"{SAVE_MODEL_NAME}_epoch_{i+1}.pth"):
                # Load the model from the previous job
                model = WidebandYoloModel(train_dataset.get_num_samples()).to(device)
                checkpoint = torch.load(
                    f"{SAVE_MODEL_NAME}_epoch_{i+1}.pth", map_location="cpu"
                )
                model.load_state_dict(checkpoint)
                model.to(device)
                start_epoch = i + 1
                print(f"Loaded model from epoch {i+1}")

    if start_epoch == cfg.EPOCHS:
        print("Model training complete. No more epochs to train.")
        return

    # 3) Training loop
    for epoch in range(start_epoch, cfg.EPOCHS):

        # Set the learning rate depending on the epoch. Starts at LEARNING_RATE and decreases by a factor of 10 by the last epoch.
        learn_rate = cfg.LEARNING_RATE * (
            cfg.FINAL_LR_MULTIPLE ** (epoch + 1 // cfg.EPOCHS)
        )
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        # Training
        model, avg_train_loss, train_mean_freq_err, train_cls_accuracy = train_model(
            model, train_loader, device, optimizer, criterion, epoch
        )

        if cfg.OPENSET_ENABLE:
            cfg.OPENSET_THRESHOLD = calibrate_open_set_threshold(
                model, train_loader, device
            )

        # Validation
        avg_val_loss, val_mean_freq_err, val_cls_accuracy, val_frames = validate_model(
            model, val_loader, device, criterion, epoch
        )

        # Print metrics for this epoch
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        print(
            f"  Train: Loss={avg_train_loss:.4f},"
            f"  MeanFreqErr={train_mean_freq_err:.4f},"
            f"  ClsAcc={train_cls_accuracy:.2f}%"
        )
        print(
            f"  Valid: Loss={avg_val_loss:.4f},"
            f"  MeanFreqErr={val_mean_freq_err:.4f},"
            f"  ClsAcc={val_cls_accuracy:.2f}%"
        )

        # Print a random subset of "frames"
        if cfg.VAL_PRINT_SAMPLES > 0:
            random.shuffle(val_frames)
            to_print = val_frames[
                : cfg.VAL_PRINT_SAMPLES
            ]  # up to VAL_PRINT_SAMPLES frames
            print(
                f"\n  Some random frames from validation (only {cfg.VAL_PRINT_SAMPLES} shown):"
            )
            print(f"  Prediction format: (frequency, class, confidence)")
            print(f"  GroundTruth format: (frequency, class)")
            for idx, frame_dict in enumerate(to_print, 1):
                pred_list = frame_dict["pred_list"]
                gt_list = frame_dict["gt_list"]

                print(f"    Frame {idx}:")
                print(f"      Predicted => {pred_list}")
                print(f"      GroundTruth=> {gt_list}")
            print("")

        if cfg.MULTIPLE_JOBS_PER_TRAINING:
            # Save model every epoch
            torch.save(model.state_dict(), f"{SAVE_MODEL_NAME}_epoch_{epoch+1}.pth")

            # Delete the previous model epoch to save space
            if epoch > 0:
                os.remove(f"{SAVE_MODEL_NAME}_epoch_{epoch}.pth")

    # 4) Test the model
    test_model(model, test_loader, device)


def train_model(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    total_train_loss = 0.0

    # For metrics:
    train_obj_count = 0
    train_correct_cls = 0
    train_sum_freq_err = 0.0

    for time_data, freq_data, label_tensor, _ in tqdm(
        train_loader, desc=f"Training epoch {epoch+1}/{cfg.EPOCHS}"
    ):
        time_data = time_data.to(device, non_blocking=True)
        freq_data = freq_data.to(device, non_blocking=True)
        label_tensor = label_tensor.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(time_data, freq_data)
        loss = criterion(pred, label_tensor)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Additional training metrics
        bsize = pred.shape[0]
        pred_reshape = pred.view(bsize, pred.shape[1], -1, (1 + 1 + cfg.NUM_CLASSES))
        x_pred = pred_reshape[..., 0]
        class_pred = pred_reshape[..., 2:]

        x_tgt = label_tensor[..., 0]
        conf_tgt = label_tensor[..., 1]
        class_tgt = label_tensor[..., 2:]

        obj_mask = conf_tgt > 0
        freq_err = (x_pred - x_tgt).abs()

        pred_class_idx = class_pred.argmax(dim=-1)
        true_class_idx = class_tgt.argmax(dim=-1)

        batch_obj_count = obj_mask.sum()
        batch_sum_freq_err = freq_err[obj_mask].sum()
        batch_correct_cls = (pred_class_idx[obj_mask] == true_class_idx[obj_mask]).sum()

        train_obj_count += batch_obj_count.item()
        train_sum_freq_err += batch_sum_freq_err.item()
        train_correct_cls += batch_correct_cls.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_mean_freq_err = train_sum_freq_err / train_obj_count
    train_cls_accuracy = 100.0 * (train_correct_cls / train_obj_count)

    return model, avg_train_loss, train_mean_freq_err, train_cls_accuracy


def validate_model(model, val_loader, device, criterion, epoch):
    model.eval()
    total_val_loss = 0.0

    val_obj_count = 0
    val_correct_cls = 0
    val_sum_freq_err = 0.0

    val_frames = []
    class_list = cfg.MODULATION_CLASSES

    with torch.no_grad():
        for time_data, freq_data, label_tensor, _ in tqdm(
            val_loader, desc=f"Validation epoch {epoch+1}/{cfg.EPOCHS}"
        ):
            time_data = time_data.to(device, non_blocking=True)
            freq_data = freq_data.to(device, non_blocking=True)
            label_tensor = label_tensor.to(device, non_blocking=True)

            # forward + loss
            pred = model(time_data, freq_data)
            loss = criterion(pred, label_tensor)
            total_val_loss += loss.item()

            # reshape into [B, S, B, 1+1+NUM_CLASSES]
            Bsz = pred.shape[0]
            pred = pred.view(Bsz, cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES)

            # pull out the raw pieces
            x_pred = pred[..., 0]
            conf_pred = pred[..., 1]
            class_logits = pred[..., 2:]  # raw scores

            x_tgt = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt = label_tensor[..., 2:]

            obj_mask = conf_tgt > 0
            freq_err = (x_pred - x_tgt).abs()

            # for overall metrics (unmodified)
            batch_obj_count = obj_mask.sum()
            batch_sum_freq_err = freq_err[obj_mask].sum()
            # plain argmax on logits for accuracy
            pred_class_idx = class_logits.argmax(dim=-1)
            true_class_idx = class_tgt.argmax(dim=-1)
            batch_correct_cls = (
                pred_class_idx[obj_mask] == true_class_idx[obj_mask]
            ).sum()

            val_obj_count += batch_obj_count.item()
            val_sum_freq_err += batch_sum_freq_err.item()
            val_correct_cls += batch_correct_cls.item()

            # now build per‐sample pred_list / gt_list for printing
            for i in range(Bsz):
                pred_list = []
                gt_list = []

                for s_idx in range(cfg.S):
                    for b_idx in range(cfg.B):
                        # 1) reconstruct raw frequency in Hz
                        off = x_pred[i, s_idx, b_idx].item()
                        freq_hz = (s_idx * cfg.SAMPLING_FREQUENCY / cfg.S) + off * (
                            cfg.SAMPLING_FREQUENCY / cfg.S
                        )

                        conf = conf_pred[i, s_idx, b_idx].item()

                        # 2) open‐set softmax check
                        logits = class_logits[i, s_idx, b_idx, :]  # shape [NUM_CLASSES]
                        probs = F.softmax(logits, dim=0)
                        pmax, cls_idx = torch.max(probs, dim=0)

                        if pmax.item() < cfg.OPENSET_THRESHOLD:
                            cls_str = "unknown"
                        else:
                            cls_str = class_list[int(cls_idx)]

                        # 3) only keep if conf > objectness threshold
                        if conf > cfg.CONFIDENCE_THRESHOLD:
                            # convert to human‐readable freq
                            freq_str, _ = convert_to_readable(
                                freq_hz, cls_idx, class_list
                            )
                            pred_list.append((freq_str, cls_str, float(pmax)))

                        # 4) ground‐truth
                        if conf_tgt[i, s_idx, b_idx] > 0:
                            off_g = x_tgt[i, s_idx, b_idx].item()
                            freq_hz_g = (
                                s_idx * cfg.SAMPLING_FREQUENCY / cfg.S
                            ) + off_g * (cfg.SAMPLING_FREQUENCY / cfg.S)
                            cls_g_idx = true_class_idx[i, s_idx, b_idx].item()
                            freq_str_g, cls_str_g = convert_to_readable(
                                freq_hz_g, cls_g_idx, class_list
                            )
                            gt_list.append((freq_str_g, cls_str_g))

                val_frames.append({"pred_list": pred_list, "gt_list": gt_list})

    avg_val_loss = total_val_loss / len(val_loader)
    val_mean_freq_err = val_sum_freq_err / val_obj_count
    val_cls_accuracy = 100.0 * (val_correct_cls / val_obj_count)

    return avg_val_loss, val_mean_freq_err, val_cls_accuracy, val_frames


def test_model(model, test_loader, device):
    """
    1) Test the model on the test set, gather classification accuracy, freq error, etc.
    2) Also compute these metrics per SNR
    3) Plot confusion matrix of classification
    """

    model.eval()
    total_obj_count = 0
    total_correct_cls = 0
    total_freq_err = 0.0

    # For confusion matrix
    overall_true_classes = []
    overall_pred_classes = []

    # For per-SNR stats
    snr_obj_count = {}
    snr_correct_cls = {}
    snr_freq_err = {}

    with torch.no_grad():
        for time_data, freq_data, label_tensor, snr_tensor in tqdm(
            test_loader, desc=f"Testing on test set"
        ):
            time_data = time_data.to(device)
            freq_data = freq_data.to(device)
            label_tensor = label_tensor.to(device)
            pred = model(time_data, freq_data)  # shape [batch, S, B*(1+1+NUM_CLASSES)]

            # reshape
            bsize = pred.shape[0]
            Sdim = pred.shape[1]  # should be S
            # interpret bounding boxes
            pred_reshape = pred.view(bsize, Sdim, -1, (1 + 1 + cfg.NUM_CLASSES))

            x_pred = pred_reshape[..., 0]  # [bsize, S, B]
            class_pred = pred_reshape[..., 2:]

            x_tgt = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt = label_tensor[..., 2:]

            # object mask
            obj_mask = conf_tgt > 0
            freq_err = (x_pred - x_tgt).abs()

            # new: softmax‐threshold + “unknown” class
            true_class_idx = class_tgt.argmax(dim=-1)  # [B,S,B]
            for i in range(bsize):
                for si in range(cfg.S):
                    for bi in range(cfg.B):
                        if not obj_mask[i, si, bi]:
                            continue

                        # ground truth index: 0..NUM_CLASSES-1
                        overall_true_classes.append(
                            int(true_class_idx[i, si, bi].item())
                        )

                        # predicted:
                        logits = class_pred[i, si, bi]  # shape [NUM_CLASSES]
                        probs = F.softmax(logits, dim=0)
                        pmax, cidx = probs.max(dim=0)
                        if pmax.item() < cfg.OPENSET_THRESHOLD:
                            overall_pred_classes.append(UNKNOWN_CLASS_IDX)
                        else:
                            overall_pred_classes.append(int(cidx.item()))

            # Now do per-SNR
            # We have a single snr per "sample" => shape [bsize]
            # but we have multiple bounding boxes => we can count them all with that same SNR
            snrs = snr_tensor.numpy()  # shape [bsize]
            for i in range(bsize):
                sample_snr = snrs[i]
                # bounding boxes belonging to sample i => obj_mask[i]
                # gather # of obj_mask=1 in that sample
                sample_obj_mask = obj_mask[i]  # shape [S, B]
                sample_obj_count = sample_obj_mask.sum().item()
                if sample_obj_count > 0:
                    sample_freq_err = freq_err[i][sample_obj_mask].sum().item()
                    sample_correct_cls = (
                        correct_cls_mask[i][sample_obj_mask].sum().item()
                    )

                    if sample_snr not in snr_obj_count:
                        snr_obj_count[sample_snr] = 0
                        snr_correct_cls[sample_snr] = 0
                        snr_freq_err[sample_snr] = 0.0

                    snr_obj_count[sample_snr] += sample_obj_count
                    snr_correct_cls[sample_snr] += sample_correct_cls
                    snr_freq_err[sample_snr] += sample_freq_err

    # 1) Overall
    overall_cls_acc = 100.0 * total_correct_cls / total_obj_count
    overall_freq_err = total_freq_err / total_obj_count

    print("\n=== TEST SET RESULTS ===")
    print(f"Overall bounding boxes: {total_obj_count}")
    print(f"Classification Accuracy (overall): {overall_cls_acc:.2f}%")
    print(f"Mean Frequency Error (overall): {overall_freq_err:.4f}")

    # 2) Per-SNR
    snr_keys_sorted = sorted(snr_obj_count.keys())
    for snr_val in snr_keys_sorted:
        cls_acc_snr = 100.0 * snr_correct_cls[snr_val] / snr_obj_count[snr_val]
        freq_err_snr = snr_freq_err[snr_val] / snr_obj_count[snr_val]

        print(
            f"SNR {snr_val:.1f}:  Accuracy={cls_acc_snr:.2f}%,  FreqErr={freq_err_snr:.4f}"
        )

    # 3) Confusion matrix
    if cfg.GENERATE_CONFUSION_MATRIX:
        class_list = cfg.MODULATION_CLASSES
        plot_confusion_matrix(class_list, overall_true_classes, overall_pred_classes)

    out_dir = os.path.join(data_dir, "../test_result_plots")
    # clear or create directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 4) Plot frequency domain diagram of test set samples and predictions
    if cfg.PLOT_TEST_SAMPLES:
        plot_test_samples(model, test_loader, device, out_dir)

    # 5) Write the test results to a file.
    if cfg.WRITE_TEST_RESULTS:
        write_test_results(model, test_loader, device, out_dir)


def plot_confusion_matrix(class_list, overall_true_classes, overall_pred_classes):
    cm = confusion_matrix(
        overall_true_classes, overall_pred_classes, labels=range(len(class_list))
    )
    cm_percent = cm.astype(float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = (cm[i] / row_sum) * 100.0

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_list,
        yticklabels=class_list,
    )
    plt.title("Test Set Confusion Matrix (%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png")
    plt.close()


def plot_test_samples(model, test_loader, device, out_dir):
    """
    For each sample in the test set: compute and plot its PSD,
    then draw vertical lines for each true center frequency (black dashed)
    and for each prediction (red solid), with labels showing
    predicted class, true class, and freq error below the line.
    Figures are saved as PNGs under data_dir/test_result_plots/.
    """

    # loop with progress bar
    sample_idx = 0
    for batch in tqdm(test_loader, desc="Plotting test samples"):
        time_data, freq_data, label_tensor, snr_tensor = batch
        bsz = time_data.size(0)
        for b in range(bsz):
            # reconstruct complex IQ
            x_wide = time_data[b].cpu().numpy()  # shape [2, N]
            I, Q = x_wide
            x_complex = I + 1j * Q

            # FFT + shift + PSD
            X = np.fft.fft(x_complex)
            X = np.fft.fftshift(X)
            freqs = np.fft.fftshift(
                np.fft.fftfreq(len(x_complex), d=1 / cfg.SAMPLING_FREQUENCY)
            )
            PSD = 10 * np.log10(np.abs(X) ** 2 + 1e-12)

            # extend & keep pos freqs
            freqs_ext = np.concatenate((freqs, freqs + cfg.SAMPLING_FREQUENCY))
            PSD_ext = np.concatenate((PSD, PSD))
            idx_pos = freqs_ext >= 0
            freqs_fin = freqs_ext[idx_pos]
            PSD_fin = PSD_ext[idx_pos]

            # run model for this one sample
            with torch.no_grad():
                pred = model(
                    time_data[b : b + 1].to(device),
                    freq_data[b : b + 1].to(device),
                )
            # reshape to [1, S, B, 1+1+NUM_CLASSES]
            pred = pred.view(1, cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES).cpu().numpy()[0]
            gt = label_tensor[b].view(cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES).numpy()

            # extract GT freqs & classes
            gt_lines = []
            for si in range(cfg.S):
                for bi in range(cfg.B):
                    if gt[si, bi, 1] > 0:  # confidence>0
                        xg = gt[si, bi, 0]
                        fg = (si + xg) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                        cls_g = np.argmax(gt[si, bi, 2:])
                        gt_lines.append((fg, cfg.MODULATION_CLASSES[cls_g]))

            # extract preds above a threshold
            pred_lines = []
            for si in range(cfg.S):
                for bi in range(cfg.B):
                    conf_p = pred[si, bi, 1]
                    if conf_p > cfg.CONFIDENCE_THRESHOLD:
                        xp = pred[si, bi, 0]
                        fp = (si + xp) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                        cls_p = np.argmax(pred[si, bi, 2:])

                        # find closest GT for error
                        if gt_lines:
                            errs = [abs(fp - g[0]) for g in gt_lines]
                            err = min(errs)
                        else:
                            err = np.nan

                        logits_np = pred[si, bi, 2:]
                        logits = torch.from_numpy(logits_np).float()
                        probs = F.softmax(logits, dim=0)
                        pmax, cidx = torch.max(probs, dim=0)

                        if pmax.item() < cfg.OPENSET_THRESHOLD:
                            cls_str = "unknown"
                        else:
                            cls_str = cfg.MODULATION_CLASSES[int(cidx)]

                        pred_lines.append((fp, cls_str, err))

            # plotting
            plt.figure()
            plt.plot(freqs_fin, PSD_fin)
            # title with SNR and count
            snr = snr_tensor[b].item()
            plt.title(f"SNR = {snr:.1f}; Center_freqs = {len(gt_lines)}")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [dB]")

            texts = []

            # draw GT
            for fg, cls_g in gt_lines:
                plt.axvline(fg, linestyle="--", color="black", alpha=0.7)
                texts.append(
                    plt.text(
                        fg,
                        PSD_fin.min(),
                        f"GT:{cls_g}",
                        va="top",
                        ha="center",
                    )
                )

            # draw preds
            for fp, cls_p, err in pred_lines:
                plt.axvline(fp, linestyle="-", color="red", alpha=0.7)
                texts.append(
                    plt.text(
                        fp,
                        PSD_fin.min(),
                        f"P:{cls_p}",
                        va="bottom",
                        ha="center",
                    )
                )

            adjust_text(texts)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{sample_idx:04d}.png"))
            plt.close()
            sample_idx += 1


def write_test_results(model, test_loader, device, out_dir):
    """
    Run the model over every test sample, build readable pred/gt lists
    (including 'unknown' if max‐softmax < threshold) and write them
    sorted by descending SNR to test_results.txt
    """
    import torch.nn.functional as F

    entries = []
    model.eval()
    with torch.no_grad():
        for time_data, freq_data, label_tensor, snr_tensor in tqdm(
            test_loader, desc="Gathering test results"
        ):
            Bsz = time_data.size(0)
            preds = model(time_data.to(device), freq_data.to(device))
            # reshape
            preds = preds.view(Bsz, cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES).cpu().numpy()
            labels = label_tensor.view(
                Bsz, cfg.S, cfg.B, 1 + 1 + cfg.NUM_CLASSES
            ).numpy()

            for i in range(Bsz):
                snr = snr_tensor[i].item()
                pred_list = []
                gt_list = []

                # GROUND TRUTH
                for si in range(cfg.S):
                    for bi in range(cfg.B):
                        if labels[i, si, bi, 1] > 0:
                            off_g = labels[i, si, bi, 0]
                            freq_hz_g = (
                                si * cfg.SAMPLING_FREQUENCY / cfg.S
                            ) + off_g * (cfg.SAMPLING_FREQUENCY / cfg.S)
                            cls_idx_g = int(np.argmax(labels[i, si, bi, 2:]))
                            freq_str_g, cls_str_g = convert_to_readable(
                                freq_hz_g, cls_idx_g, cfg.MODULATION_CLASSES
                            )
                            gt_list.append((freq_str_g, cls_str_g))

                # PREDICTIONS
                for si in range(cfg.S):
                    for bi in range(cfg.B):
                        conf_p = preds[i, si, bi, 1]
                        if conf_p <= cfg.CONFIDENCE_THRESHOLD:
                            continue

                        # raw logits slice
                        logits = torch.from_numpy(preds[i, si, bi, 2:]).float()
                        probs = F.softmax(logits, dim=0)
                        pmax, cls_idx = torch.max(probs, dim=0)

                        # open‐set decision
                        if pmax.item() < cfg.OPENSET_THRESHOLD:
                            cls_str = "unknown"
                        else:
                            cls_str = cfg.MODULATION_CLASSES[int(cls_idx)]

                        # freq back to Hz
                        off_p = preds[i, si, bi, 0]
                        freq_hz_p = (si * cfg.SAMPLING_FREQUENCY / cfg.S) + off_p * (
                            cfg.SAMPLING_FREQUENCY / cfg.S
                        )
                        freq_str_p, _ = convert_to_readable(
                            freq_hz_p, 0, cfg.MODULATION_CLASSES
                        )

                        pred_list.append((freq_str_p, cls_str, float(pmax)))

                entries.append((snr, len(gt_list), pred_list, gt_list))

    # sort by descending SNR
    entries.sort(key=lambda x: x[0], reverse=True)

    # write to file
    out_file = os.path.join(out_dir, "test_results.txt")
    with open(out_file, "w") as f:
        for snr, ntx, pred_list, gt_list in entries:
            f.write(f"SNR {snr:.1f}; Center_freqs = {ntx}\n")
            f.write(f"  Predicted => {pred_list}\n")
            f.write(f"  GroundTruth=> {gt_list}\n\n")


if __name__ == "__main__":
    main()
