#############################################
# main.py
#############################################
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.linalg as linalg

import json
from uuid import uuid4
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt
import config_wideband_yolo as cfg
from scipy.stats import chi2
from shutil import rmtree
from warnings import filterwarnings
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from adjustText import adjust_text
from tqdm import tqdm
from dataset_wideband_yolo import WidebandYoloDataset
from model_and_loss_wideband_yolo import WidebandYoloModel, WidebandYoloLoss


# Ignore warning messages that we'd expect to see
# 1) NumPy’s “Casting complex values to real …” This is to be expected as we're converting the IQ data to real values.
filterwarnings("ignore", category=np.ComplexWarning)

# 2) PyTorch DataLoader’s “This DataLoader will create …” This is to be expected as we're using multiple workers for the DataLoader.
filterwarnings(
    "ignore",
    message=r"This DataLoader will create .* worker processes",
    category=UserWarning,
)


SAVE_MODEL_NAME = "yolo_model"

with open("./configs/system_parameters.json") as f:
    system_parameters = json.load(f)

working_directory = system_parameters["Working_Directory"]
sys.path.append(working_directory)

rng_seed = system_parameters["Random_Seed"]
data_dir = system_parameters["Dataset_Directory"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(rng_seed)
random.seed(rng_seed)
np.random.seed(rng_seed)

CELL_WIDTH = cfg.SAMPLING_FREQUENCY / cfg.S  # width of a YOLO “bin” in Hz

EMBED_DIM = 96


def maha_dist(x, mean, inv_cov):
    """x:(...,D) – class-cond. squared Mahalanobis distance."""
    diff = x - mean
    if inv_cov.dim() == 2:
        m = torch.einsum("...d,dc,...c->...", diff, inv_cov, diff)
    else:
        m = torch.einsum("...d,...dc,...c->...", diff, inv_cov, diff)
    return m


class_means = None  # tensor [NUM_CLASSES, EMBED_DIM]
inv_cov = None  # per-class inverse covariance  [NUM_CLASSES,D,D]

UNKNOWN_IDX = cfg.NUM_CLASSES


def convert_to_readable(frequency, modclass):
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
        frequency_string = f"{frequency} {size_map[int(size/1000)]}"
    else:
        frequency_string = f"{frequency} Hz"

    if modclass < len(cfg.MODULATION_CLASSES):
        modclass_str = cfg.MODULATION_CLASSES[modclass]
    else:
        modclass_str = cfg.UNKNOWN_CLASS_NAME

    return frequency_string, modclass_str


def _hz_from_offset(off, cell):
    """Convert a normalised offset and cell-index to an absolute frequency (Hz)."""
    return (cell + off) * CELL_WIDTH


def _collect_lists(x_pred, bw_pred, conf_pred, x_tgt, conf_tgt):
    """
    Build (pred, gt) lists **for ONE frame**.

    • prediction = (centre_Hz, bandwidth_Hz) for every box whose confidence
      exceeds `CONFIDENCE_THRESHOLD`.

    • ground truth = centre_Hz for every GT box (conf_tgt>0)
    """
    pred, gt = [], []
    for s in range(cfg.S):
        for b in range(cfg.B):
            if conf_pred[s, b] > cfg.CONFIDENCE_THRESHOLD:
                f = _hz_from_offset(x_pred[s, b], s)
                bw = bw_pred[s, b] * CELL_WIDTH
                pred.append((f, bw))

            if conf_tgt[s, b] > 0:
                f = _hz_from_offset(x_tgt[s, b], s)
                gt.append(f)
    return pred, gt


def _tp_fp_fn(pred, gt):
    """
    Greedy 1-to-1 matching based on the rule

        |f_gt − f_pred| ≤ bw_pred / 2     ⇒ TP

    Unmatched predictions → FP, unmatched GT → FN
    """
    tp = fp = 0
    remaining = gt.copy()
    for f_pred, bw_pred in pred:
        matched = False
        for i, f_gt in enumerate(remaining):
            if abs(f_gt - f_pred) <= bw_pred / 2:
                tp += 1
                remaining.pop(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(remaining)
    return tp, fp, fn


def main():
    # Print the configuration file
    if cfg.PRINT_CONFIG_FILE:
        cfg.print_config_file()

    # 1) Build dataset and loaders
    train_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "training"), transform=None
    )

    cfg.MODULATION_CLASSES = train_dataset.class_list

    val_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "validation"),
        transform=None,
        class_list=cfg.MODULATION_CLASSES,
    )
    test_dataset = WidebandYoloDataset(
        os.path.join(data_dir, "testing"),
        transform=None,
        class_list=cfg.MODULATION_CLASSES,
    )

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
    criterion = WidebandYoloLoss().to(device)

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
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                    criterion.load_state_dict(checkpoint["criterion"])
                else:
                    # Backwards compatibility with checkpoints that only saved the model
                    model.load_state_dict(checkpoint)
                model.to(device)
                start_epoch = i + 1
                print(f"Loaded model from epoch {i+1}")

    if start_epoch >= cfg.EPOCHS:
        print("Model training complete. No more epochs to train.")
        return

    # 3) Training loop
    for epoch in range(start_epoch, cfg.EPOCHS):
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}]")

        # Set the learning rate depending on the epoch. Starts at LEARNING_RATE and decreases by a factor of FINAL_LR_MULTIPLE by the last epoch.
        prog = epoch / (cfg.EPOCHS - 1) if cfg.EPOCHS > 1 else 0.0
        learn_rate = cfg.LEARNING_RATE * (cfg.FINAL_LR_MULTIPLE**prog)
        # Optimizer must update both the model and the loss centres so the class prototypes are learned along with the network weights.
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=learn_rate,
        )

        # Training
        (
            model,
            avg_train_loss,
            train_mean_freq_err,
            train_cls_accuracy,
            train_prec,
            train_rec,
            train_f1,
        ) = train_model(model, train_loader, device, optimizer, criterion, epoch)

        train_mean_freq_err = convert_to_readable(train_mean_freq_err, 0)[0]

        print(
            f"\tTrain: Loss={avg_train_loss:.4f}, "
            f"MeanFreqErr={train_mean_freq_err}, "
            f"ClsAcc={train_cls_accuracy:.2f}%, "
            f"P={train_prec:.3f}, R={train_rec:.3f}, F1={train_f1:.3f}"
        )

        if cfg.VALIDATE_MODEL:
            # Validation
            (
                avg_val_loss,
                val_mean_freq_err,
                val_cls_accuracy,
                val_prec,
                val_rec,
                val_f1,
                val_frames,
            ) = validate_model(model, val_loader, device, criterion, epoch)

            # Convert frequency errors to human readable format
            val_mean_freq_err = convert_to_readable(val_mean_freq_err, 0)[0]

            print(
                f"\tValid: Loss={avg_val_loss:.4f}, "
                f"MeanFreqErr={val_mean_freq_err}, "
                f"ClsAcc={val_cls_accuracy:.2f}%, "
                f"P={val_prec:.3f}, R={val_rec:.3f}, F1={val_f1:.3f}"
            )

            # Print a random subset of "frames"
            if cfg.VAL_PRINT_SAMPLES > 0:
                random.shuffle(val_frames)
                to_print = val_frames[
                    : cfg.VAL_PRINT_SAMPLES
                ]  # up to VAL_PRINT_SAMPLES frames
                print(
                    f"\n\tSome random frames from validation (only {cfg.VAL_PRINT_SAMPLES} shown):"
                )
                print(f"\tPrediction format: (frequency, class, confidence)")
                print(f"\tGroundTruth format: (frequency, class)")
                for idx, frame_dict in enumerate(to_print, 1):
                    pred_list = frame_dict["pred_list"]
                    gt_list = frame_dict["gt_list"]

                    print(f"\t\tFrame {idx}:")
                    print(f"\t\t\tPredicted => {pred_list}")
                    print(f"\t\t\tGroundTruth=> {gt_list}")
                print("")

        if cfg.MULTIPLE_JOBS_PER_TRAINING:
            # Save model (and loss centres) every epoch for multi-job training
            torch.save(
                {
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                },
                f"{SAVE_MODEL_NAME}_epoch_{epoch+1}.pth",
            )

    # 4) Test the model
    test_model(model, test_loader, device)


def train_model(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    if cfg.DETAILED_LOSS_PRINT:
        criterion.reset_epoch_stats()

    total_train_loss = 0.0

    # For metrics:
    train_obj_count = 0
    train_correct_cls = 0
    train_sum_freq_err = 0.0

    train_tp = train_fp = train_fn = 0

    emb_acc = [[] for _ in range(cfg.NUM_CLASSES)]

    for time_data, freq_data, label_tensor, _ in tqdm(
        train_loader, desc=f"Training epoch {epoch+1}/{cfg.EPOCHS}"
    ):
        time_data = time_data.to(device, non_blocking=True)
        freq_data = freq_data.to(device, non_blocking=True)
        label_tensor = label_tensor.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred, emb = model(time_data, freq_data)
        bsize = pred.shape[0]
        loss = criterion(pred, label_tensor, emb)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.anchors.data.clamp_(0.0, 1.0)
        total_train_loss += loss.item()

        pred_r = pred.view(bsize, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES)
        tgt_r = label_tensor.view_as(pred_r)

        x_pred = pred_r[..., 0].cpu()
        bw_pred = pred_r[..., 2].cpu()
        conf_pr = pred_r[..., 1].cpu()

        x_tgt = tgt_r[..., 0].cpu()
        conf_tg = tgt_r[..., 1].cpu()

        for i in range(bsize):
            preds, gts = _collect_lists(
                x_pred[i], bw_pred[i], conf_pr[i], x_tgt[i], conf_tg[i]
            )
            tp, fp, fn = _tp_fp_fn(preds, gts)
            train_tp += tp
            train_fp += fp
            train_fn += fn

        # Additional training metrics
        pred_reshape = pred.view(
            bsize, pred.shape[1], -1, (1 + 1 + 1 + cfg.NUM_CLASSES)
        )
        x_pred = pred_reshape[..., 0]
        class_pred = pred_reshape[..., 3:]

        x_tgt = label_tensor[..., 0]
        conf_tgt = label_tensor[..., 1]
        class_tgt = label_tensor[..., 3:]

        obj_mask = conf_tgt > 0

        freq_err = (x_pred - x_tgt).abs()

        # Convert freq_err to Hz
        freq_err = freq_err * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
        pred_class_idx = class_pred.argmax(dim=-1)
        true_class_idx = class_tgt.argmax(dim=-1)

        # ---- accumulate embeddings for class Gaussians ----
        if cfg.OPENSET_ENABLE:
            with torch.no_grad():
                # only GT boxes
                embs_this = emb[obj_mask].cpu()
                labels_this = true_class_idx[obj_mask].cpu()
                for c in range(cfg.NUM_CLASSES):
                    idx = labels_this == c
                    if idx.any():
                        emb_acc[c].append(embs_this[idx])

        batch_obj_count = obj_mask.sum()
        batch_sum_freq_err = freq_err[obj_mask].sum()
        batch_correct_cls = (pred_class_idx[obj_mask] == true_class_idx[obj_mask]).sum()

        train_obj_count += batch_obj_count.item()
        train_sum_freq_err += batch_sum_freq_err.item()
        train_correct_cls += batch_correct_cls.item()

    avg_train_loss = total_train_loss / len(train_loader)

    train_mean_freq_err = train_sum_freq_err / train_obj_count
    train_cls_accuracy = 100.0 * (train_correct_cls / train_obj_count)

    precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) else 0.0
    recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if cfg.OPENSET_ENABLE:
        with torch.no_grad():
            global class_means, inv_cov
            class_means = torch.zeros(cfg.NUM_CLASSES, EMBED_DIM)
            inv_cov = torch.zeros(cfg.NUM_CLASSES, EMBED_DIM, EMBED_DIM)

            for c, lst in enumerate(emb_acc):
                if not lst:
                    continue
                m = torch.cat(lst, 0)
                mean_c = m.mean(0)
                class_means[c] = mean_c
                diff = m - mean_c
                cov_c = (diff.T @ diff) / (m.size(0) - 1)
                inv_cov[c] = linalg.inv(cov_c + 1e-6 * torch.eye(cov_c.size(0)))

            q = chi2.ppf(cfg.OPENSET_COVERAGE, EMBED_DIM)
            cfg.OPENSET_THRESHOLD = torch.full((cfg.NUM_CLASSES,), q)
            class_means = class_means.to(device)
            inv_cov = inv_cov.to(device)

    if cfg.DETAILED_LOSS_PRINT:
        criterion.print_epoch_stats()
        criterion.reset_epoch_stats()

    return (
        model,
        avg_train_loss,
        train_mean_freq_err,
        train_cls_accuracy,
        precision,
        recall,
        f1,
    )


def validate_model(model, val_loader, device, criterion, epoch):
    model.eval()
    if cfg.DETAILED_LOSS_PRINT:
        criterion.reset_epoch_stats()
    total_val_loss = 0.0

    val_obj_count = 0
    val_correct_cls = 0
    val_sum_freq_err = 0.0

    val_frames = []
    val_tp = val_fp = val_fn = 0

    with torch.no_grad():
        for time_data, freq_data, label_tensor, _ in tqdm(
            val_loader, desc=f"Validation epoch {epoch+1}/{cfg.EPOCHS}"
        ):
            time_data = time_data.to(device, non_blocking=True)
            freq_data = freq_data.to(device, non_blocking=True)
            label_tensor = label_tensor.to(device, non_blocking=True)

            pred, emb = model(time_data, freq_data)
            bsize = pred.shape[0]
            loss = criterion(pred, label_tensor, emb)
            total_val_loss += loss.item()

            pred_r = pred.view(bsize, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES)
            tgt_r = label_tensor.view_as(pred_r)

            x_pred = pred_r[..., 0].cpu()
            bw_pred = pred_r[..., 2].cpu()
            conf_pr = pred_r[..., 1].cpu()

            x_tgt = tgt_r[..., 0].cpu()
            conf_tg = tgt_r[..., 1].cpu()

            for i in range(bsize):
                preds, gts = _collect_lists(
                    x_pred[i], bw_pred[i], conf_pr[i], x_tgt[i], conf_tg[i]
                )
                tp, fp, fn = _tp_fp_fn(preds, gts)
                val_tp += tp
                val_fp += fp
                val_fn += fn

            pred_reshape = pred.view(
                bsize, pred.shape[1], -1, (1 + 1 + 1 + cfg.NUM_CLASSES)
            )

            x_pred = pred_reshape[..., 0]
            conf_pred = pred_reshape[..., 1]
            class_pred = pred_reshape[..., 3:]

            x_tgt = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt = label_tensor[..., 3:]

            obj_mask = conf_tgt > 0
            freq_err = (x_pred - x_tgt).abs()
            # Convert freq_err to Hz
            freq_err = freq_err * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S

            pred_class_idx = class_pred.argmax(dim=-1)

            if cfg.OPENSET_ENABLE and cfg.OPENSET_THRESHOLD is not None:
                # Mahalanobis distances for *every* predicted box
                flat_idx = pred_class_idx.reshape(-1)
                means_sel = class_means[flat_idx].to(device)
                cov_sel = inv_cov[flat_idx].to(device)
                d2 = maha_dist(emb.reshape(-1, EMBED_DIM), means_sel, cov_sel)
                d2 = d2.view_as(pred_class_idx)
                tau = cfg.OPENSET_THRESHOLD.to(device)[pred_class_idx]
                unknown_mask_pred = d2 > tau
                pred_class_idx[unknown_mask_pred] = UNKNOWN_IDX

            true_class_idx = class_tgt.argmax(dim=-1)

            # GT boxes whose one-hot vector is all-zeros → “UNKNOWN”
            if cfg.OPENSET_ENABLE:
                gt_unknown_mask = class_tgt.sum(dim=-1) == 0
                true_class_idx[gt_unknown_mask] = UNKNOWN_IDX

            # For metric sums
            batch_obj_count = obj_mask.sum()
            batch_sum_freq_err = freq_err[obj_mask].sum()
            batch_correct_cls = (
                pred_class_idx[obj_mask] == true_class_idx[obj_mask]
            ).sum()

            val_obj_count += batch_obj_count.item()
            val_sum_freq_err += batch_sum_freq_err.item()
            val_correct_cls += batch_correct_cls.item()

            # For printing the results in a "frame" manner:
            # We group each sample in this batch separately.
            for i in range(bsize):
                pred_list = []
                gt_list = []

                for s_idx in range(pred_reshape.shape[1]):
                    for b_idx in range(pred_reshape.shape[2]):
                        x_p = x_pred[i, s_idx, b_idx].item()  # x_offset [0,1]
                        x_p = (s_idx * CELL_WIDTH) + x_p * (
                            CELL_WIDTH
                        )  # raw frequency value.

                        conf = conf_pred[i, s_idx, b_idx].item()
                        cls_p = pred_class_idx[i, s_idx, b_idx].item()
                        x_p, cls_p = convert_to_readable(x_p, cls_p)
                        if conf > cfg.CONFIDENCE_THRESHOLD:
                            pred_list.append((x_p, cls_p, conf))

                        if conf_tgt[i, s_idx, b_idx] > 0:
                            x_g = x_tgt[i, s_idx, b_idx].item()
                            x_g = (s_idx * CELL_WIDTH) + x_g * (
                                CELL_WIDTH
                            )  # raw frequency value.

                            cls_g = true_class_idx[i, s_idx, b_idx].item()
                            x_g, cls_g = convert_to_readable(x_g, cls_g)
                            gt_list.append((x_g, cls_g))

                frame_dict = {"pred_list": pred_list, "gt_list": gt_list}
                val_frames.append(frame_dict)

    avg_val_loss = total_val_loss / len(val_loader)
    val_mean_freq_err = val_sum_freq_err / val_obj_count
    val_cls_accuracy = 100.0 * (val_correct_cls / val_obj_count)

    precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) else 0.0
    recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if cfg.DETAILED_LOSS_PRINT:
        criterion.print_epoch_stats()
        criterion.reset_epoch_stats()

    return (
        avg_val_loss,
        val_mean_freq_err,
        val_cls_accuracy,
        precision,
        recall,
        f1,
        val_frames,
    )


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

    overall_tp = overall_fp = overall_fn = 0
    snr_tp = {}
    snr_fp = {}
    snr_fn = {}

    with torch.no_grad():
        for time_data, freq_data, label_tensor, snr_tensor in tqdm(
            test_loader, desc=f"Testing on test set"
        ):
            time_data = time_data.to(device)
            freq_data = freq_data.to(device)
            label_tensor = label_tensor.to(device)
            pred, emb = model(time_data, freq_data)
            bsize = pred.shape[0]
            Sdim = pred.shape[1]  # should be S
            # interpret bounding boxes
            pred_reshape = pred.view(bsize, Sdim, -1, (1 + 1 + 1 + cfg.NUM_CLASSES))

            x_pred = pred_reshape[..., 0]  # [bsize, S, B]
            conf_pred = pred_reshape[..., 1]  # [bsize, S, B]
            bw_pred = pred_reshape[..., 2]  # [bsize, S, B]
            class_pred = pred_reshape[..., 3:]

            x_tgt = label_tensor[..., 0]
            conf_tgt = label_tensor[..., 1]
            class_tgt = label_tensor[..., 3:]

            # object mask
            obj_mask = conf_tgt > 0
            freq_err = (x_pred - x_tgt).abs()
            # Convert freq_err to Hz
            freq_err = freq_err * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S

            # predicted vs. true class => argmax
            pred_class_idx = class_pred.argmax(dim=-1)  # [bsize, S, B]

            if cfg.OPENSET_ENABLE and cfg.OPENSET_THRESHOLD is not None:
                # Mahalanobis distances for *every* predicted box
                flat_idx = pred_class_idx.reshape(-1)
                means_sel = class_means[flat_idx].to(device)
                cov_sel = inv_cov[flat_idx].to(device)
                d2 = maha_dist(emb.reshape(-1, EMBED_DIM), means_sel, cov_sel)
                d2 = d2.view_as(pred_class_idx)
                tau = cfg.OPENSET_THRESHOLD.to(device)[pred_class_idx]
                unknown_mask_pred = d2 > tau
                pred_class_idx[unknown_mask_pred] = UNKNOWN_IDX

            true_class_idx = class_tgt.argmax(dim=-1)
            # GT boxes whose one-hot vector is all-zeros → “UNKNOWN”
            if cfg.OPENSET_ENABLE:
                gt_unknown_mask = class_tgt.sum(dim=-1) == 0
                true_class_idx[gt_unknown_mask] = UNKNOWN_IDX

            # Now we accumulate stats for each bounding box with obj_mask=1
            batch_obj_count = obj_mask.sum()
            batch_sum_freq_err = freq_err[obj_mask].sum()
            correct_cls_mask = pred_class_idx == true_class_idx
            batch_correct_cls = (
                pred_class_idx[obj_mask] == true_class_idx[obj_mask]
            ).sum()

            total_obj_count += batch_obj_count.item()
            total_freq_err += batch_sum_freq_err.item()
            total_correct_cls += batch_correct_cls.item()

            # For confusion matrix, we flatten the bounding boxes =>
            # we only consider those bounding boxes with obj_mask=1
            # then we gather pred_class_idx[obj_mask] and true_class_idx[obj_mask]
            # convert to CPU
            pred_class_flat = pred_class_idx[obj_mask].cpu().numpy()
            true_class_flat = true_class_idx[obj_mask].cpu().numpy()
            overall_true_classes.extend(true_class_flat.tolist())
            overall_pred_classes.extend(pred_class_flat.tolist())

            # Now do per-SNR
            # We have a single snr per "sample" => shape [bsize]
            # but we have multiple bounding boxes => we can count them all with that same SNR
            snrs = snr_tensor.numpy()  # shape [bsize]
            for i in range(bsize):
                preds, gts = _collect_lists(
                    x_pred[i].cpu(),
                    bw_pred[i].cpu(),
                    conf_pred[i].cpu(),
                    x_tgt[i].cpu(),
                    conf_tgt[i].cpu(),
                )
                tp, fp, fn = _tp_fp_fn(preds, gts)

                overall_tp += tp
                overall_fp += fp
                overall_fn += fn

                sample_snr = snr_tensor[i].item()
                if sample_snr not in snr_tp:
                    snr_tp[sample_snr] = snr_fp[sample_snr] = snr_fn[sample_snr] = 0
                snr_tp[sample_snr] += tp
                snr_fp[sample_snr] += fp
                snr_fn[sample_snr] += fn
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

    # Convert overall_freq_err to human readable
    overall_freq_err = convert_to_readable(overall_freq_err, 0)[0]

    print("\n=== TEST SET RESULTS ===")
    print(f"Overall bounding boxes: {total_obj_count}")
    print(f"Classification Accuracy (overall): {overall_cls_acc:.2f}%")
    print(f"Mean Frequency Error (overall): {overall_freq_err}")

    overall_prec = (
        overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0.0
    )
    overall_rec = (
        overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0.0
    )
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec)
        else 0.0
    )

    print(f"Precision (overall) : {overall_prec:.3f}")
    print(f"Recall    (overall) : {overall_rec:.3f}")
    print(f"F1 score  (overall) : {overall_f1:.3f}\n")

    # 2) Per-SNR
    snr_keys_sorted = sorted(snr_obj_count.keys())
    for snr_val in snr_keys_sorted:
        cls_acc_snr = 100.0 * snr_correct_cls[snr_val] / snr_obj_count[snr_val]
        freq_err_snr = snr_freq_err[snr_val] / snr_obj_count[snr_val]

        # Convert freq_err_snr to human readable
        freq_err_snr = convert_to_readable(freq_err_snr, 0)[0]

        # If any of the snr_tp, snr_fp, snr_fn are empty at this snr, set them to 0 to avoid division by zero in the precision, recall, and f1 calculations.
        if snr_val not in snr_tp:
            snr_tp[snr_val] = 0
        if snr_val not in snr_fp:
            snr_fp[snr_val] = 0
        if snr_val not in snr_fn:
            snr_fn[snr_val] = 0

        prec = (
            snr_tp[snr_val] / (snr_tp[snr_val] + snr_fp[snr_val])
            if (snr_tp[snr_val] + snr_fp[snr_val])
            else 0.0
        )
        rec = (
            snr_tp[snr_val] / (snr_tp[snr_val] + snr_fn[snr_val])
            if (snr_tp[snr_val] + snr_fn[snr_val])
            else 0.0
        )
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        # keep the existing accuracy / freq-err print ↓ and append the new numbers
        print(
            f"SNR {snr_val:.1f}:  "
            f"Accuracy={cls_acc_snr:.2f}%,  FreqErr={freq_err_snr},  "
            f"P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}"
        )

    # 3) Confusion matrix
    if cfg.GENERATE_CONFUSION_MATRIX:
        plot_confusion_matrix(overall_true_classes, overall_pred_classes)

    if cfg.PLOT_TEST_SAMPLES or cfg.WRITE_TEST_RESULTS:
        out_dir = os.path.join(data_dir, "../test_result_plots")
        # clear or create directory
        if os.path.exists(out_dir):
            rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # 4) Plot frequency domain diagram of test set samples and predictions
    if cfg.PLOT_TEST_SAMPLES:
        plot_test_samples(model, test_loader, device, out_dir)

    # 5) Write the test results to a file.
    if cfg.WRITE_TEST_RESULTS:
        write_test_results(model, test_loader, device, out_dir)


def plot_confusion_matrix(overall_true_classes, overall_pred_classes):

    if cfg.OPENSET_ENABLE:
        class_list = cfg.MODULATION_CLASSES + [cfg.UNKNOWN_CLASS_NAME]
    else:
        class_list = cfg.MODULATION_CLASSES

    cm = confusion_matrix(
        overall_true_classes,
        overall_pred_classes,
        labels=range(len(class_list)),
    )
    cm_percent = cm.astype(float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = (cm[i] / row_sum) * 100.0

    plt.figure(figsize=(8, 6))
    heatmap(
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

    # IF the file already exists, save it with a uuid suffix
    if os.path.exists("confusion_matrix.png"):
        uuid = str(uuid4())
        plt.savefig(f"confusion_matrix_{uuid}.png")
        print(f"Saved confusion matrix as confusion_matrix_{uuid}.png")
    else:
        plt.savefig("confusion_matrix.png")
        print(f"Saved confusion matrix as confusion_matrix.png")
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

            # Cut off frequencies above the sampling frequency
            PSD_fin = PSD_fin[freqs_fin <= cfg.SAMPLING_FREQUENCY]
            freqs_fin = freqs_fin[freqs_fin <= cfg.SAMPLING_FREQUENCY]

            if PSD_fin.size == 0 or freqs_fin.size == 0:
                continue

            # run model for this one sample
            with torch.no_grad():
                pred = model(
                    time_data[b : b + 1].to(device),
                    freq_data[b : b + 1].to(device),
                )
            # reshape to [1, S, B, 1+1+NUM_CLASSES]
            pred = (
                pred.view(1, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES).cpu().numpy()[0]
            )
            gt = label_tensor[b].view(cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES).numpy()

            # extract GT freqs & classes
            gt_lines = []
            for si in range(cfg.S):
                for bi in range(cfg.B):
                    if gt[si, bi, 1] > 0:  # confidence>0
                        xg = gt[si, bi, 0]
                        fg = (si + xg) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                        cls_g = np.argmax(gt[si, bi, 3:])
                        gt_lines.append((fg, cfg.MODULATION_CLASSES[cls_g]))

            # extract preds above a threshold
            pred_lines = []
            for si in range(cfg.S):
                for bi in range(cfg.B):
                    conf_p = pred[si, bi, 1]
                    if conf_p > cfg.CONFIDENCE_THRESHOLD:
                        xp = pred[si, bi, 0]
                        fp = (si + xp) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                        cls_p = np.argmax(pred[si, bi, 3:])
                        bandwidth = pred[si, bi, 2]
                        bandwidth = bandwidth * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                        pred_lines.append(
                            (fp, cfg.MODULATION_CLASSES[cls_p], bandwidth)
                        )

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
                plt.axvline(fg, linestyle="--", color="black", alpha=1.0)
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
            for fp, cls_p, bandwidth in pred_lines:

                ax = plt.gca()

                # vertical line at the predicted centre
                ax.axvline(fp, linestyle="-", color="red", alpha=1.0)

                # shaded bandwidth span
                ax.axvspan(
                    fp - bandwidth / 2,
                    fp + bandwidth / 2,
                    color="red",
                    alpha=0.30,
                    zorder=5,
                )

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
    and write them sorted by descending SNR to
      data_dir/test_result_plots/test_results.txt
    """

    entries = []
    model.eval()
    with torch.no_grad():
        for time_data, freq_data, label_tensor, snr_tensor in tqdm(
            test_loader, desc="Gathering test results"
        ):
            bsz = time_data.size(0)
            # run model once per batch
            preds = model(time_data.to(device), freq_data.to(device))
            # reshape to [batch, S, B, 1+1+NUM_CLASSES]
            preds = (
                preds.view(bsz, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES).cpu().numpy()
            )
            labels = label_tensor.view(
                bsz, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES
            ).numpy()
            for i in range(bsz):
                snr = snr_tensor[i].item()
                pred_list = []
                gt_list = []

                # build GT list
                for si in range(cfg.S):
                    for bi in range(cfg.B):
                        if labels[i, si, bi, 1] > 0:
                            xg_norm = labels[i, si, bi, 0]
                            fg = (si + xg_norm) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                            cls_idx = np.argmax(labels[i, si, bi, 3:])
                            freq_str, cls_str = convert_to_readable(fg, cls_idx)
                            gt_list.append((freq_str, cls_str))

                # build Pred list
                for si in range(cfg.S):
                    for bi in range(cfg.B):
                        conf_p = preds[i, si, bi, 1]
                        if conf_p > cfg.CONFIDENCE_THRESHOLD:
                            xp_norm = preds[i, si, bi, 0]
                            fp = (si + xp_norm) * (cfg.SAMPLING_FREQUENCY / 2) / cfg.S
                            cls_idx = np.argmax(preds[i, si, bi, 3:])
                            freq_str, cls_str = convert_to_readable(fp, cls_idx)
                            pred_list.append((freq_str, cls_str, conf_p))

                entries.append((snr, len(gt_list), pred_list, gt_list))

    # sort by SNR descending
    entries.sort(key=lambda x: x[0], reverse=True)

    # write to file
    out_file = os.path.join(out_dir, "test_results.txt")
    with open(out_file, "w") as f:
        for _, (snr, ntx, pred_list, gt_list) in enumerate(entries, 1):
            f.write(f"SNR {snr:.1f}; Center_freqs = {ntx}\n")
            f.write(f"  Predicted => {pred_list}\n")
            f.write(f"  GroundTruth=> {gt_list}\n\n")


if __name__ == "__main__":
    main()
