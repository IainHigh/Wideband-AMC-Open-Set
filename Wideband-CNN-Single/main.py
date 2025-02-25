import torch
import torch.optim as optim
import torch.nn as nn
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from CNN import ModulationClassifier
from ModulationDataset import WidebandModulationDataset as ModulationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multi_signal_cost(predictions, gt_freqs, gt_mods, device):
    """
    predictions: list[(freq, mod)]
    gt_freqs: list[float]
    gt_mods: list[int]
    returns torch scalar cost
    """
    G = len(gt_freqs)
    P = len(predictions)
    if G == 0 and P == 0:
        return torch.tensor(0.0, device=device)
    if G == 0 and P > 0:
        return 2.0 * P * torch.tensor(1.0, device=device)
    if G > 0 and P == 0:
        return 5.0 * G * torch.tensor(1.0, device=device)

    freq_min = min(gt_freqs)
    freq_max = max(gt_freqs)
    freq_range = freq_max - freq_min if freq_max > freq_min else 1.0

    cost_mat = torch.zeros((G, P), device=device)
    for g in range(G):
        freq_gt = gt_freqs[g]
        mod_gt = gt_mods[g]
        for p in range(P):
            freq_pred, mod_pred = predictions[p]
            freq_err = torch.tensor(
                abs(freq_pred - freq_gt) / freq_range, device=device
            )
            mod_err = 0.0 if (mod_pred == mod_gt) else 1.0
            cost_mat[g, p] = freq_err + mod_err

    cost_np = cost_mat.cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    cval = cost_mat[row_idx, col_idx].sum()

    unmatched_g = G - len(row_idx)
    unmatched_p = P - len(col_idx)
    cval += 5.0 * unmatched_g + 2.0 * unmatched_p
    return cval


def train_reinforce(train_loader, model, optimizer, epochs):
    """
    REINFORCE approach:
      - For each sample => we do a "stochastic_recursive_inference" that
        yields a set of (freq,mod) predictions AND a list of log probs for each
        discrete 'split vs stop' action.
      - We compute cost => cost_i
      - reward => - cost_i
      - we do sum_of_log_probs * reward
    """
    model.to(device)
    gamma = 1.0  # no discount, single-step approach

    for epoch in range(epochs):
        model.train()
        total_return = 0.0
        num_samples = 0
        for x_wide, meta_list, f_s, f_e in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            x_wide = x_wide.to(device)
            batch_size = x_wide.size(0)

            # We'll do a standard REINFORCE approach => sum of log prob * reward
            # We'll accumulate the "loss" for each sample, then average or sum over the batch
            batch_loss = torch.tensor(0.0, device=device)
            for i in range(batch_size):
                meta_i = meta_list[i]
                gt_freqs = meta_i["center_frequencies"]
                gt_mods = meta_i["modulation_classes"]

                f_start = f_s[i].item()
                f_end = f_e[i].item()

                log_probs_accum = []
                # We'll define a special function in the model for policy-based recursion
                # so we can store each action's log prob
                preds = model.stochastic_recursive_inference(
                    x_wide[i : i + 1], f_start, f_end, log_probs_accum
                )
                # compute cost
                cost_i = multi_signal_cost(preds, gt_freqs, gt_mods, device)
                reward_i = -cost_i  # negative cost
                # sum_of_log_probs
                sum_log_p = torch.stack(log_probs_accum).sum()
                # REINFORCE gradient: - reward * sum_log_p => we want to maximize reward
                # => loss = - reward * sum_log_p
                sample_loss = -reward_i * sum_log_p
                batch_loss = batch_loss + sample_loss

                total_return += reward_i.item()
                num_samples += 1

            # average
            batch_loss = batch_loss / batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        avg_return = total_return / num_samples if num_samples > 0 else 0
        print(f"[Epoch {epoch+1}/{epochs}] Average Return={avg_return:.4f}")
    return model


def test_model(test_loader, model):
    model.eval()
    total_correct = 0
    total_preds = 0
    with torch.no_grad():
        for x_wide, meta_list, f_s, f_e in test_loader:
            x_wide = x_wide.to(device)
            bsz = x_wide.size(0)
            for i in range(bsz):
                meta_i = meta_list[i]
                gt_freqs = meta_i["center_frequencies"]
                gt_mods = meta_i["modulation_classes"]

                preds = model.stochastic_recursive_inference(
                    x_wide[i : i + 1], f_s[i].item(), f_e[i].item(), []
                )
                # naive accuracy
                for _, pred_mod in preds:
                    total_preds += 1
                    if pred_mod in gt_mods:
                        total_correct += 1
    if total_preds > 0:
        print(f"Test mod accuracy => {100.*total_correct/total_preds:.2f}%")
    else:
        print("No predictions made.")


def wideband_collate_fn(batch):
    x_list, meta_list, fs_list, fe_list = [], [], [], []
    for xb, meta, fs, fe in batch:
        x_list.append(xb)
        meta_list.append(meta)
        fs_list.append(fs)
        fe_list.append(fe)
    x_out = torch.stack(x_list, dim=0)
    f_s = torch.tensor(fs_list, dtype=torch.float)
    f_e = torch.tensor(fe_list, dtype=torch.float)
    return x_out, meta_list, f_s, f_e


def main():
    from ModulationDataset import WidebandModulationDataset

    train_data = WidebandModulationDataset("./data/training")
    test_data = WidebandModulationDataset("./data/testing")

    train_loader = DataLoader(
        train_data, batch_size=4, shuffle=True, collate_fn=wideband_collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=4, shuffle=False, collate_fn=wideband_collate_fn
    )

    model = ModulationClassifier(num_classes=len(train_data.label_to_idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train with REINFORCE
    trained_model = train_reinforce(train_loader, model, optimizer, epochs=10)
    # Evaluate
    test_model(test_loader, trained_model)


if __name__ == "__main__":
    main()
