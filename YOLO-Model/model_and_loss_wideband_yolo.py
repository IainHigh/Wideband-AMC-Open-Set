#############################################
# model_and_loss_wideband_yolo.py
#############################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import pi
from config_wideband_yolo import (
    S,
    B,
    NUM_CLASSES,
    IOU_LOSS,
    LAMBDA_NOOBJ,
    LAMBDA_CLASS,
    LAMBDA_CENTER,
    LAMBDA_CENTER_SEP,
    CONFIDENCE_THRESHOLD,
    DETAILED_LOSS_PRINT,
    NUMTAPS,
    SAMPLING_FREQUENCY,
    MERGE_SIMILAR_PREDICTIONS,
    MERGE_SIMILAR_PREDICTIONS_THRESHOLD,
    get_anchors,
)


def conv1d_batch(x, weight, pad_left, pad_right):
    x_padded = F.pad(x, (pad_left, pad_right))
    x_unf = x_padded.unfold(dimension=2, size=weight.shape[-1], step=1)
    weight = weight.unsqueeze(2)
    y = (x_unf * weight).sum(dim=-1)
    return y


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2),
            nn.BatchNorm1d(out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        res = self.residual(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        out = F.relu(concat + res)
        return out


class WidebandClassifier(nn.Module):
    def __init__(self, num_out):
        """
        num_out: output dimension, here (1 + NUM_CLASSES) where the first element is confidence.
        """
        super(WidebandClassifier, self).__init__()
        # Conv Block 1 (as in narrowband model)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # Block 2 repeated 4 times
        self.block2_layers = nn.ModuleList(
            [self._create_block2(32 if i == 0 else 96, 96) for i in range(4)]
        )
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 96-D embedding  →  (1+NUM_CLASSES) logits
        self.fc = nn.Linear(96, num_out)

    def _create_block2(self, in_channels, out_channels):
        return nn.ModuleDict(
            {
                "branch1": nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=1, stride=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch2": nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch3": nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=1, stride=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "residual": nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
                    nn.BatchNorm1d(out_channels),
                ),
            }
        )

    def forward(self, x, *, return_embedding: bool = False):
        # x: [N, 2, T] where T is the length of the downconverted signal.
        x = self.conv_block1(x)
        for block in self.block2_layers:
            residual = block["residual"](x)
            branch1 = block["branch1"](x)
            branch2 = block["branch2"](x)
            branch3 = block["branch3"](x)
            concatenated = torch.cat([branch1, branch2, branch3], dim=1)
            x = F.relu(concatenated + residual)
        x = self.global_avg_pool(x)
        feat = x.view(x.size(0), -1)  # (B,96)
        logits = self.fc(feat)
        return (logits, feat) if return_embedding else logits


class WidebandYoloModel(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

        # -----------------------
        # Stage-1: Frequency Prediction
        # -----------------------
        # Time-domain branch.
        self.first_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage1_blocks = nn.Sequential(
            ResidualBlock(32, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
        )
        self.pool_1 = nn.AdaptiveAvgPool1d(1)

        # Time–Frequency branch.
        self.tf_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(8, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.tf_fc = nn.Linear(16 * 4, 32)

        # -----------------------
        # Dynamic Anchor Setup for Frequency Prediction
        # -----------------------
        initial_anchor_values = torch.tensor(
            get_anchors(), dtype=torch.float32
        )  # shape: [B]
        # Create anchors for each cell by repeating the computed vector.
        self.anchors = nn.Parameter(
            initial_anchor_values.unsqueeze(0).repeat(S, 1)
        )  # shape: [S, B]
        self.freq_predictor = nn.Linear(128, S * B)
        self.band_predictor = nn.Linear(128, S * B)

        # Refinement branch.
        self.refinement_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.refine_fc = nn.Linear(32, S * B)

        # -----------------------
        # Stage-2: Confidence and Classification using the new classifier.
        # -----------------------
        # This classifier follows the narrowband architecture (adapted for 1+NUM_CLASSES outputs)
        self.classifier = WidebandClassifier(num_out=1 + NUM_CLASSES)

    def forward(self, x_time, x_freq):
        bsz = x_time.size(0)
        # -----------------------
        # Stage-1: Coarse Frequency Prediction
        # -----------------------
        h1 = self.first_conv(x_time)
        h1 = self.stage1_blocks(h1)
        h1 = self.pool_1(h1).squeeze(-1)  # [bsz, 96]

        spec = torch.sqrt(x_freq[:, 0, :] ** 2 + x_freq[:, 1, :] ** 2)
        spec = spec.unsqueeze(1).unsqueeze(-1)  # [bsz, 1, N_rfft, 1]
        tf_features = self.tf_branch(spec)
        tf_features = tf_features.view(bsz, -1)  # [bsz, 64]
        tf_features = self.tf_fc(tf_features)  # [bsz, 32]

        combined_features = torch.cat([h1, tf_features], dim=1)  # [bsz, 128]

        # Predict delta for frequency offset.
        raw_delta = self.freq_predictor(combined_features)  # shape: [bsz, S*B]
        raw_delta = raw_delta.view(bsz, S, B)
        delta_coarse = 0.5 * torch.tanh(raw_delta)
        # Use the learnable anchors (expanded over the batch) plus the delta.
        coarse_freq_pred = self.anchors.unsqueeze(0) + delta_coarse  # [bsz, S, B]

        refine_feat = self.refinement_branch(x_time)
        refine_feat = refine_feat.squeeze(-1)
        refine_delta = self.refine_fc(refine_feat)
        refine_delta = 0.1 * torch.tanh(refine_delta)
        refine_delta = refine_delta.view(bsz, S, B)

        # Final predicted normalized offset.
        freq_pred = coarse_freq_pred + refine_delta  # [bsz, S, B]

        bw_raw = self.band_predictor(combined_features)  # [bsz, S*B]
        MAXIMUM_BANDWIDTH = 3.0  # Maximum bandwidth in normalized units.
        bw_pred = torch.sigmoid(bw_raw.view(bsz, S, B)) * MAXIMUM_BANDWIDTH  # ∈[0,3]
        bw_pred_flat = bw_pred.view(bsz * S * B)

        cell_indices = torch.arange(
            S, device=freq_pred.device, dtype=freq_pred.dtype
        ).view(1, S, 1)
        freq_pred_raw = (cell_indices + freq_pred) * (SAMPLING_FREQUENCY / 2) / S
        freq_pred_flat = freq_pred_raw.view(bsz * S * B)

        # -----------------------
        # Stage-2: Downconversion and Classification
        # -----------------------
        x_rep = x_time.unsqueeze(1).unsqueeze(1).expand(-1, S, B, -1, -1)
        x_rep = x_rep.contiguous().view(bsz * S * B, 2, self.num_samples)

        x_filt = self._filter_raw(x_rep, freq_pred_flat, bw_pred_flat)
        x_base = self._downconvert_multiple(x_filt, freq_pred_flat)

        logits, embed = self.classifier(x_base, return_embedding=True)
        out_conf_class = logits
        embed = embed.view(bsz, S, B, -1)  # (bsz,S,B,96)
        out_conf_class = out_conf_class.view(bsz, S, B, 1 + NUM_CLASSES)

        final_out = torch.zeros(
            bsz,
            S,
            B,
            1 + 1 + 1 + NUM_CLASSES,
            dtype=out_conf_class.dtype,
            device=out_conf_class.device,
        )
        final_out[..., 0] = freq_pred  # offset
        final_out[..., 1] = out_conf_class[..., 0]  # confidence
        final_out[..., 2] = bw_pred  # bandwidth (norm.)
        final_out[..., 3:] = out_conf_class[..., 1:]  # classes
        final_out = final_out.view(bsz, S, B * (1 + 1 + 1 + NUM_CLASSES))
        if (not self.training) and MERGE_SIMILAR_PREDICTIONS:
            all_preds, all_embs = self._collect_raw_predictions(final_out, embed)
            merged = []
            merged_embs = []
            for p_list, e_list in zip(all_preds, all_embs):
                mp, me = self._merge_similar_predictions(
                    p_list, e_list, MERGE_SIMILAR_PREDICTIONS_THRESHOLD
                )
                merged.append(mp)
                merged_embs.append(me)
            emb_dim = embed.size(-1)
            final_out, embed = self._pack_merged_to_tensor(
                merged,
                merged_embs,
                final_out.device,
                final_out.dtype,
                embed.dtype,
                emb_dim,
            )
        return final_out, embed

    def _collect_raw_predictions(self, final_out, embed):
        """
        Turn raw model output (bsz×S×(B*(1+1+NUM_CLASSES))) into per-sample
        lists of (freq_Hz, class_idx, conf, bw) **and corresponding embeddings**.
        """
        bsz = final_out.size(0)
        raw = final_out.view(bsz, S, B, 1 + 1 + 1 + NUM_CLASSES)
        embed = embed.view(bsz, S, B, -1)
        pred_lists = []
        emb_lists = []
        for i in range(bsz):
            preds = []
            embs = []
            for si in range(S):
                for bi in range(B):
                    conf = raw[i, si, bi, 1].item()
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    off = raw[i, si, bi, 0].item()
                    freq = (si + off) * (SAMPLING_FREQUENCY / 2) / S
                    cls = int(raw[i, si, bi, 3:].argmax())
                    bw = raw[i, si, bi, 2].item()
                    preds.append((freq, cls, conf, bw))
                    embs.append(embed[i, si, bi].detach())
            pred_lists.append(preds)
            emb_lists.append(embs)
        return pred_lists, emb_lists

    def _pack_merged_to_tensor(
        self, merged_lists, emb_lists, device, dtype, emb_dtype, emb_dim
    ):
        """
        merged_lists: List of length bsz of predictions
        emb_lists   : matching embeddings
        returns: (pred_tensor, emb_tensor)
        """
        bsz = len(merged_lists)
        out = torch.zeros(
            bsz, S, B, 1 + 1 + 1 + NUM_CLASSES, device=device, dtype=dtype
        )
        out_emb = torch.zeros(bsz, S, B, emb_dim, device=device, dtype=emb_dtype)

        anchors = get_anchors()  # numpy array of length B
        for i, (preds, embs) in enumerate(zip(merged_lists, emb_lists)):
            for (freq, cls, conf, bw), emb in zip(preds, embs):
                freq_norm = freq / (SAMPLING_FREQUENCY / 2)
                cell = int(freq_norm * S)
                cell = min(cell, S - 1)
                off = freq_norm * S - cell
                off = float(np.clip(off, 0.0, 1.0))
                aidx = int(np.argmin(np.abs(anchors - off)))

                out[i, cell, aidx, 0] = off
                out[i, cell, aidx, 1] = conf
                out[i, cell, aidx, 2] = bw
                out[i, cell, aidx, 3 + cls] = 1.0
                out_emb[i, cell, aidx] = emb.to(device)

        return out, out_emb

    def _merge_similar_predictions(self, pred_list, emb_list, margin):
        """Return only the highest confidence prediction within ``margin``."""
        pairs = list(zip(pred_list, emb_list))
        # Sort by descending confidence so the first item in a cluster is kept
        pairs.sort(key=lambda x: x[0][2], reverse=True)

        kept_preds = []
        kept_embs = []
        for (freq, cls, conf, bw), emb in pairs:
            if all(abs(freq - kp[0]) > margin for kp in kept_preds):
                kept_preds.append((freq, cls, conf, bw))
                kept_embs.append(emb)
            # else: discard lower‑confidence prediction

        return kept_preds, kept_embs

    def _filter_raw(self, x_flat, freq_flat, bandwidth_flat):
        """
        x_flat          : [N, 2, T]
        freq_flat       : [N]          (centre freqs in Hz)
        bandwidth_flat  : [N]          (normalised bw in [0,1] wrt cell‑width)
        returns         : [N, 2, T]
        """
        N, _, T = x_flat.shape
        M = NUMTAPS
        device, dtype = x_flat.device, x_flat.dtype
        # ---------- low‑pass kernel ----------
        alpha = (M - 1) / 2.0
        n = torch.arange(M, device=device, dtype=dtype) - alpha  # [M]
        cutoff_norm = bandwidth_flat.clamp_min(1e-4) / S  # [N]
        x = cutoff_norm.unsqueeze(1) * n  # [N,M]

        # use PyTorch’s numerically‑stable sinc
        sinc = torch.sinc(x)  # sin(pi x)/(pi x)

        h_lp = cutoff_norm.unsqueeze(1) * sinc  # [N,M]
        h_lp = h_lp / h_lp.sum(dim=1, keepdim=True).clamp_min(1e-12)

        win = torch.kaiser_window(
            M, beta=8.6, periodic=False, dtype=dtype, device=device
        )
        h_lp = (h_lp * win) / (h_lp * win).sum(dim=1, keepdim=True)

        # ---------- shift to band‑pass ----------
        f0 = freq_flat.view(N, 1)  # [N,1]
        cos_factor = torch.cos(2 * pi * f0 * n / SAMPLING_FREQUENCY)  # [N,M]

        h_bp_all = h_lp * cos_factor  # [N,M]

        # duplicate for I & Q channels
        h_bp_all_expanded = h_bp_all.unsqueeze(1).repeat(1, 2, 1)  # [N,2,M]

        # ---------- apply FIR by unfolding ----------
        x_reshaped = x_flat.reshape(N * 2, 1, T)  # [N*2,1,T]
        weight = h_bp_all_expanded.reshape(N * 2, 1, M)  # [N*2,1,M]

        pad_left = M // 2
        pad_right = M - 1 - pad_left
        y = conv1d_batch(x_reshaped, weight, pad_left, pad_right)  # [N*2,1,T]
        y = y.reshape(N, 2, T)
        return y

    def _downconvert_multiple(self, x_flat, freq_flat):
        device = x_flat.device
        dtype = x_flat.dtype
        _, _, T = x_flat.shape
        t = (
            torch.arange(T, device=device, dtype=dtype).unsqueeze(0)
            / SAMPLING_FREQUENCY
        )
        freq_flat = freq_flat.unsqueeze(-1)
        angle = -2.0 * pi * freq_flat * t
        shift_real = torch.cos(angle)
        shift_imag = torch.sin(angle)
        x_real = x_flat[:, 0, :]
        x_imag = x_flat[:, 1, :]
        y_real = x_real * shift_real - x_imag * shift_imag
        y_imag = x_real * shift_imag + x_imag * shift_real
        x_base = torch.stack([y_real, y_imag], dim=1)
        return x_base


class WidebandYoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        """Reset accumulators used for detailed loss printing."""
        self._epoch_stats = {
            "iou": 0.0,
            "conf_obj": 0.0,
            "conf_noobj": 0.0,
            "cls": 0.0,
            "center": 0.0,
            "center_sep": 0.0,
        }
        self._samples = 0

    def _update_epoch_stats(
        self,
        batch_size,
        iou_loss,
        conf_loss_o,
        conf_loss_n,
        cls_loss,
        center_loss,
        center_sep_loss,
    ):
        self._epoch_stats["iou"] += iou_loss.item()
        self._epoch_stats["conf_obj"] += conf_loss_o.item()
        self._epoch_stats["conf_noobj"] += conf_loss_n.item()
        self._epoch_stats["cls"] += cls_loss.item()
        self._epoch_stats["center"] += center_loss.item()
        self._epoch_stats["center_sep"] += center_sep_loss.item()
        self._samples += batch_size

    def print_epoch_stats(self):
        """Print the averaged loss components for the epoch."""
        if self._samples == 0:
            return
        stats = {k: v / self._samples for k, v in self._epoch_stats.items()}
        print("\tDetailed Loss (avg per sample):")
        print(f"\t\tIoULoss: {stats['iou']:.4f}")
        print(f"\t\tConfLossObj: {stats['conf_obj']:.4f}")
        print(f"\t\tConfLossNoObj: {stats['conf_noobj']:.4f}")
        print(f"\t\tClsLoss: {stats['cls']:.4f}")
        print(f"\t\tCenterLoss: {stats['center']:.4f}")
        print(f"\t\tCenterSepLoss: {stats['center_sep']:.4f}")

    def _pairwise_dist(self, x):
        prod = x @ x.t()
        sq = torch.diagonal(prod)
        dist = sq.unsqueeze(1) - 2 * prod + sq.unsqueeze(0)
        dist = torch.clamp(dist, min=0.0)
        return dist

    def forward(self, pred, target, embed, centers):
        batch_size = pred.shape[0]
        pred = pred.view(batch_size, pred.shape[1], B, 1 + 1 + 1 + NUM_CLASSES)
        target = target.view_as(pred)

        x_pred = pred[..., 0]  # frequency offset
        conf_pred = pred[..., 1]
        bw_pred = pred[..., 2]  # bandwidth
        cls_pred = pred[..., 3:]

        x_tgt = target[..., 0]
        conf_tgt = target[..., 1]
        bw_tgt = target[..., 2]
        cls_tgt = target[..., 3:]

        obj_mask = (conf_tgt > 0).float()
        noobj_mask = 1.0 - obj_mask

        # ----- IoU between predicted and target frequency regions -----
        pred_low = x_pred - (bw_pred / 2.0)
        pred_high = x_pred + (bw_pred / 2.0)
        tgt_low = x_tgt - (bw_tgt / 2.0)
        tgt_high = x_tgt + (bw_tgt / 2.0)

        inter_low = torch.maximum(pred_low, tgt_low)
        inter_high = torch.minimum(pred_high, tgt_high)
        intersection = (inter_high - inter_low).clamp(min=0.0)

        union_low = torch.minimum(pred_low, tgt_low)
        union_high = torch.maximum(pred_high, tgt_high)
        union = (union_high - union_low).clamp(min=1e-6)

        iou_1d = intersection / union
        iou_loss = IOU_LOSS * torch.sum(obj_mask * (1.0 - iou_1d))

        conf_loss_o = torch.sum(obj_mask * (conf_pred - iou_1d) ** 2)
        conf_loss_n = LAMBDA_NOOBJ * torch.sum(noobj_mask * (conf_pred**2))

        with torch.no_grad():
            tgt_idx = cls_tgt.argmax(dim=-1)
        if obj_mask.sum() > 0:
            ce = F.cross_entropy(
                cls_pred[obj_mask.bool()],
                tgt_idx[obj_mask.bool()],
                reduction="sum",
            )
        else:
            ce = torch.tensor(0.0, device=cls_pred.device)
        cls_loss = LAMBDA_CLASS * ce

        # ---------- centre-loss on *GT* boxes --------------------------
        with torch.no_grad():
            gt_idx = cls_tgt.argmax(dim=-1)  # [B,S,B]
        obj_mask_flat = obj_mask.bool().view(-1)
        emb_flat = embed.view(embed.size(0), embed.size(1), B, 96)
        emb_flat = emb_flat.view(-1, 96)[obj_mask_flat]  # [N_pos,D]
        label_flat = gt_idx.view(-1)[obj_mask_flat]  # [N_pos]

        center_sel = centers[label_flat]  # [N_pos,D]
        center_loss = ((emb_flat - center_sel) ** 2).sum(1).mean() * LAMBDA_CENTER

        # ---------- maximise separation between class centres -------------
        with torch.no_grad():
            num_c = centers.size(0)
        dists_cent = self._pairwise_dist(centers)
        mask = torch.ones_like(dists_cent) - torch.eye(num_c, device=dists_cent.device)
        sep_mean = (dists_cent * mask).sum() / (num_c * (num_c - 1))
        # Take the logarithm of the mean distance to avoid getting too large values.
        sep_mean = torch.log(sep_mean + 1e-9)  # avoid log(0)
        center_sep_loss = -LAMBDA_CENTER_SEP * sep_mean

        total_loss = (
            iou_loss
            + conf_loss_o
            + conf_loss_n
            + cls_loss
            + center_loss
            + center_sep_loss
        )  # keep same scale

        if DETAILED_LOSS_PRINT:
            self._update_epoch_stats(
                batch_size,
                iou_loss,
                conf_loss_o,
                conf_loss_n,
                cls_loss,
                center_loss,
                center_sep_loss,
            )

        return total_loss / batch_size
