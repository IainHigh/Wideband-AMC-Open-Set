import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from config_wideband_yolo import (
    S,
    B,
    NUM_CLASSES,
    LAMBDA_COORD,
    LAMBDA_NOOBJ,
    LAMBDA_CLASS,
    BAND_MARGIN,
    NUMTAPS,
    SAMPLING_FREQUENCY,
    get_anchors,
    EPOCHS,
    USE_SIMILARITY_MATRIX,
    MODULATION_CLASSES, 
    SIMILARITY_DICT,
)


###############################################################################
# Helper to build a lowpass filter kernel in PyTorch
###############################################################################
def build_lowpass_filter(cutoff_hz, fs, num_taps, window="hamming"):
    M = num_taps
    n = torch.arange(M, dtype=torch.float32)
    alpha = (M - 1) / 2.0
    cutoff_norm = float(cutoff_hz) / (fs / 2.0)
    eps = 1e-9

    def sinc(x):
        return torch.where(
            torch.abs(x) < eps,
            torch.ones_like(x),
            torch.sin(math.pi * x) / (math.pi * x),
        )

    h = cutoff_norm * sinc(cutoff_norm * (n - alpha))
    if window == "hamming":
        win = 0.54 - 0.46 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "hanning":
        win = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "blackman":
        win = (
            0.42
            - 0.5 * torch.cos(2 * math.pi * n / (M - 1))
            + 0.08 * torch.cos(4 * math.pi * n / (M - 1))
        )
    elif window == "kaiser":
        win = torch.kaiser_window(M, beta=8.6, periodic=False)
    else:
        win = torch.ones(M, dtype=torch.float32)
    h = h * win
    h = h / torch.sum(h)
    return h


def conv1d_batch(x, weight, pad_left, pad_right):
    x_padded = F.pad(x, (pad_left, pad_right))
    x_unf = x_padded.unfold(dimension=2, size=weight.shape[-1], step=1)
    weight = weight.unsqueeze(2)
    y = (x_unf * weight).sum(dim=-1)
    return y


###############################################################################
# Residual block (used in Stage-1)
###############################################################################
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


###############################################################################
# New WidebandClassifier (Stage-2: Confidence and Classification)
###############################################################################
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
        # Fully Connected Layer producing (1 + NUM_CLASSES) outputs.
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

    def forward(self, x):
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


###############################################################################
# WidebandYoloModel (Complete YOLO with dynamic anchors and new classifier)
###############################################################################
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
        freq_pred = coarse_freq_pred + refine_delta # [bsz, S, B]

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

        x_filt = self._filter_raw(x_rep, freq_pred_flat)
        x_base = self._downconvert_multiple(x_filt, freq_pred_flat)

        out_conf_class = self.classifier(x_base)  # [bsz*S*B, 1+NUM_CLASSES]
        out_conf_class = out_conf_class.view(bsz, S, B, 1 + NUM_CLASSES)

        final_out = torch.zeros(
            bsz,
            S,
            B,
            (1 + 1 + NUM_CLASSES),
            dtype=out_conf_class.dtype,
            device=out_conf_class.device,
        )
        final_out[..., 0] = freq_pred  # normalized frequency offset
        final_out[..., 1:] = out_conf_class
        final_out = final_out.view(bsz, S, B * (1 + 1 + NUM_CLASSES))
        return final_out

    def _filter_raw(self, x_flat, freq_flat):
        N, _, T = x_flat.shape
        M = NUMTAPS
        alpha = (M - 1) / 2.0
        n = torch.arange(M, device=x_flat.device, dtype=x_flat.dtype) - alpha
        h_lp = build_lowpass_filter(
            cutoff_hz=BAND_MARGIN,
            fs=SAMPLING_FREQUENCY,
            num_taps=NUMTAPS,
            window="kaiser",
        )
        h_lp = h_lp.to(x_flat.device)
        h_lp = h_lp.unsqueeze(0)
        f0 = freq_flat.view(N, 1)
        cos_factor = torch.cos(2 * math.pi * f0 * n / SAMPLING_FREQUENCY)
        h_bp_all = h_lp * cos_factor
        h_bp_all_expanded = h_bp_all.unsqueeze(1).repeat(1, 2, 1)
        x_reshaped = x_flat.reshape(N * 2, 1, T)
        weight = h_bp_all_expanded.reshape(N * 2, 1, M)
        pad_left = M // 2
        pad_right = M - 1 - pad_left
        y = conv1d_batch(x_reshaped, weight, pad_left, pad_right)
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
        angle = -2.0 * math.pi * freq_flat * t
        shift_real = torch.cos(angle)
        shift_imag = torch.sin(angle)
        x_real = x_flat[:, 0, :]
        x_imag = x_flat[:, 1, :]
        y_real = x_real * shift_real - x_imag * shift_imag
        y_imag = x_real * shift_imag + x_imag * shift_real
        x_base = torch.stack([y_real, y_imag], dim=1)
        return x_base


###############################################################################
# WidebandYoloLoss remains unchanged.
###############################################################################
class WidebandYoloLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, current_epoch=None):
        batch_size = pred.shape[0]
        pred = pred.view(batch_size, pred.shape[1], B, (1 + 1 + NUM_CLASSES))
        x_pred = pred[..., 0]
        conf_pred = pred[..., 1]
        class_pred = pred[..., 2:]
        x_tgt = target[..., 0]
        conf_tgt = target[..., 1]
        class_tgt = target[..., 2:]
        obj_mask = (conf_tgt > 0).float()
        noobj_mask = 1.0 - obj_mask      
        coord_loss = LAMBDA_COORD * torch.sum(obj_mask * (x_pred - x_tgt) ** 2)
        iou_1d = 1.0 - torch.abs(x_pred - x_tgt)
        conf_loss_obj = torch.sum(obj_mask * (conf_pred - iou_1d) ** 2)
        conf_loss_noobj = LAMBDA_NOOBJ * torch.sum(noobj_mask * (conf_pred**2))
        
        if USE_SIMILARITY_MATRIX:
            sim_matrix = torch.ones((NUM_CLASSES, NUM_CLASSES), dtype=class_pred.dtype, device=class_pred.device)
            for i, mod1 in enumerate(MODULATION_CLASSES):
                for j, mod2 in enumerate(MODULATION_CLASSES):
                    if mod1 == mod2:
                        sim_matrix[i, j] = 1.0
                    elif (mod1, mod2) in SIMILARITY_DICT:
                        sim_matrix[i, j] = SIMILARITY_DICT[(mod1, mod2)]
                    elif (mod2, mod1) in SIMILARITY_DICT:
                        sim_matrix[i, j] = SIMILARITY_DICT[(mod2, mod1)]
                    else:
                        sim_matrix[i, j] = 1.0  # default value if not defined
            true_class_idx = torch.argmax(class_tgt, dim=-1)
            weights = sim_matrix[true_class_idx]
            class_loss = LAMBDA_CLASS * torch.sum(obj_mask.unsqueeze(-1) * weights * (class_pred - class_tgt) ** 2)
        else:
            class_loss = LAMBDA_CLASS * torch.sum(obj_mask.unsqueeze(-1) * (class_pred - class_tgt) ** 2)
            
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss / batch_size
