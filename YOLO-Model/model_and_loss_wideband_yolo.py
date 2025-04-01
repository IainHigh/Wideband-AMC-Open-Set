#############################################
# model_and_loss_wideband_yolo.py
#############################################
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
)

###############################################################################
# Helper to build a lowpass filter kernel in PyTorch
###############################################################################
def build_lowpass_filter(cutoff_hz, fs, num_taps, window="hamming"):
    """
    Build a real, time-domain lowpass FIR filter via the windowed-sinc method.
    
    cutoff_hz: The passband edge in Hz.
    fs: Sampling frequency in Hz.
    num_taps: Number of taps (should be odd for zero phase shift).
    window: One of "hamming", "hanning", "blackman", or "kaiser".
    
    Returns:
      A 1D PyTorch tensor of shape [num_taps] representing the filter kernel.
    """
    M = num_taps
    n = torch.arange(M, dtype=torch.float32)
    alpha = (M - 1) / 2.0
    cutoff_norm = float(cutoff_hz) / (fs / 2.0)  # normalized cutoff in [0,1]
    eps = 1e-9
    def sinc(x):
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    
    h = cutoff_norm * sinc(cutoff_norm * (n - alpha))
    
    if window == "hamming":
        win = 0.54 - 0.46 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "hanning":
        win = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "blackman":
        win = 0.42 - 0.5 * torch.cos(2 * math.pi * n / (M - 1)) + 0.08 * torch.cos(4 * math.pi * n / (M - 1))
    elif window == "kaiser":
        win = torch.kaiser_window(M, beta=8.6, periodic=False)
    else:
        win = torch.ones(M, dtype=torch.float32)
    
    h = h * win
    h = h / torch.sum(h)
    return h  # shape [num_taps]


def conv1d_batch(x, weight, pad_left, pad_right):
    """
    Performs a batched 1D convolution where each sample in the batch uses its own kernel.
    
    x: Tensor of shape [N, 1, T]
    weight: Tensor of shape [N, 1, M]
    pad_left, pad_right: integers for padding
    Returns: Tensor of shape [N, 1, T] containing the convolved outputs.
    """
    x_padded = F.pad(x, (pad_left, pad_right))
    x_unf = x_padded.unfold(dimension=2, size=weight.shape[-1], step=1)  # [N, 1, T, M]
    weight = weight.unsqueeze(2)  # [N, 1, 1, M]
    y = (x_unf * weight).sum(dim=-1)  # [N, 1, T]
    return y


###############################################################################
# Residual block
###############################################################################
class ResidualBlock(nn.Module):
    """
    A residual block with 3 branches. Each branch downsamples with stride=2.
    """
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
        concat = torch.cat([b1, b2, b3], dim=1)  # [batch, 96, length/2]
        out = F.relu(concat + res)
        return out

    
################################################################################
# WidebandYoloModel
################################################################################
class WidebandYoloModel(nn.Module):
    """
    Two-stage YOLO approach:
      Stage-1: Predict S*B frequency offsets in [0,1] using a dual-branch network.
               One branch processes time-domain features; the new branch uses the precomputed
               frequency–domain representation for enhanced frequency localization.
               A refinement branch further corrects the coarse predictions.
      Stage-2: For each predicted box, downconvert the input signal and extract features
               for confidence and classification.
               
    The final YOLO output is [batch, S, B*(1+1+NUM_CLASSES)].
    """
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
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
        )
        self.pool_1 = nn.AdaptiveAvgPool1d(1)
        
        # New Time–Frequency branch: now takes precomputed frequency data.
        # The dataset returns x_freq of shape (batch, 2, N_rfft). We compute the magnitude.
        self.tf_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 1), stride=1, padding=(1,0)),  # change kernel size to (3,1)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # pool only along frequency axis
            nn.Conv2d(8, 16, kernel_size=(3, 1), stride=1, padding=(1,0)),  # change kernel size to (3,1)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Project the TF branch output to a feature vector of size 32.
        self.tf_fc = nn.Linear(16, 32)
        
        # Combine time-domain (96) and TF branch (32) features = 128.
        self.freq_predictor = nn.Linear(128, S * B)
        
        # Refinement branch for frequency correction.
        self.refinement_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.refine_fc = nn.Linear(32, S * B)
        
        # -----------------------
        # Stage-2: Confidence and Classification (unchanged)
        # -----------------------
        self.second_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2_blocks = nn.Sequential(
            ResidualBlock(32, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
        )
        self.pool_2 = nn.AdaptiveAvgPool1d(1)
        self.conf_class_predictor = nn.Linear(96, 1 + NUM_CLASSES)

    def forward(self, x_time, x_freq):
        """
        x_time: [batch, 2, num_samples] (time-domain IQ)
        x_freq: [batch, 2, N_rfft] (frequency-domain representation, e.g. from np.fft.rfft)
        
        Returns final YOLO output: [batch, S, B*(1+1+NUM_CLASSES)]
        """
        bsz = x_time.size(0)
        # -----------------------
        # Stage-1: Coarse Frequency Prediction
        # -----------------------
        # Time-domain branch.
        h1 = self.first_conv(x_time)          # [bsz, 32, T1]
        h1 = self.stage1_blocks(h1)             # [bsz, 96, T1']
        h1 = self.pool_1(h1).squeeze(-1)        # [bsz, 96]
        
        # Time–Frequency branch: use provided frequency data.
        # Compute the magnitude spectrum.
        # x_freq is [bsz, 2, N_rfft]; compute magnitude and reshape to (bsz, 1, N_rfft, 1)
        spec = torch.sqrt(x_freq[:, 0, :]**2 + x_freq[:, 1, :]**2)
        spec = spec.unsqueeze(1).unsqueeze(-1)   # [bsz, 1, N_rfft, 1]
        tf_features = self.tf_branch(spec)        # [bsz, 16, 1, 1]
        tf_features = tf_features.view(bsz, -1)    # [bsz, 16]
        tf_features = self.tf_fc(tf_features)       # [bsz, 32]
        
        # Combine features from both branches.
        combined_features = torch.cat([h1, tf_features], dim=1)  # [bsz, 128]
        coarse_freq_pred_unnorm = self.freq_predictor(combined_features)  # [bsz, S*B]
        coarse_freq_pred = torch.sigmoid(coarse_freq_pred_unnorm)         # in [0,1]
        coarse_freq_pred = coarse_freq_pred.view(bsz, S, B)                # [bsz, S, B]
        
        # Refinement branch: compute correction delta from raw IQ.
        refine_feat = self.refinement_branch(x_time)  # [bsz, 32, 1]
        refine_feat = refine_feat.squeeze(-1)         # [bsz, 32]
        delta = self.refine_fc(refine_feat)             # [bsz, S*B]
        delta = 0.1 * torch.tanh(delta)                 # constrain correction to [-0.1, 0.1]
        delta = delta.view(bsz, S, B)
        
        # Final frequency prediction.
        freq_pred = torch.clamp(coarse_freq_pred + delta, 0.0, 1.0)  # [bsz, S, B]
        
        # Convert normalized offsets to raw frequencies for downconversion.
        cell_indices = torch.arange(S, device=freq_pred.device, dtype=freq_pred.dtype).view(1, S, 1)
        freq_pred_raw = (cell_indices + freq_pred) * (SAMPLING_FREQUENCY / 2) / S  # [bsz, S, B]
        freq_pred_flat = freq_pred_raw.view(bsz * S * B)
        
        # -----------------------
        # Stage-2: Downconversion and Classification (uses time-domain signal only)
        # -----------------------
        # Replicate x_time for each bounding box.
        x_rep = x_time.unsqueeze(1).unsqueeze(1).expand(-1, S, B, -1, -1)
        x_rep = x_rep.contiguous().view(bsz * S * B, 2, self.num_samples)
        
        # Apply the fixed lowpass filter.
        x_filt = self._filter_raw(x_rep, freq_pred_flat)
        # Downconvert using the raw frequency predictions.
        x_base = self._downconvert_multiple(x_filt, freq_pred_flat)
        
        # Extract features for confidence and classification.
        h2 = self.second_conv(x_base)              # [bsz*S*B, 32, T2]
        h2 = self.stage2_blocks(h2)                 # [bsz*S*B, 96, T2']
        h2 = self.pool_2(h2).squeeze(-1)            # [bsz*S*B, 96]
        out_conf_class = self.conf_class_predictor(h2)  # [bsz*S*B, 1+NUM_CLASSES]
        out_conf_class = out_conf_class.view(bsz, S, B, 1 + NUM_CLASSES)
        
        # Merge frequency prediction with confidence and class outputs.
        final_out = torch.zeros(bsz, S, B, (1 + 1 + NUM_CLASSES),
                                  dtype=out_conf_class.dtype,
                                  device=out_conf_class.device)
        final_out[..., 0] = freq_pred   # normalized frequency offset
        final_out[..., 1:] = out_conf_class
        final_out = final_out.view(bsz, S, B * (1 + 1 + NUM_CLASSES))
        return final_out

    # _filter_raw and _downconvert_multiple remain unchanged.
    def _filter_raw(self, x_flat, freq_flat):
        N, _, T = x_flat.shape
        M = NUMTAPS
        alpha = (M - 1) / 2.0
        n = torch.arange(M, device=x_flat.device, dtype=x_flat.dtype) - alpha
        h_lp = build_lowpass_filter(cutoff_hz=BAND_MARGIN, fs=SAMPLING_FREQUENCY, num_taps=NUMTAPS, window="kaiser")
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
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(0) / SAMPLING_FREQUENCY
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

    
################################################################################
# WidebandYoloLoss
################################################################################
class WidebandYoloLoss(nn.Module):
    """
    Same YOLO loss as before, adjusted for the new output dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred: [batch, S, B*(1 + 1 + NUM_CLASSES)]
        target: [batch, S, B, (1 + 1 + NUM_CLASSES)]
        """
        batch_size = pred.shape[0]
        pred = pred.view(batch_size, pred.shape[1], B, (1 + 1 + NUM_CLASSES))

        x_pred     = pred[..., 0]
        conf_pred  = pred[..., 1]
        class_pred = pred[..., 2:]

        x_tgt      = target[..., 0]
        conf_tgt   = target[..., 1]
        class_tgt  = target[..., 2:]

        obj_mask   = (conf_tgt > 0).float()
        noobj_mask = 1.0 - obj_mask

        coord_loss = LAMBDA_COORD * torch.sum(obj_mask * (x_pred - x_tgt)**2)
        iou_1d = 1.0 - torch.abs(x_pred - x_tgt)
        iou_1d = torch.clamp(iou_1d, min=0.0, max=1.0)
        conf_loss_obj = torch.sum(obj_mask * (conf_pred - iou_1d)**2)
        conf_loss_noobj = LAMBDA_NOOBJ * torch.sum(noobj_mask * (conf_pred**2))
        class_diff = (class_pred - class_tgt)**2
        class_loss = LAMBDA_CLASS * torch.sum(obj_mask.unsqueeze(-1) * class_diff)

        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss / batch_size
