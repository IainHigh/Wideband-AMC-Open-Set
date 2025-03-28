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
    IOU_SCALING_FACTOR
)

###############################################################################
# Helper to build a lowpass filter kernel in PyTorch
###############################################################################
def build_lowpass_filter(cutoff_hz, fs, num_taps, window="hamming"):
    """
    Build a real, time-domain lowpass FIR filter via windowed-sinc method in PyTorch.
    cutoff_hz:  The passband edge (in Hz).
    fs:         Sampling rate (in Hz).
    num_taps:   Number of FIR taps (must be odd for no phase shift, typically).
    window:     'hamming', 'hanning', or 'blackman' (as an example).
    
    Returns:
      lp: A 1D PyTorch tensor of shape [num_taps], representing the filter kernel.
    """
    # Center index
    M = num_taps
    n = torch.arange(M, dtype=torch.float32)
    alpha = (M - 1) / 2.0
    # Ideal sinc
    cutoff_norm = float(cutoff_hz) / (fs / 2.0)  # normalized freq in [0..1], 1 => Nyquist
    # handle corner cases
    eps = 1e-9
    def sinc(x):
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(math.pi*x)/(math.pi*x))

    # time index for "normalized" frequency
    h = cutoff_norm * sinc(cutoff_norm*(n - alpha))
    
    # window
    if window == "hamming":
        win = 0.54 - 0.46 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "hanning":
        win = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (n / (M - 1)))
    elif window == "blackman":
        win = 0.42 - 0.5*torch.cos(2*math.pi*n/(M-1)) + 0.08*torch.cos(4*math.pi*n/(M-1))
    else:
        # no window
        win = torch.ones(M, dtype=torch.float32)

    h = h * win
    # Normalize so sum of taps = 1.0
    h = h / torch.sum(h)
    return h  # shape [num_taps]

###############################################################################
# Residual block
###############################################################################
class ResidualBlock(nn.Module):
    """
    Residual block with 3 branches, like your previous pipeline approach.
    branch1 => stride=2 with kernel_size=1
    branch2 => stride=2 with kernel_size=3
    branch3 => stride=2 with kernel_size=1
    Then we concat, plus a skip connection that also has stride=2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # branch 1
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # branch 2
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # branch 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # residual (skip)
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2),
            nn.BatchNorm1d(out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        res = self.residual(x)  # shape => [batch, out_ch, length/2]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)  # => [batch, 32+32+32=96, length/2]
        out = F.relu(concat + res)
        return out
    
################################################################################
# WidebandYoloModel
################################################################################
class WidebandYoloModel(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        
        # Stage-1: Predict relative frequency (in [0,1])
        self.first_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.first_block = ResidualBlock(32, 96)
        self.pool_1 = nn.AdaptiveAvgPool1d(1)
        self.freq_predictor = nn.Linear(96, S * B)

        # Fixed Lowpass Filter
        lp_taps = build_lowpass_filter(
            cutoff_hz=BAND_MARGIN,
            fs=SAMPLING_FREQUENCY,
            num_taps=NUMTAPS,
            window="hamming"
        )
        self.conv_lowpass = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=NUMTAPS,
            groups=2,
            bias=False,
            padding="same"
        )
        with torch.no_grad():
            self.conv_lowpass.weight[0, 0, :] = lp_taps
            self.conv_lowpass.weight[1, 0, :] = lp_taps
        for param in self.conv_lowpass.parameters():
            param.requires_grad = False

        # Stage-2: Conf/Class
        self.second_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.second_block = ResidualBlock(32, 96)
        self.pool_2 = nn.AdaptiveAvgPool1d(1)
        self.conf_class_predictor = nn.Linear(96, 1 + NUM_CLASSES)

    def forward(self, x):
        bsz = x.size(0)
        h1 = self.first_conv(x)
        h1 = self.first_block(h1)
        h1 = self.pool_1(h1).squeeze(-1)
        
        freq_pred_unnorm = self.freq_predictor(h1)
        freq_pred = torch.sigmoid(freq_pred_unnorm)
        freq_pred = freq_pred.view(bsz, S, B)
        
        # Debug print of predicted relative frequency for first sample
        if bsz > 0:
            print("Debug: freq_pred (relative) for first sample:", freq_pred[0].detach().cpu().numpy())
        
        # Instead of computing (cell_index + offset)/S, we now use the predicted relative frequency directly.
        abs_freq_pred = freq_pred  
        if bsz > 0:
            abs_freq_pred_first = abs_freq_pred[0].detach().cpu().numpy()
            abs_freq_pred_hz_first = abs_freq_pred_first * SAMPLING_FREQUENCY
            print("Debug: abs_freq_pred (relative) for first sample:", abs_freq_pred_first)
            print("Debug: abs_freq_pred in Hz for first sample:", abs_freq_pred_hz_first)
        
        abs_freq_pred_flat = abs_freq_pred.view(bsz * S * B)
        
        # Downconvert
        x_rep = x.unsqueeze(1).unsqueeze(1)
        x_rep = x_rep.expand(-1, S, B, -1, -1)
        x_rep = x_rep.contiguous().view(bsz*S*B, 2, self.num_samples)
        x_base = self._downconvert_multiple(x_rep, abs_freq_pred_flat)
        
        # Apply fixed lowpass filter (if needed)
        x_filt = x_base
        
        # Stage-2: conf/class
        h2 = self.second_conv(x_filt)
        h2 = self.second_block(h2)
        h2 = self.pool_2(h2).squeeze(-1)
        out_conf_class = self.conf_class_predictor(h2)
        out_conf_class = out_conf_class.view(bsz, S, B, 1 + NUM_CLASSES)

        final_out = torch.zeros(
            bsz, S, B, (1 + 1 + NUM_CLASSES),
            dtype=out_conf_class.dtype,
            device=out_conf_class.device
        )
        # Now the first channel is the predicted relative frequency.
        final_out[..., 0] = freq_pred
        final_out[..., 1:] = out_conf_class
        final_out = final_out.view(bsz, S, B*(1 + 1 + NUM_CLASSES))
        return final_out

    def _downconvert_multiple(self, x_flat, freq_flat):
        device = x_flat.device
        dtype  = x_flat.dtype
        _, _, T = x_flat.shape
        
        if freq_flat.numel() > 0:
            print("Debug: freq_flat for downconversion (first element):", freq_flat[0].item())
        
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)
        freq_flat = freq_flat.unsqueeze(-1)
        # Multiply the normalized frequency by SAMPLING_FREQUENCY to get the true frequency in Hz.
        angle = -2.0 * math.pi * (freq_flat * SAMPLING_FREQUENCY) * t
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

    def forward(self, pred, target):
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

        coord_loss = LAMBDA_COORD * torch.sum(obj_mask*(x_pred - x_tgt)**2)
        
        iou_1d = 1.0 - (torch.abs(x_pred - x_tgt) / IOU_SCALING_FACTOR)
        iou_1d = torch.clamp(iou_1d, min=0.0, max=1.0)

        conf_loss_obj = torch.sum(obj_mask*(conf_pred - iou_1d)**2)
        conf_loss_noobj = LAMBDA_NOOBJ * torch.sum(noobj_mask*(conf_pred**2))

        class_diff = (class_pred - class_tgt)**2
        class_loss = LAMBDA_CLASS*torch.sum(obj_mask.unsqueeze(-1)*class_diff)

        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss / batch_size