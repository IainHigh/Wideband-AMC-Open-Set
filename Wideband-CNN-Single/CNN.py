import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.signal import firwin


class ModulationClassifier(nn.Module):
    """
    Full 'Deep Learning-Based Automatic Modulation Classification' architecture,
    with parallel conv blocks, etc.

    Then we add a final policy head of size (2 + 1 + num_classes):
      - 2 policy logits: [split vs. stop]
      - 1 freq fraction: in [0,1] after sigmoid
      - num_classes: mod classification logit distribution

    We also incorporate bandpass filtering + downconversion each time we do recursion
    on subbands. This is done inside `stochastic_recursive_inference`.
    """

    def __init__(self, num_classes, max_depth=5, fs=20e6):
        """
        num_classes: how many modulation classes
        max_depth: optional limit to recursion depth
        fs: default sampling rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.fs = (
            fs  # default sampling rate if you want it. Or pass it in recursion calls
        )

        # -- Original CNN Architecture from your code:
        #  ConvBlock1 => in_channels=2 => out_channels=32 => kernel=8 => batchnorm => relu => maxpool
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Then 4 repeated blocks (Block2), each has parallel branches of size 32 and a 96-dim residual
        self.block2_layers = nn.ModuleList(
            [
                self._create_block2(
                    input_channels=(32 if i == 0 else 96), output_channels=96
                )
                for i in range(4)
            ]
        )

        # global average pool => (1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # final linear => produce 2 + 1 + num_classes
        # 2 => policy logits, 1 => freq fraction, rest => mod logits
        self.fc_outdim = 2 + 1 + num_classes
        self.fc = nn.Linear(96, self.fc_outdim)

    def _create_block2(self, input_channels, output_channels):
        """
        Parallel branches each produce 32 channels => total 96 => residual => 96
        """
        return nn.ModuleDict(
            {
                "branch1": nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=1, stride=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch2": nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "branch3": nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=1, stride=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                ),
                "residual": nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=2),
                    nn.BatchNorm1d(output_channels),
                ),
            }
        )

    def forward(self, x):
        """
        x: shape [batch_size, 2, length]
        returns: shape [batch_size, 2 + 1 + num_classes]
        """
        # conv block 1
        x = self.conv_block1(x)

        # repeated block2 layers
        for block in self.block2_layers:
            residual = block["residual"](x)
            branch_outputs = [
                block["branch1"](x),
                block["branch2"](x),
                block["branch3"](x),
            ]
            cat_out = torch.cat(branch_outputs, dim=1)  # shape => [batch_size, 96, L/2]
            x = F.relu(cat_out + residual)

        # global avg pool => shape => [batch_size, 96, 1]
        x = self.global_avg_pool(x)
        # flatten => [batch_size, 96]
        x = x.squeeze(-1)

        # final fc => [batch_size, 2 + 1 + num_classes]
        out = self.fc(x)
        return out

    # -------------------------------------------------------
    # RECURSIVE INFERENCE CODE
    # -------------------------------------------------------
    def stochastic_recursive_inference(
        self, wideband_x, freq_start, freq_end, log_probs_accum, depth=0
    ):
        """
        - wideband_x: shape => [1,2,L], the signal
        - freq_start, freq_end => the freq range in Hz
        - log_probs_accum => list to store log-probs for policy gradient
        - depth => current recursion depth

        returns: a list of (center_freq, mod_class)
        """

        # 1) If we hit max depth, we will produce a final signal and stop
        if depth >= self.max_depth:
            # produce final => skip the policy
            # We'll create a pseudo-forward with "action=1"
            out = self._forward_subband(wideband_x)
            policy_logits = out[:, :2]
            freq_raw = out[:, 2]
            mod_logits = out[:, 3:]

            # skip sampling the policy => forcibly "stop"
            fraction = torch.sigmoid(freq_raw)
            center_freq = freq_start + fraction.item() * (freq_end - freq_start)

            mod_dist = F.softmax(mod_logits, dim=1)
            mod_action = torch.distributions.Categorical(mod_dist).sample()
            return [(center_freq, mod_action.item())]

        # 2) If depth < max => we do the standard approach
        out = self._forward_subband(wideband_x, freq_start, freq_end)
        # parse out => shape => [1, 2 + 1 + num_classes]
        policy_logits = out[:, :2]  # shape => [1,2]
        freq_raw = out[:, 2]  # shape => [1]
        mod_logits = out[:, 3:]  # shape => [1, num_classes]

        # we interpret policy => 0 => "split", 1 => "stop"
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        action = policy_dist.sample()  # shape => [1], 0 or 1
        log_p = policy_dist.log_prob(action)
        log_probs_accum.append(log_p)

        if action.item() == 0:
            # split => subdiv
            mid = 0.5 * (freq_start + freq_end)
            left_preds = self.stochastic_recursive_inference(
                wideband_x, freq_start, mid, log_probs_accum, depth + 1
            )
            right_preds = self.stochastic_recursive_inference(
                wideband_x, mid, freq_end, log_probs_accum, depth + 1
            )
            return left_preds + right_preds
        else:
            # "stop => produce final freq + mod"
            fraction = torch.sigmoid(freq_raw)  # in [0,1]
            center_freq = freq_start + fraction.item() * (freq_end - freq_start)

            mod_dist = F.softmax(mod_logits, dim=1)
            mod_action = torch.distributions.Categorical(mod_dist).sample()
            return [(center_freq, mod_action.item())]

    def _forward_subband(self, wideband_x, freq_start=None, freq_end=None):
        """
        Filter the wideband_x to the sub-band [freq_start, freq_end],
        downconvert to baseband, then pass through the CNN forward.

        wideband_x: shape [1,2,L]
        freq_start, freq_end => freq range in Hz
        If freq_start/freq_end is None => means "no subband filtering". (like max_depth case)

        returns => shape [1, 2 + 1 + num_classes]
        """
        x_np = wideband_x.squeeze(0).cpu().numpy()  # shape => [2,L], float
        if freq_start is None or freq_end is None or freq_end <= freq_start:
            # skip filtering => pass the raw
            pass
        else:
            # bandpass + downconvert
            # x_np => shape => [2,L]
            I = x_np[0, :]
            Q = x_np[1, :]
            complex_w = I + 1j * Q

            # bandpass
            fs = self.fs
            lowcut = freq_start
            highcut = freq_end

            # we use the method below => _fir_bandpass_complex
            filtered = self._fir_bandpass_complex(complex_w, fs, lowcut, highcut)
            # downconvert => center freq => (lowcut+highcut)/2
            center_fc = 0.5 * (lowcut + highcut)
            baseband = self._downconvert_to_baseband(filtered, fs, center_fc)

            # build new array => shape [2, Lnew]
            I2 = baseband.real
            Q2 = baseband.imag
            x_np = np.stack([I2, Q2], axis=0)

        # to Torch
        sub_x = torch.tensor(
            x_np, dtype=torch.float32, device=wideband_x.device
        ).unsqueeze(0)
        # shape => [1,2,L]
        out = self.forward(sub_x)
        return out

    # -------------------------------------------------------
    # Utility for bandpass + downconvert in each recursion
    # -------------------------------------------------------
    def _fir_bandpass_complex(self, x, fs, lowcut, highcut, numtaps=101, beta=8.6):
        """
        x => complex wideband, shape => [L,], fs => sampling rate
        """
        nyq = fs * 0.5
        # clamp
        if lowcut < 1.0:
            lowcut = 1.0
        if highcut > (nyq - 1.0):
            highcut = nyq - 1.0
        import math

        if lowcut >= highcut:
            # trivial => skip
            return x

        taps = firwin(
            numtaps,
            [lowcut / nyq, highcut / nyq],
            pass_zero=False,
            window=("kaiser", beta),
        )
        from scipy.signal import filtfilt

        real_filtered = filtfilt(taps, [1.0], x.real)
        imag_filtered = filtfilt(taps, [1.0], x.imag)
        return real_filtered + 1j * imag_filtered

    def _downconvert_to_baseband(self, x, fs, fc):
        """
        x => complex array, shape => [L,]
        fc => center freq => we shift down by fc => multiply by exp(-j2pifc t)
        """
        N = len(x)
        t = np.arange(N) / fs
        phase = -1j * 2.0 * np.pi * fc * t
        shift = np.exp(phase)
        return x * shift
