##########################################
# dataset_wideband_yolo.py
##########################################
import torch
from torch.utils.data import Dataset

import os
import json
import numpy as np
from config_wideband_yolo import (
    S,  # number of grid cells
    B,  # boxes per cell
    NUM_CLASSES,
    SAMPLING_FREQUENCY,
    get_anchors,
)


class WidebandYoloDataset(Dataset):
    """
    Reads wideband .sigmf-data files from a directory.
    For each file, it:
      1) Bandpass filters and downconverts the signal (later in the model)
      2) Builds a YOLO label [S, B, (1+1+NUM_CLASSES)] with normalized frequency offsets.

    This version additionally computes the Fourier transform (using np.fft.rfft)
    of the IQ data so that a frequency–domain representation is available for the model.
    """

    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform

        # Gather all .sigmf-data files in this directory.
        self.files = [
            fname.replace(".sigmf-data", "")
            for fname in os.listdir(directory)
            if fname.endswith(".sigmf-data")
        ]
        self.files.sort()
        if len(self.files) == 0:
            raise RuntimeError(f"No .sigmf-data files found in {directory}!")

        # Build a label -> index mapping for classes.
        self.class_list = self._discover_mod_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # Determine num_samples from the first file.
        self.num_samples = self._find_num_samples(self.files[0])

    def _discover_mod_classes(self):
        all_mods = set()
        for base in self.files:
            meta_path = os.path.join(self.directory, base + ".sigmf-meta")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ann = meta["annotations"][0]
            mod_list = ann["rfml_labels"]["modclass"]
            if isinstance(mod_list, str):
                mod_list = [mod_list]
            all_mods.update(mod_list)
            if len(all_mods) >= NUM_CLASSES:
                break

        return sorted(all_mods)

    def _find_num_samples(self, base):
        # Check the first file to see how many samples (2*N interleaved)
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        with open(data_path, "rb") as f:
            iq_data = np.load(f)
        return len(iq_data) // 2

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base = self.files[idx]
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        meta_path = os.path.join(self.directory, base + ".sigmf-meta")

        # Load IQ data (time domain)
        with open(data_path, "rb") as f:
            iq_data = np.load(f)
        I = iq_data[0::2]
        Q = iq_data[1::2]
        x_complex = I + 1j * Q

        # Compute Fourier transform (using rfft to return nonnegative frequencies)
        x_fft = np.fft.rfft(x_complex)  # shape: (N_rfft,)
        # Stack real and imaginary parts (resulting shape: (2, N_rfft))
        x_freq = np.stack(
            [x_fft.real.astype(np.float32), x_fft.imag.astype(np.float32)], axis=0
        )

        # Load metadata.
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ann = meta["annotations"]
        snr_value = ann[1]["channel"]["snr"]
        center_freqs = ann[0]["center_frequencies"]
        mod_list = ann[0]["rfml_labels"]["modclass"]

        sampling_rate = ann[0]["sampling_rate"]  # Fs in Hz
        sps_list = meta["annotations"][1]["filter"]["sps"]
        beta = meta["annotations"][1]["filter"]["rolloff"]

        if isinstance(mod_list, str):
            mod_list = [mod_list]
        if isinstance(center_freqs, (float, int)):
            center_freqs = [center_freqs]
        if isinstance(sps_list, (int, float)):
            sps_list = [sps_list] * len(center_freqs)

        chan_bw = [(sampling_rate / sps) * (1.0 + beta) for sps in sps_list]

        # Normalise bandwidth to the width of one YOLO grid‑cell (“bin”)
        bin_width = (SAMPLING_FREQUENCY / 2) / S
        bw_norm = [(bw / bin_width) for bw in chan_bw]

        # Convert to real time-domain IQ: shape (2, N)
        x_real = x_complex.real.astype(np.float32)
        x_imag = x_complex.imag.astype(np.float32)
        x_wide = np.stack([x_real, x_imag], axis=0)  # shape (2, N)

        if self.transform:
            x_wide = self.transform(x_wide)
            # Optionally transform x_freq as well, if needed.

        # Build YOLO label: shape [S, B, 1+1+NUM_CLASSES]
        label_tensor = np.zeros((S, B, 1 + 1 + 1 + NUM_CLASSES), dtype=np.float32)
        for c_freq, m_str, bw_n in zip(center_freqs, mod_list, bw_norm):
            # Normalize frequency.
            freq_norm = c_freq / (SAMPLING_FREQUENCY / 2)  # in [0, 1]
            cell_idx = int(freq_norm * S)
            if cell_idx >= S:
                cell_idx = S - 1
            x_offset = (freq_norm * S) - cell_idx
            x_offset = np.clip(x_offset, 0.0, 1.0)

            # Obtain anchor values using linspace as defined in the config file.
            anchor_values = get_anchors()
            # Find the anchor index that is closest to the computed offset.
            anchor_idx = int(np.argmin(np.abs(anchor_values - x_offset)))

            # Assign the label for the best matching anchor.
            label_tensor[cell_idx, anchor_idx, 0] = x_offset
            label_tensor[cell_idx, anchor_idx, 1] = 1.0
            label_tensor[cell_idx, anchor_idx, 2] = bw_n  # bandwidth
            class_idx = self.class_to_idx.get(m_str, None)
            if class_idx is not None:
                label_tensor[cell_idx, anchor_idx, 3 + class_idx] = 1.0

        # Replace any NaN values in x_wide or x_freq with zeros.
        x_wide = np.nan_to_num(x_wide, nan=0.001)
        x_freq = np.nan_to_num(x_freq, nan=0.001)

        # Return time-domain IQ, frequency-domain representation, label, and SNR.
        return (
            torch.tensor(x_wide),
            torch.tensor(x_freq),
            torch.tensor(label_tensor),
            torch.tensor(snr_value, dtype=torch.float32),
        )
