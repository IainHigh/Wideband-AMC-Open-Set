##########################################
# dataset_wideband_yolo.py
##########################################
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from config_wideband_yolo import (
    S,      # number of grid cells
    B,      # boxes per cell
    NUM_CLASSES,
    SAMPLING_FREQUENCY
)

class WidebandYoloDataset(Dataset):
    """
    Reads wideband .sigmf-data files from a directory.
    For each file, it:
      1) Bandpass filter around [min_center_freq - margin, max_center_freq + margin]
      2) Downconvert that filtered chunk so the middle is baseband
      3) Builds a YOLO label [S, B, (1+1+NUM_CLASSES)] with x_offset in [0,1], etc.

    If multiple signals exist, we find min and max of 'center_frequencies' (plus margin).
    """

    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform

        # Gather all .sigmf-data files in this directory
        self.files = [
            fname.replace(".sigmf-data", "")
            for fname in os.listdir(directory)
            if fname.endswith(".sigmf-data")
        ]
        self.files.sort()
        if len(self.files) == 0:
            raise RuntimeError(f"No .sigmf-data files found in {directory}!")

        # Build a label -> index mapping for classes
        self.class_list = self._discover_mod_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # Determine num_samples from the first file
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
            for m in mod_list:
                all_mods.add(m)
                if len(all_mods) == NUM_CLASSES:
                    return sorted(all_mods)
        return sorted(all_mods)

    def _find_num_samples(self, base):
        # Check the first file to see how many samples (2*N interleaved)
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        with open(data_path, "rb") as f:
            iq_data = np.load(f)
        return len(iq_data)//2

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base = self.files[idx]
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        meta_path = os.path.join(self.directory, base + ".sigmf-meta")

        # Load IQ
        with open(data_path, "rb") as f:
            iq_data = np.load(f)
        I = iq_data[0::2]
        Q = iq_data[1::2]
        x_complex = I + 1j*Q

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ann = meta["annotations"][0]

        center_freqs = ann["center_frequencies"]
        if isinstance(center_freqs, (float,int)):
            center_freqs = [center_freqs]
        mod_list = ann["rfml_labels"]["modclass"]
        if isinstance(mod_list, str):
            mod_list = [mod_list]

        # convert to real 2xN
        # x_real = x_base.real.astype(np.float32)
        # x_imag = x_base.imag.astype(np.float32)
        x_real = x_complex.real.astype(np.float32)
        x_imag = x_complex.imag.astype(np.float32)
        
        x_wide = np.stack([x_real, x_imag], axis=0)  # shape (2, N)

        # optional transform
        if self.transform:
            x_wide = self.transform(x_wide)

        # 4) Build YOLO label => shape [S, B, 1+1+NUM_CLASSES]
        #   But now we treat freq offsets relative to c_mid
        label_tensor = np.zeros((S, B, 1 + 1 + NUM_CLASSES), dtype=np.float32)

        for c_freq, m_str in zip(center_freqs, mod_list):
            # 1) Normalize frequency
            freq_norm = c_freq / SAMPLING_FREQUENCY  # in [0, 1] if your c_freq <= FREQ_MAX

            # 2) find which cell
            cell_idx = int(freq_norm * S)
            if cell_idx >= S:
                cell_idx = S - 1  # clamp

            # 3) offset in [0,1]
            x_offset = (freq_norm * S) - cell_idx
            if x_offset < 0: 
                x_offset = 0.0
            if x_offset > 1: 
                x_offset = 1.0

            # 4) fill YOLO label at (cell_idx, b)
            for b_i in range(B):
                # if no object assigned yet
                if label_tensor[cell_idx, b_i, 1] == 0.0:
                    label_tensor[cell_idx, b_i, 0] = x_offset  # store offset 
                    label_tensor[cell_idx, b_i, 1] = 1.0       # confidence=1
                    class_idx = self.class_to_idx.get(m_str, None)
                    if class_idx is not None:
                        label_tensor[cell_idx, b_i, 2 + class_idx] = 1.0
                    break
        
        # Retrieve the SNR value as well, this is useful for final results of accuracy and error per SNR value.
        try:
            snr_value = meta["annotations"][1]["channel"]["snr"]
        except (KeyError, IndexError):
            snr_value = None

        return torch.tensor(x_wide), torch.tensor(label_tensor), torch.tensor(snr_value, dtype=torch.float32)