import os
import torch
import numpy as np
import json
from scipy.signal import filtfilt, firwin, find_peaks
from torch.utils.data import Dataset


class WidebandModulationDataset(Dataset):
    """
    Minimal recursion-friendly dataset. Returns:
      - wideband signal (shape [2, length])
      - metadata dict (with center frequencies + modulation classes)
      - f_start, f_end (initial frequency range for recursion)
    """

    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.files = [
            f.replace(".sigmf-data", "")
            for f in os.listdir(directory)
            if f.endswith(".sigmf-data")
        ]
        self.transform = transform
        self.label_to_idx = self._build_label_mapping()  # Unchanged

    def _fir_bandpass_complex(self, x, fs, lowcut, highcut, numtaps=101, beta=8.6):
        """
        Example bandpass for complex signals using FIR filter.
        """
        nyq = fs * 0.5
        if lowcut <= 0:
            lowcut = 1.0
        if highcut >= nyq:
            highcut = nyq - 1.0
        taps = firwin(
            numtaps,
            [lowcut / nyq, highcut / nyq],
            pass_zero=False,
            window=("kaiser", beta),
        )
        real_filtered = filtfilt(taps, [1.0], x.real)
        imag_filtered = filtfilt(taps, [1.0], x.imag)
        return real_filtered + 1j * imag_filtered

    def _downconvert_to_baseband(self, x, fs, fc):
        """
        Multiply by exp(-j 2 pi fc t) to shift the channel to baseband.
        """
        N = len(x)
        t = np.arange(N) / float(fs)
        return x * np.exp(-1j * 2.0 * np.pi * fc * t)

    def _build_label_mapping(self):
        labels = set()
        for base in self.files:
            meta_path = os.path.join(self.directory, base + ".sigmf-meta")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # The first annotation often has 'rfml_labels'
            ann = meta["annotations"][0]
            mod_list = ann["rfml_labels"]["modclass"]
            if isinstance(mod_list, str):
                labels.add(mod_list)
            else:
                for m in mod_list:
                    labels.add(m)
        return {lab: i for i, lab in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
          wideband_tensor: [2, length] float32
          metadata_dict: {
              "modulation_classes": [list of class indices],
              "center_frequencies": [list of float centers]
          }
          f_start: float
          f_end: float
        """
        base = self.files[idx]
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        meta_path = os.path.join(self.directory, base + ".sigmf-meta")

        with open(data_path, "rb") as f:
            data = np.load(f)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Rebuild wideband signal (unchanged)
        I = data[0::2]
        Q = data[1::2]
        x_wide = I + 1j * Q

        # Grab metadata
        annotation = meta["annotations"][0]
        fs = annotation["sampling_rate"]
        mod_list = annotation["rfml_labels"]["modclass"]
        center_freqs = annotation["center_frequencies"]
        if isinstance(mod_list, str):
            mod_list = [mod_list]
        if isinstance(center_freqs, float):
            center_freqs = [center_freqs]

        # Convert each mod scheme to label index
        class_indices = [self.label_to_idx[m] for m in mod_list]

        # We define an initial freq range (0, fs/2.0) or whichever you prefer
        f_start = 0.0
        f_end = fs / 2.0

        # For demonstration, let's skip the sub-sample bandpass & downconversion
        # and just return the entire wideband signal as x_stacked
        # If you still want to do a partial bandpass, you can do so here
        # Example: (commented out)
        # x_filt = self._fir_bandpass_complex(x_wide, fs, f_start, f_end)
        # x_stacked = np.stack([x_filt.real, x_filt.imag], axis=0)

        x_stacked = np.stack([x_wide.real, x_wide.imag], axis=0)

        if self.transform:
            x_stacked = self.transform(x_stacked)

        wideband_tensor = torch.tensor(x_stacked, dtype=torch.float32)
        metadata_dict = {
            "modulation_classes": class_indices,
            "center_frequencies": center_freqs,
        }

        return wideband_tensor, metadata_dict, f_start, f_end


##################################################
#   ModulationDataset - unchanged (example usage)
##################################################
class ModulationDataset(Dataset):
    """
    Narrowband or single-signal dataset (unchanged).
    Included for reference, if needed.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".sigmf-data")]
        self.labels, self.snrs = self._extract_labels_and_snrs()
        self.transform = transform

    def _extract_labels_and_snrs(self):
        labels = []
        snrs = []
        for file in self.files:
            meta_file = file.replace(".sigmf-data", ".sigmf-meta")
            meta_path = os.path.join(self.data_dir, meta_file)

            with open(meta_path, "r") as f:
                metadata = json.load(f)

                # Extract modulation class
                label = metadata["annotations"][0]["rfml_labels"]["modclass"]
                labels.append(label)

                # Extract SNR (assuming the SNR is found in the "channel" section)
                try:
                    snr = metadata["annotations"][1]["channel"]["snr"]
                except (KeyError, IndexError):
                    snr = None

                snrs.append(snr)

        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return [self.label_to_idx[label] for label in labels], snrs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.files[idx])
        label = self.labels[idx]
        snr = self.snrs[idx]

        iq_data = np.load(data_path)
        I = iq_data[0::2]
        Q = iq_data[1::2]
        x = np.stack([I, Q], axis=0)  # shape: [2, N]

        if self.transform:
            x = self.transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            snr,
        )
