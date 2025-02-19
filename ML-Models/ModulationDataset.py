import os
import torch
import numpy as np
import json
from scipy.signal import butter, filtfilt, firwin, find_peaks
from torch.utils.data import Dataset


class WidebandModulationDataset(Dataset):
    """
    For each wideband .sigmf-data file, detect multiple signals, return a list of sub-samples.
    Each sub-sample is a single modulation with shape (2, sub_signal_len).
    """

    def __init__(self, directory, transform=None, sub_signal_len=1024):
        super().__init__()
        self.directory = directory
        self.files = [
            f.replace(".sigmf-data", "")
            for f in os.listdir(directory)
            if f.endswith(".sigmf-data")
        ]
        self.transform = transform
        self.sub_signal_len = sub_signal_len

        # Build label -> index mapping by scanning all metadata
        self.label_to_idx = self._build_label_mapping()

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
        Return:
          sub_samples: list of shape [2, sub_signal_len] Tensors
          sub_labels:  list of int
          sub_snrs:    list of floats (or None)
        """
        base = self.files[idx]
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        meta_path = os.path.join(self.directory, base + ".sigmf-meta")

        with open(data_path, "rb") as f:
            data = np.load(f)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Rebuild complex wideband
        I = data[0::2]
        Q = data[1::2]
        x_wide = I + 1j * Q

        # Grab some metadata
        annotation = meta["annotations"][0]
        fs = annotation["sampling_rate"]

        # These might be lists if multiple signals are present
        mod_list = annotation["rfml_labels"]["modclass"]
        center_freqs = annotation["center_frequencies"]
        if isinstance(mod_list, str):
            mod_list = [mod_list]
        if isinstance(center_freqs, float):
            center_freqs = [center_freqs]

        # For SNR (if you store it in "channel" or "channel_params" etc.)
        # We'll store one SNR per sub-signal, or None if not found.
        snr_value = None
        if "channel" in annotation:
            snr_value = annotation["channel"].get("snr", None)

        # 1) We'll check we "detected" these center frequencies, or just proceed.
        #    Example: simple approach is to do an FFT peak detection. Adjust threshold as needed.
        X = np.fft.fftshift(np.fft.fft(x_wide))
        mag = np.abs(X)
        # Find peaks that are at least 20% of the max amplitude, separated by 1/10 of length
        peak_height = mag.max() * 0.2
        distance = int(0.1 * len(X))
        peaks, _ = find_peaks(mag, height=peak_height, distance=distance)
        # Convert peak indices to frequencies
        freqs = np.fft.fftshift(np.fft.fftfreq(len(x_wide), 1 / fs))
        detected_freqs = freqs[peaks]

        sub_samples = []
        sub_labels = []
        sub_snrs = []

        # 2) For each known center frequency in meta, produce sub-sample
        for i, fc in enumerate(center_freqs):
            mod_str = mod_list[i]
            label_idx = self.label_to_idx[mod_str]

            # Optionally check if fc is near any freq in detected_freqs
            # (If you do exact matching, or some tolerance approach.)
            # For now, we skip that check and always bandpass at fc.

            # Example channel BW. Adjust as needed:
            bw = fs / 20.0  # e.g. 1/20 of sampling rate
            lowcut = fc - (bw / 2)
            highcut = fc + (bw / 2)

            # Bandpass around [lowcut, highcut]
            x_filt = self._fir_bandpass_complex(x_wide, fs, lowcut, highcut)

            # Downconvert to baseband
            x_base = self._downconvert_to_baseband(x_filt, fs, fc)

            # Convert to (2, L)
            x_real = np.real(x_base)
            x_imag = np.imag(x_base)
            x_stacked = np.stack([x_real, x_imag], axis=0)

            # Optional transform
            if self.transform:
                x_stacked = self.transform(x_stacked)

            sub_samples.append(torch.tensor(x_stacked, dtype=torch.float32))
            sub_labels.append(label_idx)
            # If multiple signals have different SNR in your meta, adapt accordingly:
            sub_snrs.append(snr_value if snr_value is not None else -999.0)

        return sub_samples, sub_labels, sub_snrs


class ModulationDataset(Dataset):
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

                # Extract SNR (assuming the SNR is found in the "channel" section of the second annotation)
                try:
                    snr = metadata["annotations"][1]["channel"]["snr"]
                except (KeyError, IndexError):
                    snr = None  # If SNR is missing, set to None or handle appropriately

                snrs.append(snr)

        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return [self.label_to_idx[label] for label in labels], snrs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.files[idx])
        label = self.labels[idx]
        snr = self.snrs[idx]  # Extract SNR for this sample

        # Load I/Q data
        iq_data = np.load(data_path)
        I = iq_data[0::2]
        Q = iq_data[1::2]
        x = np.stack([I, Q], axis=0)  # Shape: [2, N]

        if self.transform:
            x = self.transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            snr,
        )
