import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json


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
