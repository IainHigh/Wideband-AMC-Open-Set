import os
import torch
import json
import numpy as np
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

        x_stacked = np.stack([x_wide.real, x_wide.imag], axis=0)

        if self.transform:
            x_stacked = self.transform(x_stacked)

        wideband_tensor = torch.tensor(x_stacked, dtype=torch.float32)
        metadata_dict = {
            "modulation_classes": class_indices,
            "center_frequencies": center_freqs,
        }

        return wideband_tensor, metadata_dict, f_start, f_end
