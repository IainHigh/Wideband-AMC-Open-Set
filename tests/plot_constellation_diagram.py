#!/usr/bin/python3

import os
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, "..")

import json
import ctypes
import numpy as np
import matplotlib.pyplot as plt
np.Inf = np.inf  # Fix for a bug in the numpy library
from utils import mod_str2int as mod_map

save_plots = True
verbose = ctypes.c_int(0)
seed = ctypes.c_int(-1)

idx = 0
example_len = 512
dataset_path = '../data/perfect_signal'

rrc = ctypes.CDLL(os.path.abspath('../cmodules/rrc_rx'))
linear = ctypes.CDLL(os.path.abspath('../cmodules/linear_demodulate'))
am = ctypes.CDLL(os.path.abspath('../cmodules/am_demodulate'))
fm = ctypes.CDLL(os.path.abspath('../cmodules/fm_demodulate'))
fsk = ctypes.CDLL(os.path.abspath('../cmodules/fsk_demodulate'))

files = os.listdir(os.path.abspath(dataset_path))  # both data and meta files
files = [os.path.join(dataset_path, f.split(".")[0]) for f in files]  # add path
files = list(set(files))

for i in range(len(files)):
    f = files[i]

    ## Get metadata
    with open(f + ".sigmf-meta") as _f:
        f_meta = json.load(_f)

    capture_start = f_meta["captures"][0]["core:sample_start"]
    capture_len = f_meta["captures"][0]["core:length"]

    f_meta = f_meta["annotations"][0]

    ## Get data
    with open(f + ".sigmf-data", 'rb') as _f:
        f_data = np.load(_f)
    I = f_data[capture_start * 2 : (example_len + capture_start) * 2 : 2]
    Q = f_data[(capture_start * 2) + 1 : ((example_len + capture_start) * 2) + 1 : 2]
    modscheme = f_meta["rfml_labels"]["modclass"]

    ## Calculate amplitude for color intensity
    amplitude = np.sqrt(I**2 + Q**2)

    ## Apply threshold to amplitude
    max_amplitude = np.max(amplitude)
    threshold = 0.9 * max_amplitude
    mask = amplitude > threshold

    ## Filter I, Q, and amplitude values
    I_filtered = I[mask]
    Q_filtered = Q[mask]
    amplitude_filtered = amplitude[mask]

    ## Plot constellation diagram
    plt.figure(figsize=(8, 8))
    plt.scatter(I_filtered, Q_filtered, c=amplitude_filtered, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Amplitude')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title(f"Constellation Diagram: {modscheme} (Filtered)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.savefig(f'../tests/figures/constellation_{modscheme}_{i}_filtered.png')
    plt.close()
