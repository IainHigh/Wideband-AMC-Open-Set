#!/usr/bin/python3

import os
import sys
import json
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
np.Inf = np.inf  # Fix for a bug in the numpy library
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, "..")

save_plots = True
verbose = ctypes.c_int(0)
seed = ctypes.c_int(-1)

dataset_path = '../data/default'

rrc = ctypes.CDLL(os.path.abspath('../cmodules/rrc_rx'))
linear = ctypes.CDLL(os.path.abspath('../cmodules/linear_demodulate'))
am = ctypes.CDLL(os.path.abspath('../cmodules/am_demodulate'))
fm = ctypes.CDLL(os.path.abspath('../cmodules/fm_demodulate'))
fsk = ctypes.CDLL(os.path.abspath('../cmodules/fsk_demodulate'))

files = os.listdir(os.path.abspath(dataset_path))  # Both data and meta files
files = [os.path.join(dataset_path, f.split(".")[0]) for f in files]  # Add path
files = list(set(files))

for i in range(len(files)):
    f = files[i]

    ## Get meta
    with open(f + ".sigmf-meta") as _f:
        f_meta = json.load(_f)

    capture_start = f_meta["captures"][0]["core:sample_start"]
    capture_len = f_meta["captures"][0]["core:length"]

    f_meta = f_meta["annotations"][0]

    ## Get data
    with open(f + ".sigmf-data", 'rb') as _f:
        f_data = np.load(_f)
    I = f_data[capture_start * 2 : (capture_len + capture_start) * 2 : 2]
    Q = f_data[(capture_start * 2) + 1 : ((capture_len + capture_start) * 2) + 1 : 2]
    modscheme = f_meta["rfml_labels"]["modclass"]

    # Density-based coloring
    xy = np.vstack([I, Q])
    z = gaussian_kde(xy)(xy)  # Compute density
    idx = z.argsort()         # Sort points by density for better visualization
    I, Q, z = I[idx], Q[idx], z[idx]

    # Plot constellation diagram
    plt.figure()
    plt.title(f"Constellation Diagram ({modscheme})")
    plt.scatter(I, Q, c=z, s=1, cmap='viridis')  # Density-colored scatter plot
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')  # Horizontal axis
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')  # Vertical axis
    plt.xlabel('I (In-phase)')
    plt.ylabel('Q (Quadrature)')
    plt.grid(True, linestyle='--', alpha=0.5)
    cbar = plt.colorbar()  # Add colorbar to explain density
    cbar.set_label('Density')
    plt.savefig(f'../tests/figures/Constellation-Diagrams/constellation_{modscheme}.png')
    plt.close()
