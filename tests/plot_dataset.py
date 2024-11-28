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
np.Inf = np.inf # Fix for a bug in the numpy library 
from utils import mod_str2int as mod_map

save_plots = True
verbose = ctypes.c_int(0)
seed = ctypes.c_int(-1)

idx = 0
example_len=100
dataset_path = '../data/default'

rrc = ctypes.CDLL(os.path.abspath('../cmodules/rrc_rx'))
linear = ctypes.CDLL(os.path.abspath('../cmodules/linear_demodulate'))
am = ctypes.CDLL(os.path.abspath('../cmodules/am_demodulate'))
fm = ctypes.CDLL(os.path.abspath('../cmodules/fm_demodulate'))
fsk = ctypes.CDLL(os.path.abspath('../cmodules/fsk_demodulate'))

files = os.listdir(os.path.abspath(dataset_path))    # both data and meta files
files = [os.path.join(dataset_path, f.split(".")[0]) for f in files] # add path
files = list(set(files))

for i in range(len(files)):
    f = files[i]

    ## get meta
    with open(f + ".sigmf-meta") as _f:
        f_meta = json.load(_f)

    capture_start = f_meta["captures"][0]["core:sample_start"]
    capture_len = f_meta["captures"][0]["core:length"]

    f_meta = f_meta["annotations"][0] 

    ## get data
    with open(f + ".sigmf-data", 'rb') as _f:
        f_data = np.load(_f)
    I = f_data[capture_start*2 : (example_len+capture_start)*2 : 2]
    Q = f_data[(capture_start*2)+1 : ((example_len+capture_start)*2)+1 : 2]
    modscheme = f_meta["rfml_labels"]["modclass"]

    plt.figure()
    plt.title("Rx Samples")
    plt.plot(I, label='I')
    plt.plot(Q, label='Q')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('../tests/figures/'+modscheme+str(i)+'.png')
    plt.close()