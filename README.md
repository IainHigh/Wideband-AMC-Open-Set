# TODO
FRIDAY:
- Fix Rayleigh implementation.
3) Plan plots + CNN models to show implementation of Rayleigh + Rician multipath. e.g. different parameters to plot / train with and compare to.
    - Get CNN-Model-From-Literature trained on Rician data at different SNRs - this will validate Rician + all data generation.
    3.1) Decide "Default" Rician & Rayleigh parameters (Rician mirror parameters from existing literature)
    3.2) Plots to create:
        - Rician with default parameters at SNR = 0; 10; 20.
        - Rayleigh with default parameters at SNR = 0; 10; 20.
        - The BPSK can then be compared to the plots from last week.
    ~~3.2) Create new model based on the implementation details in Deep Learning Based Automatic Modulation Classification Using Robust CNN~~
    3.3) Model training with Rician & Rayleigh (train on new model - Rician results should be identical as same dataset generation and model implementation):
        - Rician with default parameters at SNR ranges (-20 -> 20)
        - Rayleigh with default parameters at SNR ranges (-20 -> 20)
        - AWGN at SNR ranges (-20 -> 20)
- Record results in the report document.
- By EOB should have Rician + Rayleigh fully implemented + documented.
- Should be looking to move onto wideband after this.


4) Move to wideband data generation
- Move to wideband by combining multiple different narrowbands together at different frequencies?
5) Plot + train models on wideband data generation - this will require a new CNN model most likely.

    "channel": {
        "type": "rician",
        "awgn": true,               Include AWGN as well (True / False)
        "snr": 20,                  Signal to Noise Ratio (dB)
        "fo": 0.0,                  frequency offset (see below)
        "po": 0.0,                  phase offset (see below)
        "k_factor": 4.0,            Ratio of line-of-sight (LOS) to non-line-of-sight (NLOS). Higher value = more placed on LOS. K=0 = equal weight of LOS vs NLOS components.
        "num_taps": 3,              number of different NLOS paths.
        "path_delays": [0, 2, 3],   path delays - length should equal num_taps - measured in samples.
        "path_gains": [0, -2, -10]  path gains - length should equal num_taps - measured in dB. Usually negative.
    },

# NOTES TO RUN ON ECDF COMPUTE CLUSTER EDDIE:
### The conda environment set up will need to be run monthly (reminder in outlook calendar)
1)  $ cd /home/s2062378
2)  $ rm -rf /exports/eddie/scratch/s2062378
3)  $ mkdir /exports/eddie/scratch/s2062378/anaconda
4)  $ mkdir /exports/eddie/scratch/s2062378/data
5)  $ mkdir /exports/eddie/scratch/s2062378/anaconda/envs
6)  $ mkdir /exports/eddie/scratch/s2062378/anaconda/pkgs
7)  $ module load anaconda
8)  $ module load cuda
9)  $ conda create -n mypython python=3.7 matplotlib numpy tqdm scipy -y
10) $ conda activate mypython
11) $ conda install pip -y
12) $ pip3 install torch torchvision torchaudio
13) $ pip install SigMF==1.1.1

### To rebuild C code on Eddie:
Need to run the following commands in terminal before running make:
1) $ cd Synthetic-Radio-Frequency-Data-Generator/cmodules
2) $ export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
3) $ export PATH=$HOME/liquid-dsp-install/bin:$PATH
4) $ export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
5) $ export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH
6) $ make
7) $ cd ../..

# NOTES TO RUN LOCALLY:
1) Activate WSL
    $ wsl
2) Activate virtual python environment
    $ source venv/bin/activate
3) To run code:
    $ python3 generator.py ./configs/defaults.json
4) Output will be found in the data directory.
5) To run plotting python scripts:
    $ cd tests; python3 plot_dataset.py; cd ..

To close:
1) close the virtual environment
    $ deactivate
2) Exit wsl
    $ exit
  

# Requirements:
matplotlib==3.5.3
numpy==1.23.2
SigMF==1.1.1
tqdm==4.67.1
Pytorch (depends on CUDA) - https://pytorch.org/get-started/locally/

# Synthetic Radio Frequency Data Generator

Python tool to generate synthetic radio frequency (RF) datasets.

This repo contains code to synthetically generate 22 types of raw RF signals (psk, qam, fsk, analog modulation variants) in an Additive White Gaussian Noise (AWGN) channel via Python wrappers around [liquid-dsp](https://github.com/jgaeddert/liquid-dsp).

## Usage
Datasets are generated using the `generator.py` script.
For example, the following command will generate an example dataset, `./datasets/example.sigmf`.

```
python generator.py ./configs/example.json
``` 

A JSON configuration file must be provided on the command line which contains the desired dataset size, signal types, and signal generation parameters.
Basic error checking is performed in `./utils/config_utils.py`, and defaults parameters (set in `./configs/defaults.json`) are provided for any missing values.
Configuration files should contain the following parameters:

 - `n_captures`: the number of captures to generate per modulation scheme. e.g. 10 will create 10 different files for each modulation type.
 - `n_samps`: the number of raw IQ samples per capture. This will be the length the IQ list after taking samples.
 - `modulations`: the modulation schemes to include in the dataset (may include "bpsk", "qpsk", "8psk", "16psk", "4dpsk", "16qam", "32qam", "64qam", "16apsk", "32apsk", "fsk5k", "fsk75k", "gfsk5k", "gfsk75k", "msk", "gmsk", "fmnb", "fmwb", "dsb", "dsbsc", "lsb", "usb", and "awgn")
 - `symbol_rate`: number of symbols per frame, list of desired symbol rates accepted. Lower value means less samples per signal = more signals over total sample space. AKA Samples per Symbol.
 - `am_defaults`: default analog modulation parameters, including modulation index in the form [start, stop, step]
 - `fmnb_defaults`: default narrowband frequency modulation parameters, including modulation factor in the form [start, stop, step]
 - `fmwb_defaults`: default wideband frequency modulation parameters, including modulation factor in the form [start, stop, step]
 - `filter`: default transmit filter parameters, including the type of filter (Gaussian or root-raised cosine (RRC)), the excess bandwidth or `beta`, the symbol overlap or `delay`, and the fractional sample delay or `dt` (all gfsk/gmsk signals use Gaussian filters, all remaining fsk/msk signals use square filters, all psk/qam signals use RRC filters)
 - `channel`: synthetic channel parameters, including the type of channel (only AWGN is implemented, currently), signal-to-noise-ratio (`snr`), frequency offset (`fo`), and phase offset (`po`) in the form [start, stop, step]
 - `savepath`: the dataset location
 - `verbose`: 0 for minimal verbosity, 1 for debugging
 - `archive`: create a SigMF archive of dataset when complete

Datasets are saved in SigMF format. 
Each dataset is a *SigMF Archive* composed of multiple *SigMF Recordings*. 
Each *SigMF Recording* contains a single capture, saved as a binary file (.sigmf-data files), with an associated metadata file (.sigmf-meta) containing the parameters used to generate that capture. 
See the [SigMF specification](https://github.com/gnuradio/SigMF/blob/master/sigmf-spec.md) to read more. 

## Requirements & Setup

In addition to the python packages listed in `requirements.txt`, the code in this repo is dependent upon [liquid-dsp](https://github.com/jgaeddert/liquid-dsp). 
To install liquid-dsp, clone the repo linked, and follow the installation instructions in the README. 
Ensure that you rebind your dynamic libraries using `sudo ldconfig`.

Additionally, the first time using the synthetic RF dataset generator, you'll need to run

```
>> cd ./cmodules && make && cd ../
```

