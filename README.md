# DONE
 - Ensure random numbers are seeded to help with reproducibility - even in the C code.
 - Add the config file with random number seed and directory locations to make swapping between Eddie and local easier.
 - Ensure new code is functional with parameters in the system_config file.
 - Ensure random numbers are seeded correctly:
        - Test SNR/BER values for BPSK modulation scheme. Run the script twice, ensure you get same answer. Will need to decrease n_samples to something smaller for testing so it runs quicker.
        - Test Training a simple CNN model. Once again run twice, ensure same answer twice.
        - Check "Verifying Seeded Random.txt" Evidence of Output.
        - Can now seed for identical data generation and identical model training, ensuring repeatability of further experiments.
 - Move code off of Eddie & Get it working locally (should simply be a case of modifying the system_config) parameters.
 - Go through Eddie conda environment set up again (this should ensure it's there when Eddie comes back online.)
 - Complete "Verifying Seeded Random.txt" - simply run the code multiple times.
  - Create plots for BER/SNR across different channel models (should just be a case of running existing code and changing the channel model in configs/BER_Tests.json)

# TODO

Week Commencing 17/02/25:
 - Replicate parameters and model training from literature paper, should be able to get very similar results.
 - Should also add background noise across the entire spectrum – currently noise is only
        added to transmitted signals whereas we’d ideally want constant background noise across the
        entire spectrum. (THIS MIGHT ALREADY BE DONE)
 - Currently all signals generated on the wideband are of the same modulation scheme. This
should be relatively easy to fix.
 - Create a model that works on predicting signal modulation schemes in the wideband.
 Document the above changes (BER/SNR plots in weekly research report) + paragraph about seeding for reproducibility.
            - Document the system_config file in the read_me.md.
            - Address Popoolas comments on previous Project Report in this weeks project report.
            - Document everything currently in the "DONE" section.

Backburner:
1) Research "Standardised" Rician and Rayleigh multipath scenarios. e.g. path delays + path gains for city / rural / town environments.
2) Look into blind equalisation as a pre-processing step before ML model training for Rician and Rayleigh multipath fading. Other methods include maximum likelihood.

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

Datasets are saved in SigMF format. 
Each dataset is a *SigMF Archive* composed of multiple *SigMF Recordings*. 
Each *SigMF Recording* contains a single capture, saved as a binary file (.sigmf-data files), with an associated metadata file (.sigmf-meta) containing the parameters used to generate that capture. 
See the [SigMF specification](https://github.com/gnuradio/SigMF/blob/master/sigmf-spec.md) to read more. 

## Requirements & Setup

matplotlib==3.5.3
numpy==1.23.2
SigMF==1.1.1
tqdm==4.67.1
Pytorch (depends on CUDA) - https://pytorch.org/get-started/locally/

In addition to the python packages listed above, the code in this repo is dependent upon [liquid-dsp](https://github.com/jgaeddert/liquid-dsp). 
To install liquid-dsp, clone the repo linked, and follow the installation instructions in the README. 
Ensure that you rebind your dynamic libraries using `sudo ldconfig`.

Additionally, the first time using the synthetic RF dataset generator, you'll need to run

```
>> cd ./cmodules && make && cd ../
```

