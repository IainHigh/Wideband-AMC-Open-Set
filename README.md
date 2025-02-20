# DONE
- None

# TODO
- Document the system_config file in the read_me.md. Write more read_me documents - for plotting and CNNs. Ensure readme's are fully updated.
- See TODO in ModulationDataset.py: TODO: RIGHT NOW THIS IS ESSENTIALLY CHEATING BY USING THE CENTER FREQUENCY. YOU SHOULD BE DETECTING THE SIGNALS INSTEAD.
- Research "Standardised" Rician and Rayleigh multipath scenarios. e.g. path delays + path gains for city / rural / town environments.
- Symbol rate / baud rate should be unique for each center frequency?

Backburner:
1) Look into blind equalisation as a pre-processing step before ML model training for Rician and Rayleigh multipath fading. Other methods include maximum likelihood.

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

