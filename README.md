# TODO:
1) Currently symbol rate (baud rate) and AWGN SNR are determined first - meaning they are consistent for all center_frequencies. Each center frequency should have it's own baud rate and its on SNR value.
2) Implement YOLO style single step detection algorithm for wideband automatic modulation classification.

# Synthetic Radio Frequency Data Generator

Python tool to generate synthetic radio frequency (RF) datasets.

This repo contains code to synthetically generate 22 types of raw RF signals (psk, qam, fsk, analog modulation variants) in an Additive White Gaussian Noise (AWGN) channel via Python wrappers around [liquid-dsp](https://github.com/jgaeddert/liquid-dsp). This code is originally from a deprecated Intel project, further work for adapting to wideband, improved channel models, and implementation of ML models has been completed by Iain High.

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
In addition to the python packages listed in JobScripts/install_requirements.sh, the code in this repo is dependent upon [liquid-dsp](https://github.com/jgaeddert/liquid-dsp). 
To install liquid-dsp, clone the repo linked, and follow the installation instructions in the README. 
Ensure that you rebind your dynamic libraries using `sudo ldconfig`.

Additionally, the first time using the synthetic RF dataset generator, you'll need to run

```
>> cd ./cmodules && make && cd ../
```

Once requirements have been installed, the system_parameters also have to be modified. Details on this can be found in configs/README.md.

# Notes for Self:
This section details notes that are primarily useful for myself. These should be removed or heavily modified before final publication of the open-source software.

### NOTES TO RUN LOCALLY:
1) Activate WSL
    $ wsl
2) Activate virtual python environment
    $ source venv/bin/activate
To close:
1) close the virtual environment
    $ deactivate
2) Exit wsl
    $ exit

### Notes to remake C code:
On the ECDF Compute cluster the environment variable paths to liquid-dsp install need to be set first, hence:
1) $ cd Synthetic-Radio-Frequency-Data-Generator/cmodules
2) $ export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
3) $ export PATH=$HOME/liquid-dsp-install/bin:$PATH
4) $ export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
5) $ export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH
6) $ make
7) $ cd ../..

This has to be done every time the C code is modified.
