# TODO:

1. Get the center_frequency prediction working. This will hopefully increase the accuracy substantially.

- If predictions are close to each other they are probably predicting the same object - look into this.

2. Try it with more realistic data generation - see frequencies used for long wave RF communication and use them.

## Later:

1. Loss Function Change - Could look at a variable loss function - Have a similarity matrix between the different modulation schemes - e.g. getting confused between 32QAM and 64QAM is slightly more understandable than getting confused between BPSK and 64 QAM. Loss function change - start with just getting the center_frequency (high loss for frequencies) then shift focus to correct classification. Look into curriculum learning -> Start with only learning to get the center_frequency correct -> This is required for accurate detection of the class. See the Next Steps in previous weekly report.
2. Probably can't use the metadata to calculate the bandwidth of the signal as in reality, we wouldn't have access to this metadata. Better solution would be to either predict the bandwidth (in a similar way that the center_frequency is predicted) or use the YOLO style method of having set predefined bandwidths and then predicting the best one to use. See TODO in config_wideband_yolo.py - Bandwidth should be estimated by the CNN model - currently it is stated by calculating it from the sps, and sampling_rate. This should be predicted from the CNN as well as the centre_frequency. This might not be necessary - could do it without. Could also just use 4 different sizes of bandwidths, estimate the best one and then use that - similar to the YOLO model.
3. Similar to above - Currently symbol rate (baud rate) and AWGN SNR are determined first - meaning they are consistent for all center_frequencies. Each center frequency should have it's own baud rate and its on SNR value.

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
Each dataset is a _SigMF Archive_ composed of multiple _SigMF Recordings_.
Each _SigMF Recording_ contains a single capture, saved as a binary file (.sigmf-data files), with an associated metadata file (.sigmf-meta) containing the parameters used to generate that capture.
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

1. Activate WSL
   $ wsl
2. Activate virtual python environment
   $ source venv/bin/activate
   To close:
3. close the virtual environment
   $ deactivate
4. Exit wsl
   $ exit

### Notes to remake C code:

On the ECDF Compute cluster the environment variable paths to liquid-dsp install need to be set first, hence:

1. $ cd Synthetic-Radio-Frequency-Data-Generator/cmodules
2. $ export LD_LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LD_LIBRARY_PATH
3. $ export PATH=$HOME/liquid-dsp-install/bin:$PATH
4. $ export C_INCLUDE_PATH=$HOME/liquid-dsp-install/include:$C_INCLUDE_PATH
5. $ export LIBRARY_PATH=$HOME/liquid-dsp-install/lib:$LIBRARY_PATH
6. $ make
7. $ cd ../..

This has to be done every time the C code is modified.
