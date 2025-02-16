# Documentation of Config JSON Files

# TODO: THIS HAS TO BE UPDATED TO INCLUDE THE FOLLOWING:
    "sampling_rate": 20e6,
    "center_frequencies": [5e6],

    "channel": {
        "type": "rician",
        "awgn": true,               Include AWGN as well (True / False)
        "snr": 20,                  Signal to Noise Ratio (dB)
        "fo": 0.0,                  frequency offset (see below)
        "po": 0.0,                  phase offset (see below)
        "k_factor": 4.0,            Ratio of line-of-sight (LOS) to non-line-of-sight (NLOS). Higher value = more placed on LOS. K=0 = equal weight of LOS vs NLOS components.
        "path_delays": [0, 2, 3],   path delays - measured in samples
        "path_gains": [0, -2, -10]  path gains - measured in dB. Usually negative.
    },
    
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