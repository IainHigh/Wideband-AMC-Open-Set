# Documentation of sytem_parameters.json

This JSON is designed to hold the system parameters which are required in several places by the Python code. By design, this should be the only file that should have to be modified to get the code functional and recreate results. It requires modifying the following parameters:

- `Working_Directory`: The full system path to where the Synthetic-Radio-Frequency-Data-Generator directory is found.
- `Dataset_Directory`: The full system path of the directory the generated dataset will be saved to.
- `Random_Seed`: The initial seed for all random number generation used throughout the code. By default 2025.

# Documentation of Config JSON Files

Configuration files should contain the following parameters:

- `n_captures`: the number of captures to generate per modulation scheme. e.g. 10 will create 10 different files for each modulation type.
- `n_samps`: the number of raw IQ samples per capture. This will be the length the IQ list after taking samples.
- `modulations`: the modulation schemes to include in the dataset (may include "bpsk", "qpsk", "8psk", "16psk", "4dpsk", "16qam", "32qam", "64qam", "16apsk", "32apsk", "fsk5k", "fsk75k", "gfsk5k", "gfsk75k", "msk", "gmsk", "fmnb", "fmwb", "dsb", "dsbsc", "lsb", "usb", and "awgn")
- `symbol_rate`: number of symbols per frame, list of desired symbol rates accepted. Lower value means less samples per signal = more signals over total sample space. AKA Samples per Symbol.
- `am_defaults`: default analog modulation parameters, including modulation index in the form [start, stop, step]
- `fmnb_defaults`: default narrowband frequency modulation parameters, including modulation factor in the form [start, stop, step]
- `fmwb_defaults`: default wideband frequency modulation parameters, including modulation factor in the form [start, stop, step]
- `filter`: default transmit filter parameters, including the type of filter (Gaussian or root-raised cosine (RRC)), the excess bandwidth or `beta`, the symbol overlap or `delay`, and the fractional sample delay or `dt` (all gfsk/gmsk signals use Gaussian filters, all remaining fsk/msk signals use square filters, all psk/qam signals use RRC filters)
- `channel`: synthetic channel parameters, including the type of channel, signal-to-noise-ratio (`snr`), frequency offset (`fo`), and phase offset (`po`) in the form [start, stop, step]
- `savepath`: the dataset location
- `verbose`: 0 for minimal verbosity, 1 for debugging
- `archive`: create a SigMF archive of dataset when complete

New parameters added for adapting to wideband:

- `sampling_rate`:
- `center_frequencies`:
- `randomly_generated_center_frequencies`: Ramdomly generate center frequency carriers: [lower_bound, upper_bound, number_of_carriers]. Should not be provided when `center_frequencies` is provided.

New channel model parameters:
`channel`: {
`type`: Can either be `awgn`, `rayleigh`, or `rician`
`awgn`: For `rayleigh` and `rician`: include AWGN as well (True / False)
`snr`: Signal to Noise Ratio (dB)
`fo`: Frequency offset (see above)
`po`: Phase offset (see above)
`k_factor`: For `rician`: the K-factor. The ratio of line-of-sight (LOS) to non-line-of-sight (NLOS).
`path_delays`: For `rician` and `rayleigh`: path delays - measured in samples
`path_gains`: For `rician` and `rayleigh`: path gains - measured in dB. Usually negative.
},

# Channel Model Parameters.

Part of this research involved looking into "standardised" channel model parameters for different scenarios. This is written about in much more detail in Progress Report 6 (24/02/25 - 07/03/25). For simplicity, the final parameters chosen are stated here for easy reference.

All scenarios below use Rician channel model with a K-factor of 5. The path delay is measured in samples, the path gain measured in dB. This is the same units that the config files use.

| **Scenario** | **Path Delay**    | **path_gain**                 |
| ------------ | ----------------- | ----------------------------- |
| Rural        | [0,2]             | [0,-8]                        |
| Suburban     | [0,8,22]          | [0,-3,-8]                     |
| Urban        | [0,4,16,24,46,74] | [0,-4.9,-6.9,-8.0,-7.8,-23.9] |
