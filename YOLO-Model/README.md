# TODO:
0) Document this (write this readme)
1) See TODO in config_wideband_yolo.py
2) Look into curriculum learning
3) Improve complexity of CNN model.
4) Add remaining features that are in generic CNN into this model.
5) Make sure randomness is correctly seeded.
6) Bandwidth should be estimated by the CNN model - currently it is stated by calculating it from the sps, and sampling_rate. This should be predicted from the CNN as well as the centre_frequency.
7) Could look at a variable loss function - Have a similarity matrix between the different modulation schemes - e.g. getting confused between 32QAM and 64QAM is slightly more understandable than getting confused between BPSK and 64 QAM.