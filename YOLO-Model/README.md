# TODO:

## Now:
- increase model size - currently training is pretty quick, model may just be fairly small - improved by increasing size. Should be able to do this purely from the parameters in config_wideband_yolo.py

## Later:
- Document this (write this readme)

- See TODO in config_wideband_yolo.py - Bandwidth should be estimated by the CNN model - currently it is stated by calculating it from the sps, and sampling_rate. This should be predicted from the CNN as well as the centre_frequency.

- Loss Function Change:
Could look at a variable loss function - Have a similarity matrix between the different modulation schemes - e.g. getting confused between 32QAM and 64QAM is slightly more understandable than getting confused between BPSK and 64 QAM. Loss function change - start with just getting the center_frequency (high loss for frequencies) then shift focus to correct classification. Look into curriculum learning -> Start with only learning to get the center_frequency correct -> This is required for accurate detection of the class.

- Add ability to save and load models - implement the ability to train a model for 48 hours on one qsub job, then continue training on another job. TRAIN -> SAVE -> JOB ENDS -> NEW JOB STARTS -> LOAD -> TRAIN -> REPEAT. ALLOWS US TO TRAIN FOR MORE THAN 48 HOURS. Only required if models take too long to train. Don't bother implementing yet.