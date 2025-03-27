# TODO:

1. Get it functional - Ensure that the loss and frequency_err are both decreasing and the classification accuracy is increasing.
2. Check for all TODOs in the code. Either fix them or move them to here.
3. Optimise the model - currently 3 hours per epoch is way too long, need to get it back down to 1 hour per epoch. Also depends on how fast it is on the GPU.
4. For output, display the frequency as their raw value in Hz, currently it's as a ratio of the sampling rate. Should be an easy fix - ensure it's implemented for both the prediction and ground truth values. See the Next Steps in previous weekly report.
5. Documentation & Refactoring. Ensure that all code is well written, commented, and this ReadMe is written. Transfer code to local machine and run it through a linting tool.
6. Loss Function Change:
   Could look at a variable loss function - Have a similarity matrix between the different modulation schemes - e.g. getting confused between 32QAM and 64QAM is slightly more understandable than getting confused between BPSK and 64 QAM. Loss function change - start with just getting the center_frequency (high loss for frequencies) then shift focus to correct classification. Look into curriculum learning -> Start with only learning to get the center_frequency correct -> This is required for accurate detection of the class. See the Next Steps in previous weekly report.

## Later:

- See TODO in config_wideband_yolo.py - Bandwidth should be estimated by the CNN model - currently it is stated by calculating it from the sps, and sampling_rate. This should be predicted from the CNN as well as the centre_frequency. This might not be necessary - could do it without. Could also just use 4 different sizes of bandwidths, estimate the best one and then use that - similar to the YOLO model.

- Add ability to save and load models - implement the ability to train a model for 48 hours on one qsub job, then continue training on another job. TRAIN -> SAVE -> JOB ENDS -> NEW JOB STARTS -> LOAD -> TRAIN -> REPEAT. ALLOWS US TO TRAIN FOR MORE THAN 48 HOURS. Only required if models take too long to train. Don't bother implementing yet.
