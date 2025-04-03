# TODO (This Week):

1. Get the center_frequency prediction working. This will hopefully increase the accuracy substantially.
2. Documentation & Refactoring. Ensure that all code is well written, commented, and this ReadMe is written. Transfer code to local machine and run it through a linting tool.
3. Try it with more realistic data generation - see frequencies used for long wave RF communication and use them.

## Later:

1. Loss Function Change - Could look at a variable loss function - Have a similarity matrix between the different modulation schemes - e.g. getting confused between 32QAM and 64QAM is slightly more understandable than getting confused between BPSK and 64 QAM. Loss function change - start with just getting the center_frequency (high loss for frequencies) then shift focus to correct classification. Look into curriculum learning -> Start with only learning to get the center_frequency correct -> This is required for accurate detection of the class. See the Next Steps in previous weekly report.
2. Probably can't use the metadata to calculate the bandwidth of the signal as in reality, we wouldn't have access to this metadata. Better solution would be to either predict the bandwidth (in a similar way that the center_frequency is predicted) or use the YOLO style method of having set predefined bandwidths and then predicting the best one to use. See TODO in config_wideband_yolo.py - Bandwidth should be estimated by the CNN model - currently it is stated by calculating it from the sps, and sampling_rate. This should be predicted from the CNN as well as the centre_frequency. This might not be necessary - could do it without. Could also just use 4 different sizes of bandwidths, estimate the best one and then use that - similar to the YOLO model.
