# TODO:

Why is GT occasionally above 0.5? Generator should only be generating up to half of the sampling rate. Investigate this:

Epoch [4/20]
Train: Loss=2.7274, MeanFreqErr=0.3604, ClsAcc=11.76%
Valid: Loss=2.7110, MeanFreqErr=0.3394, ClsAcc=11.70%

Some random frames from validation (only 2 shown):
Prediction format: (freq, class, conf)
GroundTruth format: (freq, class)
Frame 1:
Predicted => [(7171691.358089447, 6, 0.6346373558044434), (14084969.311952591, 5, 0.3096928894519806), (14278629.273176193, 5, 0.2818264663219452)]
GroundTruth=> [(0.6709756851196289, 6), (0.17191354930400848, 1)]
Frame 2:
Predicted => [(13368955.40356636, 5, 0.4721372723579407), (12175006.27040863, 5, 0.46962666511535645), (6083285.808563232, 5, 0.31380799412727356)]
GroundTruth=> [(0.8900049328804016, 1), (0.9723557233810425, 2)]

This is because the grount truth is being stored as the OFFSET relative to a unknown frequency_bin. This is why it's between [0, 1] and not [0, 0.5].

TODO:
Try moving back to using the x_offset for storing and prediction opposed to the relative frequency value. JUST BE SURE TO MODIFY THE MODEL AND LOSS TO UTILISE THE RAW FREQUENCY WHEN REQUIRED.

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
