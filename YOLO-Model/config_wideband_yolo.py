#############################################
# config_wideband_yolo.py
#############################################
"""
Configuration file for the wideband YOLO-style AMC system.
All modifiable parameters are grouped here.
"""

CREATE_NEW_DATASET = False

#####################
# Model saving/loading parameters
#####################

# TODO: Implement these (and implement testing + plotting of BER results + more detailed test analysis)
SAVE_MODEL = True
TEST_ONLY = False  # If True, will skip training and only test a previously saved model.
save_model_path = (
    "modulation_classifier.pth"  # Path to save/load model if test_only is True.
)

#####################
# Model Parameters
#####################
S = 16               # Number of grid cells
B = 2                # Boxes per cell
NUM_CLASSES = 9      # Number of classes
OUT_CHANNELS = B * (1 + 1 + NUM_CLASSES)
# Explanation:
#   1 -> x offset
#   1 -> confidence
#   NUM_CLASSES -> class probabilities

CNN_CHANNELS = [16, 32, 64] # Channels in the CNN backbone
KERNEL_SIZE = 9
STRIDE = 2
PADDING = 4

#####################
# Training Parameters
#####################
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.01

########################
# Loss Function Weights
########################
LAMBDA_COORD = 5.0       # Weight for coordinate (x offset) loss
LAMBDA_NOOBJ = 0.5       # Weight for confidence loss in no-object cells
LAMBDA_CLASS = 1.0       # Weight for classification loss
