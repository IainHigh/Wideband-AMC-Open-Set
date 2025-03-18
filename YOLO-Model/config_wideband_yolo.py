#############################################
# config_wideband_yolo.py
#############################################
"""
Configuration file for the wideband YOLO-style AMC system.
All modifiable parameters are grouped here.
"""

#####################
# Miscellaneous Parameters
#####################

CREATE_NEW_DATASET = False # If True, will generate a new dataset. If False, it will use the (presumably) existing dataset.
VAL_PRINT_SAMPLES = 2 # The number of samples to print during validation. Helps to see how the model is doing.
PRINT_CONFIG_FILE = True # If True, will print the configuration file to the console.

#####################
# Dataset Filtering Parameters
#####################

BAND_MARGIN = 1e6  # Band margin - determines the start frequency and end frequency from the calculated center frequency. TODO: CALCULATE THIS DON'T JUST ASSUME IT.
NUMTAPS = 101 # Number of taps for the filter - Higher number of taps means better filtering but slower processing.
BETA = 8.6 # Beta value for the filter - Higher beta means better filtering but slower processing.

#####################
# Model Parameters
#####################
S = 16               # Number of grid cells
B = 2                # Boxes per cell
NUM_CLASSES = 9      # Number of classes

STRIDE = 2          # Stride for the first conv layer
INIT_CHANNELS = 32  # for first conv
NUM_BLOCKS = 4      # how many repeated residual blocks
BLOCK_OUT_CH = 96   # output channels of each block
KERNEL_SIZE = 8    # kernel size for the residual block

#####################
# Training Parameters
#####################
BATCH_SIZE = 512
EPOCHS = 1
LEARNING_RATE = 0.01

########################
# Loss Function Weights
########################
LAMBDA_COORD = 2.0       # Weight for coordinate (x offset) loss
LAMBDA_NOOBJ = 5.0       # Weight for confidence loss in no-object cells
LAMBDA_CLASS = 1.0       # Weight for classification loss
IOU_SCALING_FACTOR = 1e6 # Scaling factor for IOU loss - Should be same magnitude as the frequency ranges we're working with.

def print_config_file():
    """
    Print the configuration file to the console.
    """
    print("Configuration File:")
    print("CREATE_NEW_DATASET:", CREATE_NEW_DATASET)
    print("VAL_PRINT_SAMPLES:", VAL_PRINT_SAMPLES)
    print("BAND_MARGIN:", BAND_MARGIN)
    print("NUMTAPS:", NUMTAPS)
    print("BETA:", BETA)
    print("S:", S)
    print("B:", B)
    print("NUM_CLASSES:", NUM_CLASSES)
    print("STRIDE:", STRIDE)
    print("INIT_CHANNELS:", INIT_CHANNELS)
    print("NUM_BLOCKS:", NUM_BLOCKS)
    print("BLOCK_OUT_CH:", BLOCK_OUT_CH)
    print("KERNEL_SIZE:", KERNEL_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("LAMBDA_COORD:", LAMBDA_COORD)
    print("LAMBDA_NOOBJ:", LAMBDA_NOOBJ)
    print("LAMBDA_CLASS:", LAMBDA_CLASS)