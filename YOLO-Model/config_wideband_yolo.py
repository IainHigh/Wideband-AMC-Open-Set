#############################################
# config_wideband_yolo.py
#############################################
import json
import os
"""
Configuration file for the wideband YOLO-style AMC system.
All modifiable parameters are grouped here.
"""

def calculate_band_margin():
    with open("./configs/system_parameters.json") as f:
        system_parameters = json.load(f)
    dataset_directory = system_parameters["Dataset_Directory"] + "/training"
    file_name = os.listdir(os.path.abspath(dataset_directory))[0]
    file_name = os.path.join(dataset_directory, file_name.split(".")[0])
    
    with open(file_name + ".sigmf-meta") as _f:
        f_meta = json.load(_f)

    # Extract metadata
    annotation = f_meta["annotations"][0]
    sampling_rate = annotation["sampling_rate"]
    sps = f_meta["annotations"][1]["filter"]["sps"]
    beta = f_meta["annotations"][1]["filter"]["rolloff"]
    
    symbol_rate = sampling_rate / sps
    channel_bw = symbol_rate * (1 + beta)
    return channel_bw, sampling_rate
    
#####################
# Miscellaneous Parameters
#####################

VAL_PRINT_SAMPLES = 2 # The number of samples to print during validation. Helps to see how the model is doing.
PRINT_CONFIG_FILE = True # If True, will print the configuration file to the console.

#####################
# Dataset Filtering Parameters
#####################

BAND_MARGIN, SAMPLING_FREQUENCY = calculate_band_margin()  # Band margin - determines the start frequency and end frequency from the calculated center frequency.
NUMTAPS = 101 # Number of taps for the filter - Higher number of taps means better filtering but slower processing.

#####################
# Model Parameters
#####################
S = 4               # Number of grid cells
B = 1                # Boxes per cell
NUM_CLASSES = 9      # Number of classes

#####################
# Training Parameters
#####################
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0005

########################
# Loss Function Weights
########################
LAMBDA_COORD = 5.0       # Weight for coordinate (x offset) loss
LAMBDA_NOOBJ = 0.5       # Weight for confidence loss in no-object cells
LAMBDA_CLASS = 1.0       # Weight for classification loss
    

def print_config_file():
    """
    Print the configuration file to the console.
    """
    print("Configuration File:")
    print("\tVAL_PRINT_SAMPLES:", VAL_PRINT_SAMPLES)
    print("\tBAND_MARGIN:", BAND_MARGIN)
    print("\tNUMTAPS:", NUMTAPS)
    print("\tS:", S)
    print("\tB:", B)
    print("\tNUM_CLASSES:", NUM_CLASSES)
    print("\tBATCH_SIZE:", BATCH_SIZE)
    print("\tEPOCHS:", EPOCHS)
    print("\tLEARNING_RATE:", LEARNING_RATE)
    print("\tLAMBDA_COORD:", LAMBDA_COORD)
    print("\tLAMBDA_NOOBJ:", LAMBDA_NOOBJ)
    print("\tLAMBDA_CLASS:", LAMBDA_CLASS)
    print("")