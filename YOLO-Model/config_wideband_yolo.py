#############################################
# config_wideband_yolo.py
#############################################
import json
import os
import numpy as np

"""
Configuration file for the wideband YOLO-style AMC system.
All modifiable parameters are grouped here.
"""


def get_anchors():
    """
    Compute evenly spaced anchor values inside (0,1) based on the number of boxes per cell (B).
    For example, for B=4 this returns [0.2, 0.4, 0.6, 0.8].
    """
    from config_wideband_yolo import B  # ensure B is imported from config

    # Compute B evenly spaced points between 1/(B+1) and B/(B+1)
    return np.linspace(1 / (B + 1), B / (B + 1), B)


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

VAL_PRINT_SAMPLES = 2  # The number of samples to print during validation. Helps to see how the model is doing.
PRINT_CONFIG_FILE = True  # If True, will print the configuration file to the console.
GENERATE_CONFUSION_MATRIX = (
    True  # If True, will generate a confusion matrix after training.
)
MULTIPLE_JOBS_PER_TRAINING = True  # If true, will save the model after each validation step. When the current job script is finished, it will start the next job script and resume training from the last saved model.
MODULATION_CLASSES = [] # The modulation classes will be determined by the dataset discovery process.


#####################
# Dataset Filtering Parameters
#####################

BAND_MARGIN, SAMPLING_FREQUENCY = (
    calculate_band_margin()
)  # Band margin - determines the start frequency and end frequency from the calculated center frequency.
BAND_MARGIN = BAND_MARGIN * 2  # Band margin - determines the start frequency and end frequency from the calculated center frequency.
NUMTAPS = 101  # Number of taps for the filter - Higher number of taps means better filtering but slower processing.

#####################
# Model Parameters
#####################
S = 4  # Number of grid cells
B = 4  # Anchors / Boxes per cell
NUM_CLASSES = 7  # Number of classes

#####################
# Training Parameters
#####################
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005 # Initial learning rate
FINAL_LR_MULTIPLE = 0.01 # Final learning rate multiple - the final learning rate will be this multiple of the initial learning rate.

########################
# Loss Function Weights
########################
USE_SIMILARITY_MATRIX = True  

# Dictionary for inter-modulation similarity scores.
# Keys are unordered tuples of modulation names (e.g., ("BPSK", "QPSK") is equivalent to ("QPSK", "BPSK")).
# Adjust the scores as needed. Lower values imply a lower penalty for confusion.
SIMILARITY_DICT = {
    ("bpsk", "qpsk"): 1.5,
    ("bpsk", "8psk"): 2.0,
    ("bpsk", "16qam"): 3.0,
    ("bpsk", "32qam"): 4.0,
    ("bpsk", "16apsk"): 3.5,
    ("bpsk", "32apsk"): 4.0,
    ("qpsk", "8psk"): 1.6,
    ("qpsk", "16qam"): 2.2,
    ("qpsk", "32qam"): 3.0,
    ("qpsk", "16apsk"): 2.8,
    ("qpsk", "32apsk"): 3.2,
    ("8psk", "16qam"): 1.4,
    ("8psk", "32qam"): 2.0,
    ("8psk", "16apsk"): 1.8,
    ("8psk", "32apsk"): 2.2,
    ("16qam", "32qam"): 1.2,
    ("16qam", "16apsk"): 1.5,
    ("16qam", "32apsk"): 1.8,
    ("32qam", "16apsk"): 1.3,
    ("32qam", "32apsk"): 1.1,
    ("16apsk", "32apsk"): 1.0,
}

LAMBDA_COORD = 5.0  # Weight for coordinate (x offset) loss
LAMBDA_NOOBJ = 0.5  # Weight for confidence loss in no-object cells
LAMBDA_CLASS = 1.0  # Weight for classification loss


def print_config_file():
    """
    Print the configuration file to the console.
    """
    print("Configuration File:")
    print("USE SIMILARITY MATRIX:", USE_SIMILARITY_MATRIX)
    print("\tVAL_PRINT_SAMPLES:", VAL_PRINT_SAMPLES)
    print("\tGENERATE_CONFUSION_MATRIX:", GENERATE_CONFUSION_MATRIX)
    print("\tMULTIPLE_JOBS_PER_TRAINING:", MULTIPLE_JOBS_PER_TRAINING)
    print("\tSAMPLING_FREQUENCY:", SAMPLING_FREQUENCY)
    print("\tBAND_MARGIN:", BAND_MARGIN)
    print("\tNUMTAPS:", NUMTAPS)
    print("\tS:", S)
    print("\tB:", B)
    print("\tNUM_CLASSES:", NUM_CLASSES)
    print("\tBATCH_SIZE:", BATCH_SIZE)
    print("\tEPOCHS:", EPOCHS)
    print("\tLEARNING_RATE:", LEARNING_RATE)
    print("\tFINAL_LR_MULTIPLE:", FINAL_LR_MULTIPLE)
    print("\tLAMBDA_COORD:", LAMBDA_COORD)
    print("\tLAMBDA_NOOBJ:", LAMBDA_NOOBJ)
    print("\tLAMBDA_CLASS:", LAMBDA_CLASS)
    print("")
