#############################################
# config_wideband_yolo.py
#############################################
from numpy import linspace

"""
Configuration file for the wideband YOLO-style AMC system.
All modifiable parameters are grouped here.
"""


def get_anchors():
    """
    Compute evenly spaced anchor values inside (0,1) based on the number of boxes per cell (B).
    For example, for B=4 this returns [0.2, 0.4, 0.6, 0.8].
    """
    # Compute B evenly spaced points between 1/(B+1) and B/(B+1)
    return linspace(1 / (B + 1), B / (B + 1), B)


#####################
# Mahalanobis Open-Set Recognition Parameters
#####################

OPENSET_ENABLE = True  # master switch
OPENSET_COVERAGE = 0.95  # tail kept inside each class Gaussian
UNKNOWN_CLASS_NAME = "UNKNOWN"
EMBED_DIM = 96  # TODO: Don't have this as a configurable parameter, but rather a constant in the model.

#####################
# Miscellaneous Parameters
#####################

VAL_PRINT_SAMPLES = 0  # The number of samples to print during validation. Helps to see how the model is doing.
VALIDATE_MODEL = False
PRINT_CONFIG_FILE = True  # If True, will print the configuration file to the console.
WRITE_TEST_RESULTS = False  # If True, will write the test results to a file.
GENERATE_CONFUSION_MATRIX = True  # Generate a confusion matrix after training.
PLOT_TEST_SAMPLES = False  # If True, will plot the test samples and predictions.
MULTIPLE_JOBS_PER_TRAINING = False  # If true, will save the model after each validation step. When the current job script is finished, it will start the next job script and resume training from the last saved model.
MODULATION_CLASSES = []

#####################
# Dataset Filtering Parameters
#####################

SAMPLING_FREQUENCY = 1e9
MERGE_SIMILAR_PREDICTIONS = False  # Merge similar predictions into one prediction. TODO: Investigate why when this is True, the model crashes with NaN error with one transmitter.
MERGE_SIMILAR_PREDICTIONS_THRESHOLD = (
    SAMPLING_FREQUENCY / 15
)  # The threshold for merging similar predictions. If the distance between two predictions is less than this value, they will be merged.
NUMTAPS = 101  # Number of taps for the filter - Higher number of taps means better filtering but slower processing.

#####################
# Model Parameters
#####################
S = 8  # Number of grid cells
B = 4  # Anchors / Boxes per cell
NUM_CLASSES = 7  # Number of classes

#####################
# Training Parameters
#####################
BATCH_SIZE = 64
EPOCHS = 75
LEARNING_RATE = 0.001  # Initial learning rate
FINAL_LR_MULTIPLE = 0.005  # Final learning rate multiple - the final learning rate will be this multiple of the initial learning rate.

########################
# Loss Function Weights
########################
LAMBDA_NOOBJ = 0.5  # Weight for confidence loss in no-object cells

LAMBDA_COORD = 1.0  # Weight for coordinate (x offset) loss
LAMBDA_BW = 1.0

LAMBDA_CLASS = 2.0  # Weight for classification loss
LAMBDA_CENTER = 5.0  # Weight for the embedding distance loss

CONFIDENCE_THRESHOLD = 0.16  # Confidence threshold for filtering predictions


def print_config_file():
    """
    Print the configuration file to the console.
    """
    print("Configuration File:")
    print("\tVAL_PRINT_SAMPLES:", VAL_PRINT_SAMPLES)
    print("\tPLOT_TEST_SAMPLES:", PLOT_TEST_SAMPLES)
    print("\tWRITE_TEST_RESULTS:", WRITE_TEST_RESULTS)
    print("\tGENERATE_CONFUSION_MATRIX:", GENERATE_CONFUSION_MATRIX)
    print("\tMULTIPLE_JOBS_PER_TRAINING:", MULTIPLE_JOBS_PER_TRAINING)
    print("\tSAMPLING_FREQUENCY:", SAMPLING_FREQUENCY)
    print("\tMERGE_SIMILAR_PREDICTIONS:", MERGE_SIMILAR_PREDICTIONS)
    if MERGE_SIMILAR_PREDICTIONS:
        print(
            "\tMERGE_SIMILAR_PREDICTIONS_THRESHOLD:",
            MERGE_SIMILAR_PREDICTIONS_THRESHOLD,
        )
    print("\tNUMTAPS:", NUMTAPS)
    print("\tS:", S)
    print("\tB:", B)
    print("\tNUM_CLASSES:", NUM_CLASSES)
    print("\tBATCH_SIZE:", BATCH_SIZE)
    print("\tEPOCHS:", EPOCHS)
    print("\tLEARNING_RATE:", LEARNING_RATE)
    print("\tFINAL_LR_MULTIPLE:", FINAL_LR_MULTIPLE)
    print("\tLOSS WEIGHT LAMBDAS:")
    print("\t\tLAMBDA_NOOBJ:", LAMBDA_NOOBJ)
    print("\t\tLAMBDA_COORD:", LAMBDA_COORD)
    print("\t\tLAMBDA_BW:", LAMBDA_BW)
    print("\t\tLAMBDA_CLASS:", LAMBDA_CLASS)
    print("\t\tLAMBDA_CENTER:", LAMBDA_CENTER)

    print("\tCONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)
    if OPENSET_ENABLE:
        print("\tOPENSET RECOGNITION PARAMETERS:")
        print("\t\tOPENSET_COVERAGE:", OPENSET_COVERAGE)
        print("\t\tUNKNOWN_CLASS_NAME:", UNKNOWN_CLASS_NAME)
        print("\t\tEMBED_DIM:", EMBED_DIM)
    print("")
