import psutil

RESIZE = (64, 64)
DEPTH = 2
MODEL_NAME = f"petdata_crop{RESIZE[0]}_depth{DEPTH}"
OUTPUT_PATH = "./output/"
EPOCHS = 20  # Number of epochs to train

# Using Adam optimizer
LEARNING_RATE = 0.0001  # 0.00005
WEIGHT_DICE_LOSS = 0.85  # Combined loss weight for dice versus BCE

FEATURE_MAPS = 64
PRINT_MODEL = True  # Print the model

BLOCKTIME = 0
NUM_INTER_THREADS = 6
# Default is to use the number of physical cores available

# Figure out how many physical cores we have available
# Minimum of either the CPU affinity or the number of physical cores
NUM_INTRA_THREADS = min(len(psutil.Process().cpu_affinity()), psutil.cpu_count(logical=False))

USE_AUGMENTATION = True  # Use data augmentation during training
USE_DROPOUT = True