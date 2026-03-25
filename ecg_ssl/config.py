from pathlib import Path
import random
import math
import numpy as np
import torch

# -------------------------
# Reproducibility / device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------------
# Paths
# -------------------------
BASE = Path("/data/ramzi/MIMIC_Dataset/physionet.org/files/mimic-iv-ecg/1.0")

# -------------------------
# Data / model
# -------------------------
PATCH_SIZE = 50                        # 50 samples = 100 ms at 500 Hz
SEQ_LEN = 5000 // PATCH_SIZE           # 100 tokens
IN_CHANNELS = 12
PATCH_DIM = IN_CHANNELS * PATCH_SIZE   # 600
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MLP_RATIO = 4
DROPOUT = 0.1

# -------------------------
# Masking
# -------------------------
MASK_RATIO = 0.20
MASK_SPAN_LEN = 1

# -------------------------
# Data loading
# -------------------------
MAX_RECORDS = None      # None = full dataset
TRAIN_FRAC = 0.95
BATCH_SIZE = 64
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
MAX_RETRIES = 50

# -------------------------
# Training
# -------------------------
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.05
LOG_EVERY = 50
PLOT_EVERY = 500
