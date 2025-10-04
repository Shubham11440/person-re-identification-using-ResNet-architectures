# config.py

import torch

class Config:
    # --- General ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # --- Dataset ---
    DATA_DIR = "./data"
    MARKET_1501_DIR = f"{DATA_DIR}/Market-1501-v15.09.15"
    CUHK03_DIR = f"{DATA_DIR}/cuhk03/detected" # or 'labeled'
    DUKEMTMC_DIR = f"{DATA_DIR}/DukeMTMC-reID"

    # --- Model ---
    NUM_PARTS = 6
    USE_GAUSSIAN_SMOOTHING = False # Set to False as recommended

    # --- Training ---
    EPOCHS = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 0.02
    WEIGHT_DECAY = 5e-4
    LR_SCHEDULER = "cosine" # or 'step'
    WARMUP_EPOCHS = 5

    # --- Loss Function ---
    CE_WEIGHT = 1.0
    TRIPLET_WEIGHT = 1.0
    TRIPLET_MARGIN = 0.3
    LABEL_SMOOTHING_EPSILON = 0.1

# Instantiate the config
cfg = Config()