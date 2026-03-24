"""Configuration parameters for sparse autoencoder training."""

# Model Architecture
IN_CHANNELS = 8
BASE_CHANNELS = 32
LATENT_DIM = 256

# Training Parameters
BATCH_SIZE = 32
AE_EPOCHS = 30
CLS_EPOCHS = 30
MAX_SAMPLES = 100000  # Limit to avoid loading entire dataset

# Optimization
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5
CLS_HEAD_LR = 1e-3
CLS_FULL_LR = 1e-4
CLS_DROPOUT = 0.3

# Fine-tuning
HEAD_ONLY_EPOCHS = 5

# Pruning Analysis
PRUNING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Data Paths (relative to project root)
UNLABELLED_DATA_PATH = "/data/b23_chiranjeevi/EDE/Dataset_Specific_Unlabelled.h5"
LABELLED_DATA_PATH = "/data/b23_chiranjeevi/EDE/Dataset_Specific_labelled.h5"

# Model Weights Paths (relative to project root)
AE_WEIGHTS_PATH = "models/sparse_ae.pth"
CLASSIFIER_WEIGHTS_PATH = "models/sparse_classifier.pth"

# Data Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (inferred from remaining)

# Random Seed
RANDOM_SEED = 42

# Logging
LOG_DIR = "logs"
