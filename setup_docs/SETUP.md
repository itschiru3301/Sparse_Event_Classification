# Setup Guide - Sparse Autoencoder

## Table of Contents
1. [Quick Setup](#quick-setup)
2. [Manual Setup](#manual-setup)
3. [Troubleshooting](#troubleshooting)
4. [Configuration](#configuration)

## Quick Setup

The fastest way to get started is using the automated setup script:

```bash
# From the project root directory
bash setup.sh
```

This script will:
- Verify Python 3 installation
- Create necessary directories (`src/`, `models/`, `images/`, `logs/`)
- Install all dependencies from `requirements.txt`
- Verify CUDA availability (if installed)
- Test module imports

## Manual Setup

If you prefer to set up manually:

### Step 1: Create Directory Structure

```bash
mkdir -p src models images logs setup_docs
```

### Step 2: Install Python Dependencies

```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install from requirements.txt
python3 -m pip install -r requirements.txt
```

**Note on CUDA:** The default requirements include `spconv-cu118` for CUDA 11.8. If you need a different CUDA version, adjust as needed:
- `spconv-cu102` for CUDA 10.2
- `spconv-cu111` for CUDA 11.1

### Step 3: Verify Installation

```bash
# Test PyTorch and CUDA
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# Test module imports
python3 -c "from src import SparseAutoencoder, configs; print('✓ Modules imported')"
```

## Running the Pipeline

### Full Training (Phase 1, 2 & 3)

Trains a sparse autoencoder from scratch, fine-tunes a classifier, and analyzes pruning:

```bash
python3 -m src.train
```

Or as a background job with logging:

```bash
nohup python3 -m src.train > logs/training_$(date +%s).log 2>&1 &
```

### Classifier Fine-tuning Only (Phase 2 & 3)

Fine-tune the classifier using a pre-trained autoencoder:

```bash
python3 -m src.finetune
```

## Configuration

All hyperparameters and paths are centralized in `src/configs.py`:

```python
# Model Architecture
IN_CHANNELS = 8
BASE_CHANNELS = 32
LATENT_DIM = 256

# Training
BATCH_SIZE = 32
AE_EPOCHS = 30
CLS_EPOCHS = 30
MAX_SAMPLES = 100000

# Data Paths
UNLABELLED_DATA_PATH = "/data/b23_chiranjeevi/EDE/Dataset_Specific_Unlabelled.h5"
LABELLED_DATA_PATH = "/data/b23_chiranjeevi/EDE/Dataset_Specific_labelled.h5"

# Model Weights
AE_WEIGHTS_PATH = "models/sparse_ae.pth"
CLASSIFIER_WEIGHTS_PATH = "models/sparse_classifier.pth"
```

To customize, edit `src/configs.py` before running the pipeline.

## Project Structure

```
sparse_ae_minimal/
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── configs.py                # Configuration parameters
│   ├── models.py                 # Model architectures
│   ├── datasets.py               # Dataset classes
│   ├── utils.py                  # Utility functions
│   ├── train.py                  # Full training pipeline
│   └── finetune.py               # Fine-tuning pipeline
├── models/                        # Pre-trained weights
│   ├── sparse_ae.pth             # Autoencoder weights
│   └── sparse_classifier.pth     # Classifier weights
├── images/                        # Generated visualizations
│   ├── pruning_analysis.png
│   └── reconstruction_quality.png
├── logs/                          # Training logs
├── setup_docs/                    # Documentation
│   ├── SETUP.md                  # This file
│   └── ARCHITECTURE.md           # Model architecture details
├── train.py                       # Entry point for training
├── finetune.py                    # Entry point for fine-tuning
├── run.sh                         # Bash runner (legacy)
├── finetune.sh                    # Bash runner (legacy)
├── setup.sh                       # Setup script
├── requirements.txt               # Python dependencies
├── solution.ipynb                 # Jupyter notebook
└── README.md                      # Main documentation
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'spconv'`

**Solution:** Install spconv with the correct CUDA version:
```bash
pip install spconv-cu118  # For CUDA 11.8
```

### Issue: `CUDA out of memory`

**Solution:** Reduce batch size or max samples in `src/configs.py`:
```python
BATCH_SIZE = 16  # Was 32
MAX_SAMPLES = 50000  # Was 100000
```

### Issue: Data files not found

**Solution:** Update paths in `src/configs.py`:
```python
UNLABELLED_DATA_PATH = "/path/to/your/unlabelled/data.h5"
LABELLED_DATA_PATH = "/path/to/your/labelled/data.h5"
```

### Issue: `No module named 'src'`

**Solution:** Ensure you're running from the project root directory:
```bash
cd /data/b23_chiranjeevi/EDE/sparse_ae_minimal
python3 -m src.train
```

### Issue: GPU not being used

**Solution:** Verify CUDA installation:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

If CUDA is not available but installed, reinstall PyTorch with correct CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Reproducibility

Results use fixed random seeds for reproducibility:

```python
torch.manual_seed(42)
np.random.seed(42)
```

These are set automatically in the training scripts.

## Next Steps

1. Review model architecture: See `setup_docs/ARCHITECTURE.md`
2. Customize training: Edit `src/configs.py`
3. Run training: `python3 -m src.train`
4. Analyze results: Check `images/` directory
5. Use trained models: See README.md for usage examples
