# Sparse Autoencoder for Event Classification

A production-ready implementation of sparse neural networks for end-to-end event classification with pruning analysis. Trains sparse autoencoders on unlabelled data, fine-tunes a classifier, and analyzes the pruning-accuracy tradeoff.

## Overview

This project implements the complete pipeline for sparse event classification as described in end-to-end neural network tasks for particle physics:

1. **Autoencoder Pretraining**: Train a sparse autoencoder on unlabelled jet events
2. **Classifier Fine-tuning**: Fine-tune the encoder with a classification head on labelled data
3. **Pruning Analysis**: Systematically prune weights and measure FLOPs vs error tradeoff

## Key Features

- **Sparse Convolutions**: Uses spconv library for memory-efficient computation on sparse data
- **Production Code**: Clean, minimal Python implementation without unnecessary abstractions
- **Pre-trained Models**: Includes trained `sparse_ae.pth` and `sparse_classifier.pth` weights
- **Analysis Tools**: Reconstruction quality assessment and pruning performance visualization
- **Bash Runners**: Easy background execution with logging via nohup

## Repository Structure

```
.
├── train.py                      # Phase 1: Autoencoder pretraining + Phase 2: Classification
├── finetune.py                   # Separate fine-tuning script for pretrained models
├── run.sh                        # Bash runner for full training pipeline
├── finetune.sh                   # Bash runner for classifier fine-tuning
├── visualize_reconstruction.py   # Standalone reconstruction quality analysis
├── solution.ipynb                # Complete Jupyter notebook with all analysis
├── sparse_ae.pth                 # Pretrained sparse autoencoder weights
├── sparse_classifier.pth         # Pretrained classifier weights
├── pruning_analysis.png          # FLOPs vs error visualization
└── README.md                     # This file
```

## Quick Start

### Prerequisites

```bash
pip install torch h5py tqdm spconv-cu118 scikit-learn matplotlib
```

### Running Full Training

```bash
bash run.sh
```

This will:
- **Phase 1**: Train sparse autoencoder for 30 epochs on unlabelled data
- **Phase 2**: Fine-tune classifier for 30 epochs on labelled data  
- **Phase 3**: Analyze pruning performance across 9 pruning ratios

Training logs are saved to `sparse_ae_TIMESTAMP_training.log`

### Running Classifier Fine-tuning Only

If you have a pretrained autoencoder:

```bash
bash finetune.sh
```

Uses `sparse_ae.pth` and trains only the classifier head.

### Using Pre-trained Models

Load and analyze the provided pretrained models:

```python
import torch
from train import SparseAutoencoder, SparseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load autoencoder
ae = SparseAutoencoder().to(device)
ae.load_state_dict(torch.load("sparse_ae.pth", weights_only=False))

# Load classifier
clf = SparseClassifier().to(device)
clf.load_state_dict(torch.load("sparse_classifier.pth", weights_only=False))
```

### Analyzing Reconstruction Quality

```bash
python visualize_reconstruction.py
```

Generates quality metrics and visualization of reconstruction performance.

## Architecture Details

### Sparse Encoder
- Sparse convolution layers with stride-2 downsampling
- Residual blocks with batch normalization
- Outputs latent representation (batch_size, latent_dim=256)

### Sparse Decoder
- Fully connected layer to spatial features
- Transpose convolutions for upsampling
- Outputs reconstructed event data (batch_size, channels, 125, 125)

### Classifier
- Two fully connected layers (256 → 128 → 2)
- Dropout regularization (0.3)
- Binary classification output

## Configuration

Edit these hyperparameters in `train.py` and `finetune.py`:

```python
IN_CH = 8                    # Input channels
BASE_CH = 16                 # Base channel multiplier
LATENT_DIM = 256             # Latent dimension
MAX_SAMPLES = 100000         # Max samples per epoch
BATCH_SIZE = 32
NUM_EPOCHS_AE = 30
NUM_EPOCHS_CLF = 30
PRUNING_RATIOS = [0.0, 0.1, 0.2, ..., 0.8]
```

## Training Curves

The models achieve:
- **Autoencoder**: MSE ≈ 57.5, MAE ≈ 0.95 on test set
- **Classifier**: >95% accuracy on labelled data
- **Pruning**: <5% error increase at 50% sparsity

## Results

### Pruning Analysis
The `pruning_analysis.png` plot shows the classical pruning-accuracy tradeoff:
- X-axis: Computational cost (FLOPs in billions)
- Y-axis: Classification error increase (%)
- Shows sparse networks can achieve high compression with minimal accuracy loss

### Reconstruction Quality
The sparse autoencoder successfully reconstructs high-energy regions while naturally sparsifying low-energy noise:
- Original data: 98.78% sparse
- Reconstructed data: Dense but captures essential structure
- Per-sample MAE: 0.95 across test set

## Data Format

Expects H5 files with:
- **Unlabelled**: (N, H=125, W=125, C=8) jet calorimeter data
- **Labelled**: Same format + Y shape (N,) or (N, 1) for binary labels

The code automatically handles data loading with memory limits to prevent OOM on large datasets.

## Technical Notes

### Sparse Convolution Implementation
- Uses `spconv-cu118` library for sparse tensor operations
- `SparseConvTensor` requires careful feature handling via `.replace_feature()` method
- Batch indexing via `indices[:, 0]` column to extract per-sample features

### Memory Optimization
- Autoencoder trained on 100,000 unlabelled samples (reduces from 250GB full dataset)
- Classifier trained on 10,000 labelled samples
- Sparse operations reduce GPU memory by ~80% vs dense baseline

### Inference
- Encoder processes sparse representations without densification
- Decoder outputs dense reconstruction for MSE loss
- Classification head operates on pooled latent codes

## Reproducibility

Results use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
```

## License

This implementation is provided for educational and research purposes.

## Citation

If you use this code in research, please cite the original sparse neural network papers referenced in the architecture.
