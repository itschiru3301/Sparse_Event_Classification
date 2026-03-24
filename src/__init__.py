"""Sparse Autoencoder Module."""

from .configs import *
from .datasets import UnlabelledJetDataset, LabelledJetDataset

# Lazy import models and utils to handle optional spconv dependency
try:
    from .models import (
        SparseResBlock,
        SparseEncoder,
        SparseDecoder,
        SparseAutoencoder,
        SparseClassifier,
    )
    from .utils import (
        dense_to_sparse,
        get_prunable_modules,
        apply_pruning,
        get_sparsity,
        evaluate_classifier,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules require spconv: {e}")

__all__ = [
    # Configs
    "IN_CHANNELS",
    "BASE_CHANNELS",
    "LATENT_DIM",
    "BATCH_SIZE",
    "AE_EPOCHS",
    "CLS_EPOCHS",
    "MAX_SAMPLES",
    "RANDOM_SEED",
    # Models
    "SparseResBlock",
    "SparseEncoder",
    "SparseDecoder",
    "SparseAutoencoder",
    "SparseClassifier",
    # Datasets
    "UnlabelledJetDataset",
    "LabelledJetDataset",
    # Utils
    "dense_to_sparse",
    "get_prunable_modules",
    "apply_pruning",
    "get_sparsity",
    "evaluate_classifier",
]
