"""Dataset classes for sparse autoencoder training."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UnlabelledJetDataset(Dataset):
    """Dataset for unlabelled jet calorimeter data."""
    
    def __init__(self, h5_path: str, key: str = "jets", max_samples: int = None):
        """
        Args:
            h5_path: Path to HDF5 file
            key: Key in HDF5 file containing data
            max_samples: Maximum number of samples to load (for memory efficiency)
        """
        self.h5_path = h5_path
        self.key = key
        with h5py.File(h5_path, "r") as f:
            available = list(f.keys())
            k = key if key in available else available[0]
            data = f[k][:max_samples]
        self.data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))).float()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class LabelledJetDataset(Dataset):
    """Dataset for labelled jet calorimeter data."""
    
    def __init__(self, h5_path: str, key_x: str = "jet", key_y: str = "Y", max_samples: int = None):
        """
        Args:
            h5_path: Path to HDF5 file
            key_x: Key in HDF5 file containing features
            key_y: Key in HDF5 file containing labels
            max_samples: Maximum number of samples to load (for memory efficiency)
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.X = torch.from_numpy(np.transpose(f[key_x][:max_samples], (0, 3, 1, 2))).float()
            y_data = f[key_y][:max_samples]
            if y_data.ndim > 1:
                y_data = y_data.squeeze()
            self.Y = torch.from_numpy(y_data).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx]
