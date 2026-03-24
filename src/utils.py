"""Utility functions for sparse tensor operations and pruning."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv


def dense_to_sparse(dense: torch.Tensor, threshold: float = 0.0) -> SparseConvTensor:
    """
    Convert dense tensor to sparse tensor representation.
    
    Args:
        dense: Dense tensor of shape (B, C, H, W)
        threshold: Threshold for sparsification
        
    Returns:
        SparseConvTensor representation
    """
    B, C, H, W = dense.shape
    spatial_shape = [H, W]
    active_mask = (dense.abs().sum(dim=1) > threshold)
    batch_indices = torch.where(active_mask)
    indices = torch.stack(batch_indices, dim=1).int().to(dense.device)
    features = []
    for b in range(B):
        mask_b = active_mask[b]
        feat_b = dense[b, :, mask_b]
        features.append(feat_b.T)
    features = torch.cat(features, dim=0).to(dense.device)
    sparse_tensor = SparseConvTensor(
        features=features.float(),
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=B
    )
    return sparse_tensor


def get_prunable_modules(model: nn.Module) -> list:
    """
    Get list of prunable modules in a model.
    
    Args:
        model: Neural network model
        
    Returns:
        List of (module, parameter_name) tuples
    """
    modules = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            modules.append((m, 'weight'))
    return modules


def apply_pruning(model: nn.Module, amount: float) -> None:
    """
    Apply global L1 unstructured pruning to model.
    
    Args:
        model: Neural network model
        amount: Fraction of weights to prune (0.0 to 1.0)
    """
    modules = get_prunable_modules(model)
    prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)


def get_sparsity(model: nn.Module) -> float:
    """
    Calculate sparsity percentage of a model.
    
    Args:
        model: Neural network model
        
    Returns:
        Sparsity ratio (fraction of pruned parameters)
    """
    total_params = 0
    pruned_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            if hasattr(param, 'weight_mask'):
                pruned_params += (param.weight_mask == 0).sum().item()
    return pruned_params / total_params if total_params > 0 else 0.0


def evaluate_classifier(model: nn.Module, loader, device: torch.device) -> float:
    """
    Evaluate classifier accuracy on a loader.
    
    Args:
        model: Classifier model
        loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Accuracy as fraction
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            sparse_x = dense_to_sparse(x)
            logits = model(sparse_x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0
