#!/usr/bin/env python3
"""Phase 2 & 3: Classifier fine-tuning and pruning analysis."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import copy

sys.path.insert(0, os.path.dirname(__file__))

from src import (
    configs,
    LabelledJetDataset,
    SparseAutoencoder,
    SparseClassifier,
    dense_to_sparse,
    apply_pruning,
    get_sparsity,
    evaluate_classifier as eval_clf,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_classifier(model, train_loader, val_loader, optimizer, n_epochs=20, device=device):
    """Train classifier with frozen encoder."""
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            sparse_x = dense_to_sparse(x)
            optimizer.zero_grad()
            logits = model(sparse_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                sparse_x = dense_to_sparse(x)
                logits = model(sparse_x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")
    
    return {'train_loss': train_losses, 'val_loss': val_losses, 'val_acc': val_accs}


if __name__ == "__main__":
    torch.manual_seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)

    print("\n=== Loading Data ===")
    labelled_ds = LabelledJetDataset(
        configs.LABELLED_DATA_PATH,
        max_samples=configs.MAX_SAMPLES
    )
    
    N = len(labelled_ds)
    n_train = int(configs.TRAIN_RATIO * N)
    n_val = int(configs.VAL_RATIO * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        labelled_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(configs.RANDOM_SEED)
    )

    cls_train_loader = DataLoader(train_ds, batch_size=configs.BATCH_SIZE, shuffle=True, num_workers=4)
    cls_val_loader = DataLoader(val_ds, batch_size=configs.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=configs.BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n=== PHASE 2: Classifier Fine-tuning ===")
    ae_model = SparseAutoencoder(
        in_channels=configs.IN_CHANNELS,
        base_ch=configs.BASE_CHANNELS,
        latent_dim=configs.LATENT_DIM
    ).to(device)
    
    if os.path.exists(configs.AE_WEIGHTS_PATH):
        ae_model.load_state_dict(torch.load(configs.AE_WEIGHTS_PATH, weights_only=False), strict=False)
        print(f"✓ Loaded {configs.AE_WEIGHTS_PATH}")
    else:
        print(f"⚠ Warning: {configs.AE_WEIGHTS_PATH} not found")
    
    encoder = ae_model.encoder
    cls_model = SparseClassifier(encoder, latent_dim=configs.LATENT_DIM, num_classes=2).to(device)
    
    for p in cls_model.encoder.parameters():
        p.requires_grad = False
    
    cls_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cls_model.parameters()),
        lr=configs.CLS_HEAD_LR
    )
    print("Training head only...")
    for epoch in range(configs.HEAD_ONLY_EPOCHS):
        cls_model.train()
        for x, y in tqdm(cls_train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            sparse_x = dense_to_sparse(x)
            cls_opt.zero_grad()
            logits = cls_model(sparse_x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            cls_opt.step()
    
    for p in cls_model.encoder.parameters():
        p.requires_grad = True
    
    cls_opt = torch.optim.Adam(cls_model.parameters(), lr=configs.CLS_FULL_LR)
    print("Fine-tuning full model...")
    cls_history = train_classifier(cls_model, cls_train_loader, cls_val_loader, cls_opt, n_epochs=configs.CLS_EPOCHS, device=device)
    
    os.makedirs("models", exist_ok=True)
    torch.save(cls_model.state_dict(), configs.CLASSIFIER_WEIGHTS_PATH)
    print(f"✓ Saved {configs.CLASSIFIER_WEIGHTS_PATH}")

    print("\n=== PHASE 3: Pruning Analysis ===")
    prune_accs = []
    
    for ratio in configs.PRUNING_RATIOS:
        model_copy = copy.deepcopy(cls_model)
        if ratio > 0:
            apply_pruning(model_copy, ratio)
        acc = eval_clf(model_copy, test_loader, device=device)
        prune_accs.append(acc)
        sparsity = get_sparsity(model_copy)
        print(f"Pruning {ratio:.1%} → Test Acc: {acc:.4f} (Actual Sparsity: {sparsity:.4f})")
    
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(configs.PRUNING_RATIOS, prune_accs, marker='o', linewidth=2)
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Pruning on Classifier Accuracy')
    plt.grid(True)
    plt.savefig('images/pruning_analysis.png', dpi=100, bbox_inches='tight')
    print("✓ Saved images/pruning_analysis.png")
    
    print("\n✓ Finetuning complete!")
