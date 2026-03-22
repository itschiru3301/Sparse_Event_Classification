#!/usr/bin/env python3
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch.nn.utils.prune as prune
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# DATASETS
# ============================================================================

class LabelledJetDataset(Dataset):
    def __init__(self, h5_path, key_x="jet", key_y="Y", max_samples=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.X = torch.from_numpy(np.transpose(f[key_x][:max_samples], (0, 3, 1, 2))).float()
            y_data = f[key_y][:max_samples]
            if y_data.ndim > 1:
                y_data = y_data.squeeze()
            self.Y = torch.from_numpy(y_data).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================================
# SPARSE TENSOR UTILITIES
# ============================================================================

def dense_to_sparse(dense: torch.Tensor, threshold: float = 0.0) -> SparseConvTensor:
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


# ============================================================================
# SPARSE AUTOENCODER ARCHITECTURE
# ============================================================================

class SparseResBlock(nn.Module):
    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        self.conv1 = spconv.SubMConv2d(channels, channels, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv2d(channels, channels, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(F.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(out.features + x.features)
        out = out.replace_feature(F.relu(out.features))
        return out


class SparseEncoder(nn.Module):
    def __init__(self, in_channels: int = 8, base_ch: int = 32, latent_dim: int = 256):
        super().__init__()
        self.conv1 = spconv.SparseConv2d(in_channels, base_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(base_ch)
        self.res1 = SparseResBlock(base_ch, indice_key="res1")

        self.conv2 = spconv.SparseConv2d(base_ch, 2 * base_ch, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(2 * base_ch)
        self.res2 = SparseResBlock(2 * base_ch, indice_key="res2")

        self.conv3 = spconv.SparseConv2d(2 * base_ch, 4 * base_ch, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(4 * base_ch)
        self.res3 = SparseResBlock(4 * base_ch, indice_key="res3")

        self.latent_proj = nn.Linear(4 * base_ch, latent_dim)

    def forward(self, x: SparseConvTensor):
        x = self.conv1(x)
        x = x.replace_feature(self.bn1(x.features))
        x = x.replace_feature(F.relu(x.features))
        x = self.res1(x)

        x = self.conv2(x)
        x = x.replace_feature(self.bn2(x.features))
        x = x.replace_feature(F.relu(x.features))
        x = self.res2(x)

        x = self.conv3(x)
        x = x.replace_feature(self.bn3(x.features))
        x = x.replace_feature(F.relu(x.features))
        x = self.res3(x)

        feat = x.features
        batch_indices = x.indices[:, 0]
        batch_size = x.batch_size
        
        latents = []
        for b in range(batch_size):
            mask = batch_indices == b
            batch_feat = feat[mask].mean(dim=0)
            latents.append(batch_feat)
        latent = torch.stack(latents, dim=0)
        latent = self.latent_proj(latent)
        return latent


class SparseAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 8, base_ch: int = 32, latent_dim: int = 256):
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_ch, latent_dim)
        self.decoder = None

    def forward(self, x: SparseConvTensor):
        latent = self.encoder(x)
        return latent


class SparseClassifier(nn.Module):
    def __init__(self, encoder: SparseEncoder, latent_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: SparseConvTensor):
        latent = self.encoder(x)
        logits = self.head(latent)
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def train_classifier(model, train_loader, val_loader, optimizer, n_epochs=20, device=device):
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


# ============================================================================
# PRUNING
# ============================================================================

def get_prunable_modules(model):
    modules = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            modules.append((m, 'weight'))
    return modules


def apply_pruning(model, amount):
    modules = get_prunable_modules(model)
    prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)


def get_sparsity(model):
    total_params = 0
    pruned_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            if hasattr(param, 'weight_mask'):
                pruned_params += (param.weight_mask == 0).sum().item()
    return pruned_params / total_params if total_params > 0 else 0.0


def evaluate_classifier(model, loader, device=device):
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
    return correct / total


# ============================================================================
# MAIN: FINETUNING ONLY (PHASE 2 & 3)
# ============================================================================

if __name__ == "__main__":
    IN_CH = 8
    BASE_CH = 32
    LATENT_DIM = 256
    CLS_EPOCHS = 30
    BATCH_SIZE = 32
    MAX_SAMPLES = 100000

    print("\n=== Loading Data ===")
    labelled_ds = LabelledJetDataset("/data/b23_chiranjeevi/EDE/Dataset_Specific_labelled.h5", max_samples=MAX_SAMPLES)
    
    N = len(labelled_ds)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        labelled_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    cls_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    cls_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n=== PHASE 2: Classifier Fine-tuning ===")
    ae_model = SparseAutoencoder(in_channels=IN_CH, base_ch=BASE_CH, latent_dim=LATENT_DIM).to(device)
    ae_model.load_state_dict(torch.load("sparse_ae.pth", weights_only=False), strict=False)
    print("✓ Loaded sparse_ae.pth")
    
    encoder = ae_model.encoder
    cls_model = SparseClassifier(encoder, latent_dim=LATENT_DIM, num_classes=2).to(device)
    
    for p in cls_model.encoder.parameters():
        p.requires_grad = False
    
    cls_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=1e-3)
    print("Training head only...")
    for epoch in range(5):
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
    
    cls_opt = torch.optim.Adam(cls_model.parameters(), lr=1e-4)
    print("Fine-tuning full model...")
    cls_history = train_classifier(cls_model, cls_train_loader, cls_val_loader, cls_opt, n_epochs=CLS_EPOCHS, device=device)
    torch.save(cls_model.state_dict(), "sparse_classifier.pth")
    print("✓ Saved sparse_classifier.pth")

    print("\n=== PHASE 3: Pruning Analysis ===")
    prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_accs = []
    
    for ratio in prune_ratios:
        model_copy = copy.deepcopy(cls_model)
        if ratio > 0:
            apply_pruning(model_copy, ratio)
        acc = evaluate_classifier(model_copy, test_loader, device=device)
        prune_accs.append(acc)
        sparsity = get_sparsity(model_copy)
        print(f"Pruning {ratio:.1%} → Test Acc: {acc:.4f} (Actual Sparsity: {sparsity:.4f})")
    
    plt.figure(figsize=(10, 6))
    plt.plot(prune_ratios, prune_accs, marker='o', linewidth=2)
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Pruning on Classifier Accuracy')
    plt.grid(True)
    plt.savefig('pruning_analysis.png', dpi=100, bbox_inches='tight')
    print("✓ Saved pruning_analysis.png")
    
    print("\n✓ Finetuning complete!")
