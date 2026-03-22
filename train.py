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

class UnlabelledJetDataset(Dataset):
    def __init__(self, h5_path, key="jets", max_samples=None):
        self.h5_path = h5_path
        self.key = key
        with h5py.File(h5_path, "r") as f:
            available = list(f.keys())
            k = key if key in available else available[0]
            data = f[k][:max_samples]
        self.data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class LabelledJetDataset(Dataset):
    def __init__(self, h5_path, key_x="jet", key_y="Y", max_samples=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.X = torch.from_numpy(np.transpose(f[key_x][:max_samples], (0, 3, 1, 2))).float()
            self.Y = torch.from_numpy(f[key_y][:max_samples]).long()

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
        batch_indices = x.indices[:, 0]  # First column is batch index
        batch_size = x.batch_size
        
        latents = []
        for b in range(batch_size):
            mask = batch_indices == b
            batch_feat = feat[mask].mean(dim=0)
            latents.append(batch_feat)
        latent = torch.stack(latents, dim=0)
        latent = self.latent_proj(latent)
        return latent


class SparseDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, base_ch: int = 32, out_channels: int = 8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4 * base_ch * 32 * 32)
        
        self.deconv1 = nn.ConvTranspose2d(4 * base_ch, 2 * base_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * base_ch)
        
        self.deconv2 = nn.ConvTranspose2d(2 * base_ch, base_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_ch)
        
        self.conv_final = nn.Conv2d(base_ch, out_channels, 3, padding=1, bias=True)

    def forward(self, latent: torch.Tensor, spatial_shape: list):
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        B = latent.shape[0]
        x = self.fc(latent)
        x = x.view(B, -1, 32, 32)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv_final(x)
        return x[:, :, :spatial_shape[0], :spatial_shape[1]]


class SparseAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 8, base_ch: int = 32, latent_dim: int = 256):
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_ch, latent_dim)
        self.decoder = SparseDecoder(latent_dim, base_ch, in_channels)

    def forward(self, x: SparseConvTensor):
        latent = self.encoder(x)
        recon = self.decoder(latent, x.spatial_shape)
        return recon, latent


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

def train_autoencoder(model, loader, optimizer, n_epochs=20, device=device):
    model.train()
    criterion = nn.MSELoss()
    history = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"AE Epoch {epoch+1}/{n_epochs}", leave=False):
            x = batch.to(device)
            sparse_x = dense_to_sparse(x)
            optimizer.zero_grad()
            recon, _ = model(sparse_x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")
    return history


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


# ============================================================================
# MAIN TRAINING
# ============================================================================

if __name__ == "__main__":
    IN_CH = 8
    BASE_CH = 32
    LATENT_DIM = 256
    AE_EPOCHS = 30
    CLS_EPOCHS = 30
    BATCH_SIZE = 32
    MAX_SAMPLES = 100000  # Limit to avoid loading entire 28GB file

    print("\n=== Loading Data ===")
    unlabelled_ds = UnlabelledJetDataset("/data/b23_chiranjeevi/EDE/Dataset_Specific_Unlabelled.h5", max_samples=MAX_SAMPLES)
    labelled_ds = LabelledJetDataset("/data/b23_chiranjeevi/EDE/Dataset_Specific_labelled.h5", max_samples=MAX_SAMPLES)
    
    N = len(labelled_ds)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        labelled_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    ae_train_loader = DataLoader(unlabelled_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ae_val_loader = DataLoader(unlabelled_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    cls_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    cls_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("\n=== PHASE 1: Autoencoder Pretraining ===")
    ae_model = SparseAutoencoder(in_channels=IN_CH, base_ch=BASE_CH, latent_dim=LATENT_DIM).to(device)
    ae_opt = torch.optim.Adam(ae_model.parameters(), lr=1e-3, weight_decay=1e-5)
    ae_history = train_autoencoder(ae_model, ae_train_loader, ae_opt, n_epochs=AE_EPOCHS, device=device)
    torch.save(ae_model.state_dict(), "sparse_ae.pth")
    print("✓ Saved sparse_ae.pth")

    print("\n=== PHASE 2: Classifier Fine-tuning ===")
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
        pruned = copy.deepcopy(cls_model)
        if ratio > 0:
            apply_pruning(pruned, amount=ratio)
        
        pruned.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                sparse_x = dense_to_sparse(x)
                logits = pruned(sparse_x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        prune_accs.append(acc)
        print(f"Pruning {ratio:.0%}: Accuracy = {acc:.4f}")

    print("\n✓ Training Complete!")
    
    plt.figure(figsize=(10, 5))
    plt.plot(prune_ratios, prune_accs, 'o-')
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Accuracy")
    plt.title("Pruning Analysis")
    plt.grid(True)
    plt.savefig("pruning_analysis.png", dpi=150)
    print("✓ Saved pruning_analysis.png")
