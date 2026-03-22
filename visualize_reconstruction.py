#!/usr/bin/env python3
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# MODEL ARCHITECTURE
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
        return x


class SparseAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 8, base_ch: int = 32, latent_dim: int = 256):
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_ch, latent_dim)
        self.decoder = SparseDecoder(latent_dim, base_ch, in_channels)

    def forward(self, x: SparseConvTensor):
        latent = self.encoder(x)
        recon = self.decoder(latent, x.spatial_shape)
        return recon, latent


# ============================================================================
# UTILITIES
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    IN_CH = 8
    BASE_CH = 32
    LATENT_DIM = 256
    
    print("\n" + "="*60)
    print("SPARSE AUTOENCODER RECONSTRUCTION VISUALIZATION")
    print("="*60)

    # Load data
    print("\n[1/4] Loading test data...")
    h5_path = "/data/b23_chiranjeevi/EDE/Dataset_Specific_Unlabelled.h5"
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        key = keys[0]
        data = f[key][:100]
        data = np.transpose(data, (0, 3, 1, 2))
        test_data = torch.from_numpy(data).float()
    print(f"✓ Loaded {test_data.shape[0]} samples, shape: {test_data.shape}")

    # Load model
    print("\n[2/4] Loading trained model...")
    model = SparseAutoencoder(in_channels=IN_CH, base_ch=BASE_CH, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load("sparse_ae.pth", weights_only=False))
    model.eval()
    print("✓ Model loaded")

    # Inference
    print("\n[3/4] Running inference...")
    batch_size = 8
    reconstructions = []
    latents_list = []
    
    with torch.no_grad():
        for i in range(0, test_data.shape[0], batch_size):
            batch = test_data[i:i+batch_size].to(device)
            sparse_batch = dense_to_sparse(batch)
            recon, latent = model(sparse_batch)
            reconstructions.append(recon.cpu())
            latents_list.append(latent.cpu())

    recon_data = torch.cat(reconstructions, dim=0)
    latents = torch.cat(latents_list, dim=0)
    print("✓ Inference complete")

    # Metrics
    print("\n[4/4] Computing metrics...")
    recon_data_cropped = recon_data[:, :, :125, :125]  # Crop to match input size
    
    mse_per_sample = torch.mean((test_data - recon_data_cropped) ** 2, dim=(1, 2, 3))
    mae_per_sample = torch.mean(torch.abs(test_data - recon_data_cropped), dim=(1, 2, 3))
    
    overall_mse = torch.mean(mse_per_sample).item()
    overall_mae = torch.mean(mae_per_sample).item()
    overall_rmse = torch.sqrt(torch.tensor(overall_mse)).item()
    
    original_sparsity = (test_data == 0).sum().item() / test_data.numel()
    recon_sparsity = (recon_data_cropped == 0).sum().item() / recon_data_cropped.numel()
    
    print("✓ Metrics computed")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nRECONSTRUCTION QUALITY:")
    print(f"  MSE:  {overall_mse:.6f}")
    print(f"  MAE:  {overall_mae:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")
    print(f"\nERROR STATISTICS:")
    print(f"  Min MSE:  {mse_per_sample.min():.6f}")
    print(f"  Max MSE:  {mse_per_sample.max():.6f}")
    print(f"  Std MSE:  {mse_per_sample.std():.6f}")
    print(f"\nSPARSITY:")
    print(f"  Original:      {original_sparsity:.2%}")
    print(f"  Reconstructed: {recon_sparsity:.2%}")
    print(f"\nLATENT SPACE:")
    print(f"  Mean: {latents.mean().item():.6f}")
    print(f"  Std:  {latents.std().item():.6f}")
    print(f"  Min:  {latents.min().item():.6f}")
    print(f"  Max:  {latents.max().item():.6f}")
    
    # Visualizations
    print(f"\nGenerating visualizations...")
    
    # Figure 1: Reconstruction samples
    n_samples = min(12, test_data.shape[0])
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(n_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(n_samples):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(test_data[i, 0], cmap='viridis')
        ax1.set_title(f'Original (sample {i+1})', fontsize=9)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(recon_data_cropped[i, 0], cmap='viridis')
        ax2.set_title(f'Reconstructed', fontsize=9)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[i, 2])
        diff = torch.abs(test_data[i, 0] - recon_data_cropped[i, 0])
        ax3.imshow(diff, cmap='hot')
        ax3.set_title(f'Error (MAE: {mae_per_sample[i]:.4f})', fontsize=9)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[i, 3])
        orig_energy = test_data[i].sum(dim=(1, 2)).numpy()
        recon_energy = recon_data_cropped[i].sum(dim=(1, 2)).numpy()
        channels = np.arange(len(orig_energy))
        ax4.bar(channels - 0.2, orig_energy, 0.4, label='Orig', alpha=0.7)
        ax4.bar(channels + 0.2, recon_energy, 0.4, label='Recon', alpha=0.7)
        ax4.legend(fontsize=7)
        ax4.set_ylabel('Energy', fontsize=8)
        ax4.tick_params(labelsize=6)
    
    fig.suptitle('Sparse Autoencoder Reconstruction Visualization', fontsize=14)
    plt.savefig('reconstruction_samples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved reconstruction_samples.png")
    
    # Figure 2: Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(mse_per_sample.numpy(), bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(overall_mse, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mse:.6f}')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('MSE Distribution', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mae_per_sample.numpy(), bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(overall_mae, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mae:.6f}')
    axes[1].set_xlabel('MAE', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('MAE Distribution', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved error_distribution.png")
    
    # Figure 3: Latent space
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents.numpy())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        scatter = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                                 c=mse_per_sample.numpy(), cmap='viridis', s=50, alpha=0.6)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        axes[0].set_title('Latent Space (PCA) colored by MSE', fontsize=12)
        plt.colorbar(scatter, ax=axes[0], label='MSE')
        
        axes[1].hist(latents.mean(dim=1).numpy(), bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Mean Latent Value', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Latent Vector Statistics (dim={LATENT_DIM})', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('latent_space.png', dpi=150, bbox_inches='tight')
        print("✓ Saved latent_space.png")
    except ImportError:
        print("⚠ sklearn not available, skipping latent space visualization")
    
    print("\n" + "="*60)
    print("✓ VISUALIZATION COMPLETE")
    print("="*60)
