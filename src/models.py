"""Neural network architectures for sparse autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv


class SparseResBlock(nn.Module):
    """Residual block using sparse convolutions."""
    
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
    """Sparse encoder that projects data to latent space."""
    
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
    """Decoder that reconstructs dense data from latent representation."""
    
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
    """Complete sparse autoencoder with encoder and decoder."""
    
    def __init__(self, in_channels: int = 8, base_ch: int = 32, latent_dim: int = 256):
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_ch, latent_dim)
        self.decoder = SparseDecoder(latent_dim, base_ch, in_channels)

    def forward(self, x: SparseConvTensor):
        latent = self.encoder(x)
        recon = self.decoder(latent, x.spatial_shape)
        return recon, latent


class SparseClassifier(nn.Module):
    """Classifier head using frozen encoder."""
    
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
