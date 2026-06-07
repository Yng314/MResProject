"""Masked Autoencoder (MAE) implementation for medical images"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from .resnet import ResNet50Encoder


class PatchEmbed(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size: int = 512, patch_size: int = 32, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MAEEncoder(nn.Module):
    """MAE Encoder using ResNet50 backbone"""
    
    def __init__(self, img_size: int = 512, patch_size: int = 32):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use ResNet50 as encoder
        self.resnet_encoder = ResNet50Encoder(pretrained=False)
        self.feature_dim = self.resnet_encoder.get_feature_dim()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) images
            mask: (B, num_patches) binary mask (1 = keep, 0 = remove)
        Returns:
            features: (B, feature_dim)
        """
        # Apply mask by zeroing out masked patches
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Reshape to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, C * p * p)
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = x * mask_expanded
        
        # Reshape back to image
        x = x.reshape(B, H // p, W // p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # Encode
        features = self.resnet_encoder(x)
        return features


class MAEDecoder(nn.Module):
    """Simple decoder to reconstruct images"""
    
    def __init__(self, feature_dim: int = 2048, img_size: int = 512, patch_size: int = 32):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Project features to patch representations
        patch_dim = 3 * patch_size * patch_size
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_patches * patch_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)
        Returns:
            reconstructed: (B, 3, H, W)
        """
        B = features.shape[0]
        p = self.patch_size
        
        # Decode
        x = self.decoder(features)  # (B, num_patches * patch_dim)
        x = x.reshape(B, self.num_patches, 3, p, p)
        
        # Reshape to image
        h = w = self.img_size // p
        x = x.reshape(B, h, w, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, self.img_size, self.img_size)
        
        return x


class MAE(nn.Module):
    """Masked Autoencoder for medical images"""
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 32,
        mask_ratio: float = 0.75
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        self.encoder = MAEEncoder(img_size, patch_size)
        self.decoder = MAEDecoder(self.encoder.feature_dim, img_size, patch_size)
    
    def random_masking(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random mask
        Returns:
            mask: (B, num_patches) binary mask (1 = keep, 0 = remove)
            ids_restore: indices to restore original order
        """
        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Generate binary mask: 1 is keep, 0 is remove
        mask = torch.zeros(batch_size, N, device=device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return mask, ids_restore
    
    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            imgs: (B, 3, H, W)
        Returns:
            reconstructed: (B, 3, H, W)
            mask: (B, num_patches)
            features: (B, feature_dim)
        """
        B = imgs.shape[0]
        device = imgs.device
        
        # Generate mask
        mask, _ = self.random_masking(B, device)
        
        # Encode
        features = self.encoder(imgs, mask)
        
        # Decode
        reconstructed = self.decoder(features)
        
        return reconstructed, mask, features
    
    def save_encoder(self, path: str):
        """Save encoder weights for WSL"""
        torch.save(self.encoder.resnet_encoder.state_dict(), path)
        print(f"Saved encoder to {path}")
