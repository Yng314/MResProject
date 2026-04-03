"""
Lightweight CNN for Chest X-ray Multi-label Classification
Designed for small datasets (~676 samples)
Parameters: ~500K
"""

import torch
import torch.nn as nn


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for chest X-ray classification
    
    Architecture:
        - 4 convolutional blocks with increasing channels (32->64->128->256)
        - BatchNorm and ReLU activations
        - MaxPooling for downsampling
        - Global Average Pooling instead of FC layers
        - Single FC layer for classification
    
    Parameters: ~500K (suitable for small datasets)
    """
    
    def __init__(self, num_classes=12, input_channels=1, dropout=0.5):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1: 1 -> 32, 224 -> 112
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            
            # Conv Block 2: 32 -> 64, 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            
            # Conv Block 3: 64 -> 128, 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            
            # Conv Block 4: 128 -> 256, 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            
            # Global Average Pooling: 14x14 -> 1x1
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (N, C, H, W)
        
        Returns:
            Logits (N, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params, trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model
    model = LightweightCNN(num_classes=12)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"Model: LightweightCNN")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / (1024**2):.2f} MB (float32)")
    
    # Test forward pass
    x = torch.randn(4, 1, 224, 224)  # Batch of 4 grayscale 224x224 images
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (logits): {output[0]}")
