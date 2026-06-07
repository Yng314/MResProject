"""DenseNet encoder implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    print("Warning: torchxrayvision not installed. TorchXRayVisionDenseNetEncoder will not be available.")


class DenseNet121Encoder(nn.Module):
    """DenseNet121 encoder for image feature extraction"""
    
    def __init__(self, pretrained: bool = False):
        super().__init__()
        densenet = models.densenet121(pretrained=pretrained)
        
        # Remove the final classifier layer, keep features
        self.features = densenet.features
        self.feature_dim = 1024  # DenseNet121 feature dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, 1024)
        """
        features = self.features(x)  # (B, 1024, H', W')
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (B, 1024, 1, 1)
        features = features.flatten(1)  # (B, 1024)
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


class TorchXRayVisionDenseNetEncoder(nn.Module):
    """TorchXRayVision pretrained DenseNet encoder
    
    This encoder uses DenseNet models pretrained on chest X-ray datasets
    (NIH, CheXpert, MIMIC, etc.) which provides better domain-specific features.
    """
    
    def __init__(self, weights: str = "densenet121-res224-all"):
        """
        Args:
            weights: Model weights to use. Options:
                - "densenet121-res224-all": DenseNet121 trained on all datasets (224x224)
                - "densenet121-res512-all": DenseNet121 trained on all datasets (512x512)
                - See torchxrayvision documentation for more options
        """
        super().__init__()
        
        if not XRV_AVAILABLE:
            raise ImportError(
                "torchxrayvision is not installed. "
                "Install it with: pip install torchxrayvision"
            )
        
        # Load pretrained model
        self.model = xrv.models.DenseNet(weights=weights)
        self.feature_dim = 1024  # DenseNet121 feature dimension
        
        # Store pathologies for reference
        self.pathologies = self.model.pathologies
        
        print(f"Loaded TorchXRayVision DenseNet model: {weights}")
        print(f"Pretrained on {len(self.pathologies)} pathologies: {self.pathologies}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - Can be 1 or 3 channels
               If 3 channels (RGB), convert to grayscale as XRV expects single channel
        Returns:
            features: (B, 1024)
        """
        # Convert RGB to grayscale if needed (XRV expects 1 channel)
        if x.shape[1] == 3:
            # Simple RGB to grayscale conversion
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        # Extract features (not classification logits)
        # model.features returns [B, 1024, H, W], need to pool and flatten
        features = self.model.features(x)  # (B, 1024, H, W)
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # (B, 1024, 1, 1)
        features = features.flatten(1)  # (B, 1024)
        return features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


class ClassificationHead(nn.Module):
    """Multi-label classification head
    
    Supports two modes:
    1. Binary mode (num_classes_per_label=1): Each label outputs 1 logit for binary classification
       Output shape: [batch_size, num_labels]
    2. Multi-class mode (num_classes_per_label=4): Each label outputs 4 class logits
       Output shape: [batch_size, num_labels, num_classes_per_label]
    """
    
    def __init__(self, input_dim: int, num_classes_per_label: int, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.num_classes_per_label = num_classes_per_label
        
        # Output dimension depends on mode
        output_dim = num_labels * num_classes_per_label
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            Binary mode (num_classes_per_label=1): [B, num_labels]
            Multi-class mode: [B, num_labels, num_classes_per_label]
        """
        logits = self.fc(x)  # [B, num_labels * num_classes_per_label]
        
        if self.num_classes_per_label == 1:
            # Binary classification mode: output [B, num_labels]
            return logits.view(-1, self.num_labels)
        else:
            # Multi-class mode: output [B, num_labels, num_classes_per_label]
            return logits.view(-1, self.num_labels, self.num_classes_per_label)


class DenseNetClassifier(nn.Module):
    """DenseNet121 + Classification Head for WSL
    
    Supports two types of encoders:
    1. DenseNet121Encoder: Standard ImageNet pretrained DenseNet121
    2. TorchXRayVisionDenseNetEncoder: Chest X-ray pretrained DenseNet121 (recommended)
    """
    
    def __init__(
        self,
        num_labels: int,  # Number of labels (e.g., 14)
        num_classes_per_label: int = 4,  # Number of classes per label (negative, positive, uncertain, unlabeled)
        encoder_type: str = "densenet121",  # "densenet121" or "xrv"
        xrv_weights: str = "densenet121-res224-all",  # Only used if encoder_type="xrv"
        encoder_checkpoint: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        """
        Args:
            num_labels: Number of labels (e.g., 14 for MIMIC-CXR)
            num_classes_per_label: Number of classes per label (default: 4)
                - 0: negative (0.0)
                - 1: positive (1.0)
                - 2: uncertain (-1.0)
                - 3: unlabeled (null)
            encoder_type: Type of encoder to use
                - "densenet121": Standard ImageNet pretrained DenseNet121
                - "xrv": TorchXRayVision pretrained DenseNet121 on chest X-rays (recommended)
            xrv_weights: XRV model weights (only used if encoder_type="xrv")
                Options: "densenet121-res224-all", "densenet121-res512-all", etc.
            encoder_checkpoint: Path to load custom encoder weights
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        # Initialize encoder based on type
        if encoder_type == "xrv":
            self.encoder = TorchXRayVisionDenseNetEncoder(weights=xrv_weights)
            print(f"Using TorchXRayVision DenseNet encoder: {xrv_weights}")
        elif encoder_type == "densenet121":
            self.encoder = DenseNet121Encoder(pretrained=True)
            print("Using standard ImageNet pretrained DenseNet121")
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Must be 'densenet121' or 'xrv'"
            )
        
        # Load custom encoder checkpoint if provided
        if encoder_checkpoint:
            print(f"Loading encoder from {encoder_checkpoint}")
            state_dict = torch.load(encoder_checkpoint, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=False)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
        
        self.classifier = ClassificationHead(
            self.encoder.get_feature_dim(),
            num_classes_per_label,
            num_labels
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] images
        Returns:
            logits: [B, num_labels, num_classes_per_label]
        """
        features = self.encoder(x)  # [B, 1024]
        logits = self.classifier(features)  # [B, num_labels, num_classes_per_label]
        return logits
    
    def unfreeze_top_layers(self, num_layers: int = 10):
        """Unfreeze top N layers of encoder for fine-tuning"""
        # Handle different encoder types
        if isinstance(self.encoder, TorchXRayVisionDenseNetEncoder):
            # For TorchXRayVision DenseNet encoder, access features directly
            # self.encoder.model is xrv.models.DenseNet, which has .features attribute
            all_layers = list(self.encoder.model.features.children())
        elif isinstance(self.encoder, DenseNet121Encoder):
            # For standard DenseNet121 encoder
            all_layers = list(self.encoder.features.children())
        else:
            raise ValueError(f"Unknown encoder type: {type(self.encoder)}")
        
        # Unfreeze top layers
        for layer in all_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {num_layers} layers of encoder")

