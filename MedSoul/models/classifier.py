"""
Multi-label classifier for medical image diagnosis
"""
import torch
import torch.nn as nn
from typing import Optional


class MultiLabelClassifier(nn.Module):
    """Multi-label classification head"""
    
    def __init__(self, 
                 encoder: nn.Module,
                 num_classes: int = 12,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.5):
        """
        Args:
            encoder: Feature encoder (ResNet50)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension (None for single linear layer)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        
        feature_dim = encoder.feature_dim
        
        # Classification head
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] images
            
        Returns:
            logits: [B, num_classes] logits
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def freeze_encoder(self):
        """Freeze encoder (for linear probe stage)"""
        self.encoder.freeze()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder (for fine-tuning stage)"""
        self.encoder.unfreeze()
    
    def partial_unfreeze_encoder(self, n_layers: int = 2):
        """Partially unfreeze encoder top layers"""
        self.encoder.freeze_except_top_n_layers(n_layers)


class MultiLabelBCELoss(nn.Module):
    """Binary Cross-Entropy loss for multi-label binary classification
    
    Maps labels as follows:
    - 1.0 (positive) -> 1
    - -1.0 (uncertain) -> 1  [NEW: treat uncertain as positive]
    - 0.0 (negative) -> 0
    - NaN (missing) -> masked out
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            pos_weight: Weight for positive class (for imbalanced data)
                       Set to None to disable class weighting
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, label_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [B, num_labels] predicted logits
            targets: [B, num_labels] target labels (1.0, 0.0, -1.0, or NaN)
            label_mask: [B, num_labels] optional mask for valid labels (True = valid, False = masked)
            
        Returns:
            loss: Scalar loss
        """
        # Binary label mapping:
        # - positive (1.0) -> 1
        # - uncertain (-1.0) -> 1  [treat uncertain as positive]
        # - negative (0.0) -> 0
        # - missing (NaN) -> masked out
        
        # Create mask for valid labels
        if label_mask is not None:
            # Use provided mask
            valid_mask = label_mask
        else:
            # Create mask: valid if not NaN
            valid_mask = ~torch.isnan(targets)
        
        # Convert targets to binary (0 or 1)
        # Map: 1.0 -> 1, -1.0 -> 1, 0.0 -> 0, NaN -> 0 (will be masked anyway)
        binary_targets = torch.zeros_like(targets)
        binary_targets[(targets == 1.0) | (targets == -1.0)] = 1.0
        binary_targets[targets == 0.0] = 0.0
        
        # Compute BCE loss with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, 
            binary_targets, 
            reduction='none',
            pos_weight=self.pos_weight
        )
        
        # Apply mask (only compute loss on valid labels)
        masked_loss = bce_loss * valid_mask.float()
        
        # Average over valid entries
        num_valid = valid_mask.float().sum()
        loss = masked_loss.sum() / (num_valid + 1e-8)
        
        return loss


class MultiLabelMultiClassLoss(nn.Module):
    """Cross-Entropy loss for multi-label multi-class classification
    
    Each label independently performs multi-class classification:
    - Class 0: negative (0.0)
    - Class 1: positive (1.0)
    - Class 2: uncertain (-1.0)
    - Class 3: unlabeled (null/NaN)
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            class_weights: Optional tensor of shape [num_classes] with weights for each class.
                          If None, uses equal weights (standard cross-entropy).
                          Useful for handling class imbalance (e.g., [2.0, 1.0, 1.0, 0.5] to 
                          give more weight to class 0 which is underrepresented).
        """
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_labels, num_classes_per_label] predicted logits
            targets: [B, num_labels] target labels (0.0, 1.0, -1.0, or NaN)
            
        Returns:
            loss: Scalar loss
        """
        B, num_labels, num_classes = logits.shape
        
        # Convert labels to class indices
        # 0.0 -> 0 (negative)
        # 1.0 -> 1 (positive)
        # -1.0 -> 2 (uncertain)
        # NaN -> 3 (unlabeled)
        target_classes = torch.zeros_like(targets, dtype=torch.long)
        target_classes[targets == 0.0] = 0
        target_classes[targets == 1.0] = 1
        target_classes[targets == -1.0] = 2
        target_classes[torch.isnan(targets)] = 3  # null -> class 3
        
        # Compute cross-entropy loss for each label independently
        total_loss = 0.0
        valid_count = 0
        
        for label_idx in range(num_labels):
            label_logits = logits[:, label_idx, :]  # [B, num_classes]
            label_targets = target_classes[:, label_idx]  # [B]
            
            # All samples participate in training (including unlabeled class)
            # Use class weights if provided
            if self.class_weights is not None:
                # Move weights to same device as logits
                weights = self.class_weights.to(logits.device)
            else:
                weights = None
            
            label_loss = nn.functional.cross_entropy(
                label_logits,
                label_targets,
                reduction='mean',
                weight=weights
            )
            total_loss += label_loss
            valid_count += 1
        
        # Average loss across all labels
        return total_loss / (valid_count + 1e-8) if valid_count > 0 else torch.tensor(0.0, device=logits.device)
