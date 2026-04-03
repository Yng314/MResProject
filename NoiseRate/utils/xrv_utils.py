"""
TorchXRayVision Model Utilities
Provides functions for model loading, image preprocessing, and prediction extraction
"""

import torch
import torchxrayvision as xrv
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# TorchXRayVision outputs 18 pathologies
XRV_PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
    'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
    'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
    'Lung Opacity', 'Enlarged Cardiomediastinum'
]

# CheXpert 14 pathologies (our target)
CHEXPERT_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]


def get_pathology_mapping() -> Dict[str, int]:
    """
    Create mapping from TorchXRayVision pathology indices to CheXpert pathology indices
    
    Maps 12 out of 14 CheXpert classes (excluding 'No Finding' and 'Support Devices')
    
    Returns:
        Dictionary mapping CheXpert pathology name to TorchXRayVision output index
    
    Example:
        >>> mapping = get_pathology_mapping()
        >>> mapping['Atelectasis']  # Returns 0 (index in XRV output)
        >>> len(mapping)  # Returns 12 (mappable pathologies)
    """
    mapping = {}
    
    # Direct matches
    direct_matches = {
        'Atelectasis': 'Atelectasis',
        'Consolidation': 'Consolidation',
        'Pneumothorax': 'Pneumothorax',
        'Edema': 'Edema',
        'Pneumonia': 'Pneumonia',
        'Cardiomegaly': 'Cardiomegaly',
        'Lung Lesion': 'Lung Lesion',
        'Fracture': 'Fracture',
        'Lung Opacity': 'Lung Opacity',
        'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
    }
    
    # Approximate matches (confirmed by user)
    approximate_matches = {
        'Pleural Effusion': 'Effusion',  # Effusion → Pleural Effusion
        'Pleural Other': 'Pleural_Thickening',  # Pleural_Thickening → Pleural Other
    }
    
    # Combine matches
    all_matches = {**direct_matches, **approximate_matches}
    
    # Create mapping to XRV indices
    for chexpert_name, xrv_name in all_matches.items():
        if xrv_name in XRV_PATHOLOGIES:
            xrv_idx = XRV_PATHOLOGIES.index(xrv_name)
            mapping[chexpert_name] = xrv_idx
        else:
            print(f"Warning: {xrv_name} not found in XRV pathologies")
    
    return mapping


def get_ordered_pathology_list() -> List[str]:
    """
    Get ordered list of CheXpert pathologies that can be predicted (12 classes)
    
    Returns:
        List of pathology names in consistent order
    """
    # Return only the 12 mappable pathologies in CheXpert order
    mappable = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax'
    ]
    return mappable


def load_pretrained_model(device: Optional[str] = None) -> torch.nn.Module:
    """
    Load TorchXRayVision pretrained DenseNet121 model
    
    Args:
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Loaded model in eval mode
    
    Example:
        >>> model = load_pretrained_model()
        >>> model.eval()
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading TorchXRayVision DenseNet121 model on {device}...")
    
    # Load pretrained model
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Output dimensions: {len(XRV_PATHOLOGIES)} pathologies")
    
    return model


class XRayDataset(Dataset):
    """
    Dataset for loading X-ray images
    """
    def __init__(self, image_paths: List[str], base_path: str = ""):
        """
        Args:
            image_paths: List of relative image paths
            base_path: Base directory path
        """
        self.image_paths = image_paths
        self.base_path = Path(base_path)
        
        # TorchXRayVision preprocessing
        self.transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.base_path / self.image_paths[idx]
        
        # Load image as grayscale
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        
        # Ensure image is in correct range [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Add channel dimension BEFORE transforms: (H, W) -> (1, H, W)
        # XRayCenterCrop expects (C, H, W) format
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        
        # Apply transforms (center crop and resize to 224x224)
        img = self.transform(img)
        
        # Normalize to model input range [-1024, 1024]
        img = xrv.datasets.normalize(img, maxval=255)
        
        # Ensure shape is (1, 224, 224)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        
        return {
            'image': torch.from_numpy(img).float(),
            'path': str(img_path)
        }


def create_dataloader(
    image_paths: List[str],
    base_path: str = "",
    batch_size: int = 32,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for X-ray images
    
    Args:
        image_paths: List of relative image paths
        base_path: Base directory path
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader
    
    Example:
        >>> dataloader = create_dataloader(image_paths, base_path, batch_size=16)
    """
    dataset = XRayDataset(image_paths, base_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader


def extract_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract predictions from model for all samples
    
    Args:
        model: TorchXRayVision model
        dataloader: DataLoader with images
        device: Device to run inference on
    
    Returns:
        Tuple of:
        - full_predictions: (n_samples, 18) array with all XRV predictions
        - mapped_predictions: (n_samples, 12) array with mapped CheXpert predictions
    
    Example:
        >>> full_preds, mapped_preds = extract_predictions(model, dataloader)
        >>> print(full_preds.shape)  # (676, 18)
        >>> print(mapped_preds.shape)  # (676, 12)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)  # Shape: (batch_size, 18)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} samples...")
    
    # Concatenate all predictions
    full_predictions = np.vstack(all_predictions)
    
    print(f"✅ Inference complete: {full_predictions.shape}")
    
    # Map to CheXpert 12 classes
    mapping = get_pathology_mapping()
    ordered_pathologies = get_ordered_pathology_list()
    
    mapped_predictions = np.zeros((full_predictions.shape[0], len(ordered_pathologies)))
    
    for i, pathology in enumerate(ordered_pathologies):
        if pathology in mapping:
            xrv_idx = mapping[pathology]
            mapped_predictions[:, i] = full_predictions[:, xrv_idx]
    
    print(f"✅ Mapped to CheXpert format: {mapped_predictions.shape}")
    
    return full_predictions, mapped_predictions


def validate_predictions(
    predictions: np.ndarray,
    pathology_names: Optional[List[str]] = None
) -> bool:
    """
    Validate prediction array
    
    Args:
        predictions: Prediction array to validate
        pathology_names: Optional list of pathology names for display
    
    Returns:
        True if valid
    """
    print(f"\nValidating predictions...")
    print(f"  Shape: {predictions.shape}")
    print(f"  Min value: {predictions.min():.4f}")
    print(f"  Max value: {predictions.max():.4f}")
    print(f"  Mean value: {predictions.mean():.4f}")
    
    # Check range
    if predictions.min() < 0 or predictions.max() > 1:
        print(f"⚠️  Warning: Predictions should be in [0, 1] range")
        return False
    
    # Check for NaN
    if np.isnan(predictions).any():
        print(f"❌ ERROR: Predictions contain NaN values")
        return False
    
    # Show per-class statistics
    if pathology_names is not None:
        print(f"\nPer-class statistics:")
        for i, name in enumerate(pathology_names):
            mean_prob = predictions[:, i].mean()
            print(f"  {name}: {mean_prob:.3f}")
    
    print(f"✅ Validation passed")
    return True
