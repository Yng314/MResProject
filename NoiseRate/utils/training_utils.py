"""
Training utilities for k-fold cross-validation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray images with multi-label classification
    """
    
    def __init__(self, image_paths, labels, base_path, transform=None):
        """
        Args:
            image_paths: List of relative image paths
            labels: Binary label matrix (n_samples, n_classes)
            base_path: Base directory for images
            transform: Torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = torch.FloatTensor(labels)
        self.base_path = Path(base_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.base_path / self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Convert to numpy for transforms
        image = np.array(image, dtype=np.float32)
        
        # Ensure in range [0, 255]
        if image.max() <= 1.0:
            image = image * 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        image = image[np.newaxis, :, :]
        
        # Apply transforms
        if self.transform:
            # Convert to tensor first
            image = torch.from_numpy(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        
        labels = self.labels[idx]
        
        return image, labels


def get_train_transforms(config):
    """
    Get training data augmentation transforms
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms.Compose
    """
    aug_config = config['augmentation']['train']
    img_size = config['preprocessing']['image_size']
    mean = config['preprocessing']['normalize_mean']
    std = config['preprocessing']['normalize_std']
    
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((img_size, img_size)))
    
    # Random horizontal flip
    if 'horizontal_flip' in aug_config:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip'])
        )
    
    # Random rotation
    if 'rotation_degrees' in aug_config:
        transform_list.append(
            transforms.RandomRotation(degrees=aug_config['rotation_degrees'])
        )
    
    # Random affine (translation)
    if 'translate' in aug_config:
        translate = tuple(aug_config['translate'])
        transform_list.append(
            transforms.RandomAffine(degrees=0, translate=translate)
        )
    
    # Normalize
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """
    Get validation transforms (no augmentation)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms.Compose
    """
    img_size = config['preprocessing']['image_size']
    mean = config['preprocessing']['normalize_mean']
    std = config['preprocessing']['normalize_std']
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=mean, std=std)
    ])


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset
    
    Args:
        labels: Binary label matrix (n_samples, n_classes)
    
    Returns:
        Tensor of class weights
    """
    n_samples = labels.shape[0]
    n_positive = labels.sum(axis=0)
    n_negative = n_samples - n_positive
    
    # Avoid division by zero
    n_positive = np.maximum(n_positive, 1)
    n_negative = np.maximum(n_negative, 1)
    
    # pos_weight = n_negative / n_positive
    pos_weight = n_negative / n_positive
    
    return torch.FloatTensor(pos_weight)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        Average loss
    """
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Returns:
        Average loss, predictions, ground truth labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Get probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return epoch_loss, all_preds, all_labels
