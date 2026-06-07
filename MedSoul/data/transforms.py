"""
Data augmentation and transforms for medical images
"""
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Tuple


class ImageTransforms:
    """Image transforms for training and validation"""
    
    def __init__(self, image_size: int = 512, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225), is_train: bool = True):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.is_train = is_train
        
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
    
    def __call__(self, image):
        return self.transform(image)


class MAETransforms:
    """Transforms specifically for MAE pretraining (minimal augmentation)"""
    
    def __init__(self, image_size: int = 512, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, image):
        return self.transform(image)

