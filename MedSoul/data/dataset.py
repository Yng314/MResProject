"""Dataset classes for MedSoul pipeline"""
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import gzip


class MIMICDataset(Dataset):
    """MIMIC-CXR Dataset for loading images and reports"""
    
    def __init__(
        self,
        parquet_paths: List[str],
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        pseudo_labels: Optional[Dict] = None,
        label_names: Optional[List[str]] = None
    ):
        """
        Args:
            parquet_paths: List of paths to parquet files
            indices: Subset indices to use (for train/val/test split)
            transform: Image transformations
            pseudo_labels: Dictionary mapping index to labels (from LLM)
            label_names: List of label names in order
        """
        # Load all parquet files
        dfs = []
        for path in parquet_paths:
            if Path(path).exists():
                dfs.append(pd.read_parquet(path))
        
        if not dfs:
            raise ValueError(f"No valid parquet files found in {parquet_paths}")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Use subset if indices provided
        self.original_indices = None
        if indices is not None:
            self.original_indices = indices  # Keep mapping: new_idx -> original_idx
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.transform = transform
        self.pseudo_labels = pseudo_labels
        self.label_names = label_names or []
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Map to original index if we're using a subset
        original_idx = self.original_indices[idx] if self.original_indices is not None else idx
        
        # Load image from bytes
        img_bytes = row['image']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get report text (prefer impression over findings)
        impression = row.get('impression', '')
        findings = row.get('findings', '')
        report = impression if pd.notna(impression) and impression.strip() else findings
        
        item = {
            'image': image,
            'report': report if pd.notna(report) else '',
            'idx': original_idx  # Use original index
        }
        
        # Add pseudo labels if available (use original index)
        if self.pseudo_labels is not None and str(original_idx) in self.pseudo_labels:
            labels = self.pseudo_labels[str(original_idx)]
            # Convert label dict to tensor
            label_tensor = torch.zeros(len(self.label_names), dtype=torch.float32)
            label_mask = torch.zeros(len(self.label_names), dtype=torch.bool)  # NEW: mask for valid labels
            
            for i, label_name in enumerate(self.label_names):
                if label_name in labels:
                    val = labels[label_name]
                    if val == 1.0:
                        label_tensor[i] = 1.0  # Class 1: positive
                        label_mask[i] = True
                    elif val == 0.0:
                        label_tensor[i] = 0.0  # Class 0: negative
                        label_mask[i] = True
                    elif val == -1.0:
                        label_tensor[i] = -1.0  # Class 2: uncertain
                        label_mask[i] = True
                    # None/null: set to NaN, but mark as valid (Class 3: unlabeled)
                    elif val is None:
                        label_tensor[i] = float('nan')  # Class 3: unlabeled
                        label_mask[i] = True  # null also participates in training
                else:
                    # If label doesn't exist, treat as null (unlabeled)
                    label_tensor[i] = float('nan')
                    label_mask[i] = True
            
            item['labels'] = label_tensor
            item['label_mask'] = label_mask  # NEW: indicates which labels are valid
        
        return item


class MAEDataset(Dataset):
    """Dataset for MAE pretraining (only images)"""
    
    def __init__(
        self,
        parquet_paths: List[str],
        indices: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None
    ):
        dfs = []
        for path in parquet_paths:
            if Path(path).exists():
                dfs.append(pd.read_parquet(path))
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_bytes = self.df.iloc[idx]['image']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transforms(image_size: int = 224, is_train: bool = True, skip_resize: bool = False) -> transforms.Compose:
    """
    Get image transformations
    
    Args:
        image_size: Target image size (only used if skip_resize=False)
        is_train: Whether to apply training augmentations
        skip_resize: If True, skip resize (useful for pre-resized images)
    """
    if is_train:
        transform_list = []
        if not skip_resize:
            transform_list.append(transforms.Resize((image_size, image_size)))
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(10),  # 注释掉最耗时的旋转操作
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 注释掉颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(transform_list)
    else:
        transform_list = []
        if not skip_resize:
            transform_list.append(transforms.Resize((image_size, image_size)))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(transform_list)


def split_dataset(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices into train/val/test"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()
    
    return train_indices, val_indices, test_indices


def load_test_set_for_validation(test_set_json, val_ratio=0.15, seed=42):
    """
    Load validation set from test_set_2_1_0.json
    
    Args:
        test_set_json: Path to test_set_2_1_0.json
        val_ratio: Ratio of test set to use for validation (default: 0.15 = 15%)
        seed: Random seed for splitting
    
    Returns:
        val_samples: List of samples for validation
        test_samples: List of samples for final test
    """
    from sklearn.model_selection import train_test_split
    
    with open(test_set_json, 'r') as f:
        all_samples = json.load(f)
    
    # Split test set into validation and test
    val_samples, test_samples = train_test_split(
        all_samples,
        test_size=1 - val_ratio,  # test_size is the remaining portion
        random_state=seed
    )
    
    print(f"Split test set: {len(val_samples)} for validation, {len(test_samples)} for final test")
    
    return val_samples, test_samples


class TestSetDataset(Dataset):
    """Dataset for loading from test_set_2_1_0.json (for validation or test)"""
    
    def __init__(
        self,
        test_samples: List[Dict],
        image_dir: str = 'datasets/test_set',
        transform: Optional[transforms.Compose] = None,
        label_names: Optional[List[str]] = None,
        use_ground_truth: bool = True
    ):
        """
        Args:
            test_samples: List of sample dicts from test_set_2_1_0.json
            image_dir: Directory containing test images
            transform: Image transformations
            label_names: List of label names in order
            use_ground_truth: If True, use ground truth labels; if False, expect pseudo_labels in samples
        """
        self.test_samples = test_samples
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_names = label_names or []
        self.use_ground_truth = use_ground_truth
    
    def __len__(self) -> int:
        return len(self.test_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.test_samples[idx]
        study_id = sample['study_id']
        images_info = sample['images']
        
        # Use the first image if multiple exist
        img_info = images_info[0]
        dicom_id = img_info['dicom_id']
        
        # Construct image filename
        image_filename = f"{study_id}_{dicom_id}.png"
        image_path = self.image_dir / image_filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        if self.use_ground_truth:
            labels = sample['labels']
        else:
            labels = sample.get('pseudo_labels', {})
        
        # Convert labels to tensor
        label_tensor = torch.zeros(len(self.label_names), dtype=torch.float32)
        label_mask = torch.zeros(len(self.label_names), dtype=torch.bool)
        
        for i, label_name in enumerate(self.label_names):
            val = labels.get(label_name)
            if val == 1.0:
                label_tensor[i] = 1.0
                label_mask[i] = True
            elif val == 0.0:
                label_tensor[i] = 0.0
                label_mask[i] = True
            elif val == -1.0:
                label_tensor[i] = -1.0
                label_mask[i] = True
            elif val is None:
                label_tensor[i] = float('nan')
                label_mask[i] = True
        
        return {
            'image': image,
            'labels': label_tensor,
            'label_mask': label_mask,
            'study_id': study_id,
            'idx': idx
        }


class MIMICCXRJPGDataset(Dataset):
    """
    MIMIC-CXR-JPG Dataset loader
    Loads images from JPG files and labels from CheXpert CSV
    """
    
    def __init__(
        self,
        image_root: str,
        chexpert_csv: str,
        split_csv: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        label_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            image_root: Root directory containing patient folders (e.g., datasets/mimic-cxr-jpg-2.1.0/files)
            chexpert_csv: Path to CheXpert labels CSV (can be .gz)
            split_csv: Path to split CSV (can be .gz)
            split: One of 'train', 'validate', 'test'
            transform: Image transformations
            label_names: List of label names in order (must match config)
            num_samples: Limit to first N samples (for debugging)
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_names = label_names or []
        
        # Load split file
        print(f"[INFO] Loading split file: {split_csv}")
        split_df = pd.read_csv(split_csv)
        split_df = split_df[split_df['split'] == split].reset_index(drop=True)
        print(f"[INFO] Found {len(split_df)} samples in {split} split")
        
        # Load CheXpert labels
        print(f"[INFO] Loading CheXpert labels: {chexpert_csv}")
        labels_df = pd.read_csv(chexpert_csv)
        print(f"[INFO] Found {len(labels_df)} labeled studies")
        
        # Merge split with labels
        self.df = split_df.merge(labels_df, on=['subject_id', 'study_id'], how='left')
        print(f"[INFO] Merged dataset size: {len(self.df)}")
        
        # Limit samples if requested (before validation to speed up)
        if num_samples is not None and num_samples < len(self.df):
            self.df = self.df.iloc[:num_samples].reset_index(drop=True)
            print(f"[INFO] Limited to {num_samples} samples")
        
        # Validate that image files exist (filter out missing files)
        print(f"[INFO] Validating image files (this may take a few minutes for large datasets)...")
        from tqdm import tqdm
        valid_indices = []
        missing_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating files"):
            subject_id = int(row['subject_id'])
            study_id = int(row['study_id'])
            dicom_id = row['dicom_id']
            image_path = self._construct_image_path(subject_id, study_id, dicom_id)
            
            if image_path.exists():
                valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:  # Only print first 5 missing files
                    print(f"\n[WARN] Missing file: {image_path}")
        
        if missing_count > 0:
            print(f"[WARN] Found {missing_count} missing files out of {len(self.df)} ({missing_count/len(self.df)*100:.2f}%)")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"[INFO] Filtered dataset size: {len(self.df)} samples")
        else:
            print(f"[INFO] All {len(self.df)} files validated successfully")
        
        # Get available label columns from CSV
        self.csv_label_columns = [col for col in labels_df.columns 
                                   if col not in ['subject_id', 'study_id']]
        print(f"[INFO] Available label columns in CSV: {self.csv_label_columns}")
        
        # Label name mapping (config name -> CSV name)
        # Handle difference between test set ("Airspace Opacity") and CheXpert CSV ("Lung Opacity")
        self.label_mapping = {
            "Airspace Opacity": "Lung Opacity"  # Map config name to CSV name
        }
        
        # Check if label_names match CSV columns (with mapping)
        if self.label_names:
            unmapped_labels = []
            for label in self.label_names:
                csv_label = self.label_mapping.get(label, label)
                if csv_label not in self.csv_label_columns:
                    unmapped_labels.append(label)
            
            if unmapped_labels:
                print(f"[WARN] Labels in config not found in CSV: {unmapped_labels}")
                print(f"[WARN] Will use NaN for missing labels")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _construct_image_path(self, subject_id: int, study_id: int, dicom_id: str) -> Path:
        """
        Construct image path following MIMIC-CXR structure:
        files/p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        
        Example: subject_id=10000032 -> p10/p10000032/s50414267/xxx.jpg
        """
        subject_prefix = str(subject_id)[:2]  # First 2 digits (e.g., 10 from 10000032)
        image_path = (self.image_root / 
                      f'p{subject_prefix}' / 
                      f'p{subject_id}' / 
                      f's{study_id}' / 
                      f'{dicom_id}.jpg')
        return image_path
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Get image path
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        dicom_id = row['dicom_id']
        
        image_path = self._construct_image_path(subject_id, study_id, dicom_id)
        
        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract labels
        # CheXpert uses: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), NaN (not mentioned)
        label_tensor = torch.zeros(len(self.label_names), dtype=torch.float32)
        label_mask = torch.zeros(len(self.label_names), dtype=torch.bool)
        
        for i, label_name in enumerate(self.label_names):
            # Map config label name to CSV label name (handles Airspace Opacity -> Lung Opacity)
            csv_label_name = self.label_mapping.get(label_name, label_name)
            
            if csv_label_name in self.csv_label_columns:
                val = row.get(csv_label_name)
                if pd.notna(val):
                    label_tensor[i] = float(val)
                    label_mask[i] = True
                else:
                    label_tensor[i] = float('nan')
                    label_mask[i] = False
            else:
                # Label not in CSV - mark as unavailable
                label_tensor[i] = float('nan')
                label_mask[i] = False
        
        return {
            'image': image,
            'labels': label_tensor,
            'label_mask': label_mask,
            'subject_id': subject_id,
            'study_id': study_id,
            'dicom_id': dicom_id,
            'idx': idx
        }


class MIMICCXRJPGDatasetMAE(Dataset):
    """
    MIMIC-CXR-JPG Dataset for MAE pretraining (images only, no labels)
    """
    
    def __init__(
        self,
        image_root: str,
        split_csv: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            image_root: Root directory containing patient folders
            split_csv: Path to split CSV (can be .gz)
            split: One of 'train', 'validate', 'test'
            transform: Image transformations
            num_samples: Limit to first N samples (for debugging)
        """
        self.image_root = Path(image_root)
        self.transform = transform
        
        # Load split file
        print(f"[INFO] Loading split file for MAE: {split_csv}")
        split_df = pd.read_csv(split_csv)
        self.df = split_df[split_df['split'] == split].reset_index(drop=True)
        print(f"[INFO] Found {len(self.df)} samples in {split} split")
        
        # Limit samples if requested
        if num_samples is not None and num_samples < len(self.df):
            self.df = self.df.iloc[:num_samples].reset_index(drop=True)
            print(f"[INFO] Limited to {num_samples} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _construct_image_path(self, subject_id: int, study_id: int, dicom_id: str) -> Path:
        """Construct image path following MIMIC-CXR structure"""
        subject_prefix = str(subject_id)[:2]  # First 2 digits
        image_path = (self.image_root / 
                      f'p{subject_prefix}' / 
                      f'p{subject_id}' / 
                      f's{study_id}' / 
                      f'{dicom_id}.jpg')
        return image_path
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        
        # Get image path
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        dicom_id = row['dicom_id']
        
        image_path = self._construct_image_path(subject_id, study_id, dicom_id)
        
        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class MIMICCXRTestSet(Dataset):
    """
    MIMIC-CXR-JPG 2.1.0 Official Test Set Dataset
    Loads from mimic-cxr-2.1.0-test-set-labeled.csv
    """
    
    def __init__(
        self,
        image_root: str,
        test_csv: str,
        split_csv: str,
        transform: Optional[transforms.Compose] = None,
        label_names: Optional[List[str]] = None
    ):
        """
        Args:
            image_root: Root directory containing patient folders
            test_csv: Path to test set labels CSV (mimic-cxr-2.1.0-test-set-labeled.csv)
            split_csv: Path to split CSV (to get dicom_id and subject_id)
            transform: Image transformations
            label_names: List of label names (must match CSV column names)
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_names = label_names or []
        
        # Load test set labels
        print(f"[INFO] Loading test set labels: {test_csv}")
        test_df = pd.read_csv(test_csv)
        print(f"[INFO] Found {len(test_df)} test studies")
        
        # Load split file to get dicom_id and subject_id
        print(f"[INFO] Loading split file: {split_csv}")
        split_df = pd.read_csv(split_csv)
        
        # Filter split file to only test set studies
        test_study_ids = test_df['study_id'].unique()
        split_test = split_df[split_df['study_id'].isin(test_study_ids)].copy()
        print(f"[INFO] Found {len(split_test)} test images from {len(test_study_ids)} studies")
        
        # Merge test labels with split info
        self.df = split_test.merge(test_df, on='study_id', how='left')
        print(f"[INFO] Merged test dataset size: {len(self.df)} images")
        
        # Validate that image files exist (filter out missing files)
        print(f"[INFO] Validating test image files...")
        from tqdm import tqdm
        valid_indices = []
        missing_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating test files"):
            subject_id = int(row['subject_id'])
            study_id = int(row['study_id'])
            dicom_id = row['dicom_id']
            image_path = self._construct_image_path(subject_id, study_id, dicom_id)
            
            if image_path.exists():
                valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"\n[WARN] Missing test file: {image_path}")
        
        if missing_count > 0:
            print(f"[WARN] Found {missing_count} missing test files out of {len(self.df)} ({missing_count/len(self.df)*100:.2f}%)")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"[INFO] Filtered test dataset size: {len(self.df)} images")
        else:
            print(f"[INFO] All {len(self.df)} test files validated successfully")
        
        # Get available label columns from test CSV
        self.csv_label_columns = [col for col in test_df.columns if col != 'study_id']
        print(f"[INFO] Available label columns: {self.csv_label_columns}")
        
        # Check label name matching
        if self.label_names:
            missing_labels = set(self.label_names) - set(self.csv_label_columns)
            if missing_labels:
                print(f"[WARN] Labels in config not found in test CSV: {missing_labels}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _construct_image_path(self, subject_id: int, study_id: int, dicom_id: str) -> Path:
        """Construct image path following MIMIC-CXR structure"""
        subject_prefix = str(subject_id)[:2]  # First 2 digits
        image_path = (self.image_root / 
                      f'p{subject_prefix}' / 
                      f'p{subject_id}' / 
                      f's{study_id}' / 
                      f'{dicom_id}.jpg')
        return image_path
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Get image info
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        dicom_id = row['dicom_id']
        
        image_path = self._construct_image_path(subject_id, study_id, dicom_id)
        
        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract labels (ground truth)
        label_tensor = torch.zeros(len(self.label_names), dtype=torch.float32)
        label_mask = torch.zeros(len(self.label_names), dtype=torch.bool)
        
        for i, label_name in enumerate(self.label_names):
            if label_name in self.csv_label_columns:
                val = row.get(label_name)
                if pd.notna(val):
                    label_tensor[i] = float(val)
                    label_mask[i] = True
                else:
                    label_tensor[i] = float('nan')
                    label_mask[i] = False
            else:
                # Label not in CSV
                label_tensor[i] = float('nan')
                label_mask[i] = False
        
        return {
            'image': image,
            'labels': label_tensor,
            'label_mask': label_mask,
            'subject_id': subject_id,
            'study_id': study_id,
            'dicom_id': dicom_id,
            'idx': idx
        }