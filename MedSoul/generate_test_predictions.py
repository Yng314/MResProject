"""
Generate predictions on test set for noise rate estimation

Usage:
    python generate_test_predictions.py --config slurm_jobs/config_slurm.yaml
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm


class MIMICTestDataset(Dataset):
    """Dataset for MIMIC-CXR test set"""
    
    def __init__(self, image_root, test_csv, split_csv, label_names, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_names = label_names
        
        # Load test set labels
        test_df = pd.read_csv(test_csv)
        print(f"[INFO] Loaded test set: {len(test_df)} samples")
        
        # Load split CSV to get image paths
        if split_csv.endswith('.gz'):
            split_df = pd.read_csv(split_csv, compression='gzip')
        else:
            split_df = pd.read_csv(split_csv)
        
        # Filter test split
        split_df = split_df[split_df['split'] == 'test']
        print(f"[INFO] Test split samples: {len(split_df)}")
        
        # Merge to get labels and paths
        self.df = split_df.merge(test_df, on='study_id', how='inner')
        print(f"[INFO] Matched samples: {len(self.df)}")
        
        # Map label names (handle Airspace Opacity vs Lung Opacity)
        self.label_map = {}
        for name in label_names:
            if name in test_df.columns:
                self.label_map[name] = name
            elif name == "Airspace Opacity" and "Lung Opacity" in test_df.columns:
                self.label_map[name] = "Lung Opacity"
            elif name == "Lung Opacity" and "Airspace Opacity" in test_df.columns:
                self.label_map[name] = "Airspace Opacity"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build image path
        subject_id = f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}"
        study_id = f"s{row['study_id']}"
        dicom_id = row['dicom_id']
        img_path = self.image_root / subject_id / study_id / f"{dicom_id}.jpg"
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = []
        for name in self.label_names:
            col = self.label_map.get(name, name)
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    labels.append(float('nan'))
                else:
                    labels.append(float(val))
            else:
                labels.append(float('nan'))
        
        return image, np.array(labels, dtype=np.float32), idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='slurm_jobs/config_slurm.yaml')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find best model checkpoint
    output_dir = config.get('pipeline', {}).get('output_dir', 'outputs')
    checkpoint_dir = os.path.join(output_dir, config['wsl']['checkpoint_dir'])
    
    # Try to find the best checkpoint
    checkpoint_path = None
    for name in ['fine_tune_best.pth', 'linear_probe_best.pth']:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model using the same function as train_wsl.py
    import sys
    sys.path.insert(0, os.getcwd())
    from train_wsl import create_model
    
    # Override config to match the checkpoint (model was trained with multi_class)
    # This is needed because the checkpoint was trained with 4-class output
    original_loss_type = config['wsl']['loss'].get('type', 'multi_class')
    config['wsl']['loss']['type'] = 'multi_class'  # Force multi_class to match checkpoint
    
    model = create_model(config, freeze_encoder=True, device=device)
    
    # Restore original config
    config['wsl']['loss']['type'] = original_loss_type
    
    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test dataset
    mimic_cfg = config['data']['mimic_jpg']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MIMICTestDataset(
        image_root=mimic_cfg['image_root'],
        test_csv='datasets/mimic-cxr-2.1.0-test-set-labeled.csv',
        split_csv=mimic_cfg['split_csv'],
        label_names=config['data']['labels'],
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    # Generate predictions
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for images, labels, indices in tqdm(test_loader, desc="Generating predictions"):
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.append(indices.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save predictions
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(checkpoint_dir, 'test_predictions.npz')
    
    np.savez(output_path, preds=preds, labels=labels, indices=indices)
    print(f"Saved predictions to: {output_path}")


if __name__ == '__main__':
    main()
