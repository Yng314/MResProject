"""
Phase 3: Weakly Supervised Learning with pseudo labels
Two-stage training: Linear Probe + Fine-tune
"""
import os
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from data.dataset import MIMICDataset, get_transforms, split_dataset, TestSetDataset, load_test_set_for_validation, MIMICCXRJPGDataset, MIMICCXRTestSet
from models.resnet import ResNetClassifier
from models.densenet import DenseNetClassifier
from models.classifier import MultiLabelMultiClassLoss, MultiLabelBCELoss
from utils.metrics import compute_metrics, get_predictions


def compute_class_weights_from_dataset(dataset, label_names, device='cuda'):
    """
    Compute class weights from dataset object
    
    Args:
        dataset: Dataset object (MIMICDataset or MIMICCXRJPGDataset)
        label_names: List of label names
        device: Device to create weights tensor on
    
    Returns:
        class_weights: Tensor of shape [4] with weights for each class
    """
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("[INFO] Computing class weights from dataset...")
    from tqdm import tqdm
    for idx in tqdm(range(len(dataset)), desc="Analyzing class distribution"):
        item = dataset[idx]
        labels = item['labels']  # Tensor of shape [num_labels]
        
        for i, label_name in enumerate(label_names):
            val = labels[i].item()
            if val == 0.0:
                class_counts[0] += 1
            elif val == 1.0:
                class_counts[1] += 1
            elif val == -1.0:
                class_counts[2] += 1
            elif torch.isnan(torch.tensor(val)):
                class_counts[3] += 1
    
    total = sum(class_counts.values())
    if total == 0:
        return torch.ones(4, device=device)
    
    weights = []
    for cls in range(4):
        if class_counts[cls] > 0:
            weight = total / (4.0 * class_counts[cls])
        else:
            weight = total / (4.0 * 1.0)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum() * 4.0
    
    print(f"\n[INFO] Class distribution:")
    print(f"  Class 0 (negative): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"  Class 1 (positive): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    print(f"  Class 2 (uncertain): {class_counts[2]} ({class_counts[2]/total*100:.1f}%)")
    print(f"  Class 3 (unlabeled): {class_counts[3]} ({class_counts[3]/total*100:.1f}%)")
    print(f"\n[INFO] Computed class weights: {weights.cpu().numpy()}")
    
    return weights


def compute_class_weights(pseudo_labels, label_names, device='cuda'):
    """
    Compute class weights based on training data distribution to handle class imbalance.
    
    Args:
        pseudo_labels: Dictionary mapping sample_id to label dict
        label_names: List of label names
        device: Device to create weights tensor on
    
    Returns:
        class_weights: Tensor of shape [4] with weights for each class
                     [class_0_weight, class_1_weight, class_2_weight, class_3_weight]
    """
    # Count occurrences of each class across all labels
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for sample_id, labels in pseudo_labels.items():
        for label_name in label_names:
            if label_name in labels:
                val = labels[label_name]
                if val == 0.0:
                    class_counts[0] += 1
                elif val == 1.0:
                    class_counts[1] += 1
                elif val == -1.0:
                    class_counts[2] += 1
                elif val is None:
                    class_counts[3] += 1
    
    total = sum(class_counts.values())
    if total == 0:
        # Default equal weights if no data
        return torch.ones(4, device=device)
    
    # Compute inverse frequency weights (more weight for rare classes)
    # Add small epsilon to avoid division by zero
    weights = []
    for cls in range(4):
        if class_counts[cls] > 0:
            # Inverse frequency: total / (num_classes * count)
            weight = total / (4.0 * class_counts[cls])
        else:
            # If class never appears, give it high weight to encourage learning
            weight = total / (4.0 * 1.0)
        weights.append(weight)
    
    # Normalize weights so they sum to num_classes (standard practice)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum() * 4.0
    
    print(f"\n[INFO] Class distribution:")
    print(f"  Class 0 (negative): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"  Class 1 (positive): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    print(f"  Class 2 (uncertain): {class_counts[2]} ({class_counts[2]/total*100:.1f}%)")
    print(f"  Class 3 (unlabeled): {class_counts[3]} ({class_counts[3]/total*100:.1f}%)")
    print(f"\n[INFO] Computed class weights: {weights.cpu().numpy()}")
    
    return weights


def create_model(config, encoder_checkpoint=None, freeze_encoder=False, device='cuda'):
    """Helper function to create model (ResNetClassifier or DenseNetClassifier) with encoder config
    
    Args:
        config: Full configuration dict
        encoder_checkpoint: Path to encoder checkpoint (overrides config if provided)
        freeze_encoder: Whether to freeze encoder
        device: Device to place model on
    
    Returns:
        model: ResNetClassifier or DenseNetClassifier instance on specified device
    """
    # Get model type (resnet or densenet)
    model_type = config['wsl'].get('model_type', 'resnet').lower()  # Default to resnet
    
    # Get encoder config
    encoder_config = config['wsl'].get('encoder', {})
    encoder_type = encoder_config.get('type', 'resnet50' if model_type == 'resnet' else 'densenet121')
    xrv_weights = encoder_config.get('xrv_weights', 'resnet50-res512-all' if model_type == 'resnet' else 'densenet121-res224-all')
    config_checkpoint = encoder_config.get('checkpoint', None)
    
    # Use provided checkpoint if specified, otherwise use config
    checkpoint_to_use = encoder_checkpoint if encoder_checkpoint is not None else config_checkpoint
    
    print(f"[INFO] Creating {model_type} model with encoder type: {encoder_type}")
    if encoder_type == 'xrv':
        print(f"[INFO] Using TorchXRayVision weights: {xrv_weights}")
    if checkpoint_to_use:
        print(f"[INFO] Loading encoder checkpoint: {checkpoint_to_use}")
    
    # Get number of labels
    num_labels = config['wsl'].get('num_labels', config['wsl'].get('num_classes', 14))
    
    # Determine num_classes_per_label based on loss type
    loss_type = config['wsl']['loss'].get('type', 'multi_class')
    if loss_type == 'binary':
        num_classes_per_label = 1  # Binary classification: 1 logit per label (sigmoid)
    else:
        num_classes_per_label = config['wsl'].get('num_classes_per_label', 4)  # Multi-class: 4 classes per label
    
    print(f"[INFO] Loss type: {loss_type}, Classes per label: {num_classes_per_label} ({'binary' if num_classes_per_label == 1 else 'multi-class'})")
    
    # Create model based on model_type
    if model_type == 'densenet':
        model = DenseNetClassifier(
            num_labels=num_labels,
            num_classes_per_label=num_classes_per_label,
            encoder_type=encoder_type,
            xrv_weights=xrv_weights,
            encoder_checkpoint=checkpoint_to_use,
            freeze_encoder=freeze_encoder
        ).to(device)
    else:  # Default to resnet
        model = ResNetClassifier(
            num_labels=num_labels,
            num_classes_per_label=num_classes_per_label,
            encoder_type=encoder_type,
            xrv_weights=xrv_weights,
            encoder_checkpoint=checkpoint_to_use,
            freeze_encoder=freeze_encoder
        ).to(device)
    
    return model


def find_latest_epoch_checkpoint(checkpoint_dir, phase='linear_probe'):
    """Find the latest epoch checkpoint file for a given phase"""
    epoch_files = list(checkpoint_dir.glob(f'{phase}_epoch_*.pth'))
    if not epoch_files:
        return None, 0
    
    # Extract epoch numbers
    epochs = []
    for f in epoch_files:
        try:
            epoch_num = int(f.stem.split('_')[-1])
            epochs.append((epoch_num, f))
        except (ValueError, IndexError):
            continue
    
    if not epochs:
        return None, 0
    
    # Get the latest epoch
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return latest_file, latest_epoch


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, log_interval=10):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        label_mask = batch['label_mask'].to(device)  # NEW: use mask from dataset
        
        # Forward
        logits = model(images)  # [B, num_labels] for binary, [B, num_labels, num_classes_per_label] for multi-class
        
        # Loss computation
        # For binary mode, pass label_mask; for multi-class mode, mask is handled internally
        if isinstance(criterion, MultiLabelBCELoss):
            loss = criterion(logits, labels, label_mask)
        else:
            loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Log
        if batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)
            
            logits = model(images)
            
            # Compute loss
            if isinstance(criterion, MultiLabelBCELoss):
                loss = criterion(logits, labels, label_mask)
                # For binary mode: apply sigmoid to get probabilities [B, num_labels]
                probs = torch.sigmoid(logits)  # [B, num_labels]
            else:
                loss = criterion(logits, labels)
                # For multi-class mode: apply softmax and extract positive class (class 1)
                probs_all = torch.softmax(logits, dim=-1)  # [B, num_labels, num_classes_per_label]
                probs = probs_all[:, :, 1]  # Extract class 1 (positive) probabilities -> [B, num_labels]
            
            total_loss += loss.item()
            
            # Convert to numpy for metrics
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    preds = np.concatenate(all_preds, axis=0)  # [N, num_labels]
    labels = np.concatenate(all_labels, axis=0)  # [N, num_labels]
    
    # Convert labels to binary format for metrics
    # Map: 1.0 (positive) -> 1, -1.0 (uncertain) -> 1, 0.0 (negative) -> 0, NaN -> -1 (invalid)
    labels_binary = np.zeros_like(labels)
    labels_binary[(labels == 1.0) | (labels == -1.0)] = 1.0
    labels_binary[labels == 0.0] = 0.0
    labels_binary[np.isnan(labels)] = -1.0  # Mark NaN as invalid
    
    metrics = compute_metrics(preds, labels_binary)
    avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
    
    return avg_loss, metrics


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Detect data source type
    source_type = config.get('data', {}).get('source_type', 'parquet')
    print(f"[INFO] Data source type: {source_type}")
    
    # Get image size from config (prefer wsl.image_size, fallback to mae.image_size, default 224)
    image_size = config.get('wsl', {}).get('image_size', 
                                           config.get('mae', {}).get('image_size', 224))
    skip_resize = config.get('data', {}).get('mimic_jpg', {}).get('skip_resize', False)
    print(f"[INFO] Using image size: {image_size}")
    if skip_resize:
        print(f"[INFO] Using pre-resized images (skip_resize=True)")
    train_transform = get_transforms(image_size, is_train=True, skip_resize=skip_resize)
    val_transform = get_transforms(image_size, is_train=False, skip_resize=skip_resize)
    
    if source_type == 'mimic_jpg':
        # ========== MIMIC-CXR-JPG: Use ground truth labels from CSV ==========
        print("\n[INFO] Using MIMIC-CXR-JPG dataset with ground truth labels")
        
        mimic_cfg = config['data']['mimic_jpg']
        
        # Create train dataset
        train_dataset = MIMICCXRJPGDataset(
            image_root=mimic_cfg['image_root'],
            chexpert_csv=mimic_cfg['chexpert_csv'],
            split_csv=mimic_cfg['split_csv'],
            split='train',
            transform=train_transform,
            label_names=config['data']['labels'],
            num_samples=config['data'].get('num_samples', None)
        )
        
        # Create validation dataset
        # Option 1: Use official test set (recommended for evaluating on ground truth)
        if 'test_csv' in mimic_cfg:
            print("[INFO] Using official MIMIC-CXR 2.1.0 test set for validation")
            val_dataset = MIMICCXRTestSet(
                image_root=mimic_cfg['image_root'],
                test_csv=mimic_cfg['test_csv'],
                split_csv=mimic_cfg['split_csv'],
                transform=val_transform,
                label_names=config['data']['labels']
            )
        else:
            # Option 2: Use MIMIC's validation split
            print("[INFO] Using MIMIC validation split")
            val_dataset = MIMICCXRJPGDataset(
                image_root=mimic_cfg['image_root'],
                chexpert_csv=mimic_cfg['chexpert_csv'],
                split_csv=mimic_cfg['split_csv'],
                split='validate',
                transform=val_transform,
                label_names=config['data']['labels'],
                num_samples=None
            )
        
        print(f"[INFO] Train samples: {len(train_dataset)}")
        print(f"[INFO] Validation samples: {len(val_dataset)}")
        
    else:
        # ========== Parquet: Use pseudo labels ==========
        print("\n[INFO] Using Parquet dataset with pseudo labels")
        
        # Load pseudo labels
        pseudo_labels_path = config['llm']['cache_file']
        # If relative path, resolve it relative to output_dir
        if not os.path.isabs(pseudo_labels_path):
            if 'output_dir' in config.get('pipeline', {}):
                pseudo_labels_path = os.path.join(config['pipeline']['output_dir'], pseudo_labels_path)
        
        if not Path(pseudo_labels_path).exists():
            raise FileNotFoundError(f"Pseudo labels not found at {pseudo_labels_path}. Run generate_labels.py first.")
        
        with open(pseudo_labels_path, 'r') as f:
            pseudo_labels = json.load(f)
        
        # Check if using cleaned samples
        is_cleaned = 'cleaned' in str(pseudo_labels_path)
        if is_cleaned:
            print(f"[Using Cleaned Samples] Loaded pseudo labels for {len(pseudo_labels)} samples from: {pseudo_labels_path}")
        else:
            print(f"Loaded pseudo labels for {len(pseudo_labels)} samples")
        
        # Load data splits
        import pandas as pd
        dfs = []
        for path in config['data']['parquet_paths']:
            if Path(path).exists():
                dfs.append(pd.read_parquet(path))
        
        # Train: Use ALL pseudo labels (no splitting)
        train_indices = sorted([int(idx) for idx in pseudo_labels.keys()])
        print(f"Train: {len(train_indices)} samples (all pseudo labels)")
        
        # Val: Load from test set (ground truth)
        test_set_json = config.get('data', {}).get('test_set_json', 'outputs/test_set_2_1_0.json')
        val_ratio_from_test = config.get('data', {}).get('val_ratio_from_test', 0.15)  # 15% of test set
        
        # Resolve path relative to project root if needed
        if not os.path.isabs(test_set_json):
            project_root = Path(__file__).parent  # train_wsl.py is in project root
            test_set_json = str(project_root / test_set_json)
        
        val_samples, _ = load_test_set_for_validation(
            test_set_json, 
            val_ratio=val_ratio_from_test,
            seed=config['data']['seed']
        )
        
        train_dataset = MIMICDataset(
            config['data']['parquet_paths'],
            indices=train_indices,
            transform=train_transform,
            pseudo_labels=pseudo_labels,
            label_names=config['data']['labels']
        )
        
        # Validation dataset from test set (ground truth)
        test_image_dir = config.get('data', {}).get('test_image_dir', 'datasets/test_set')
        if not os.path.isabs(test_image_dir):
            project_root = Path(__file__).parent  # train_wsl.py is in project root
            test_image_dir = str(project_root / test_image_dir)
        
        val_dataset = TestSetDataset(
            test_samples=val_samples,
            image_dir=test_image_dir,
            transform=val_transform,
            label_names=config['data']['labels'],
            use_ground_truth=True  # Use ground truth for validation
        )
    
    # ========== Stage 1: Linear Probe ==========
    print("\n" + "="*50)
    print("Stage 1: Linear Probe (Freeze Encoder)")
    print("="*50)
    
    # Resolve checkpoint directory
    checkpoint_dir = config['wsl']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir):
        if 'output_dir' in config.get('pipeline', {}):
            checkpoint_dir = os.path.join(config['pipeline']['output_dir'], checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if skip linear probe
    skip_linear_probe = config['wsl'].get('skip_linear_probe', False)
    
    if skip_linear_probe:
        print("\n[INFO] Linear Probe is disabled (skip_linear_probe=true)")
        print("[INFO] Will proceed directly to end-to-end fine-tuning")
        linear_probe_completed = True
        # Create a dummy best file if needed (so fine-tune can proceed)
        lp_best_path = checkpoint_dir / 'linear_probe_best.pth'
        if not lp_best_path.exists():
            print("[INFO] Creating placeholder for linear_probe_best.pth")
            # Will be created with encoder weights only
    else:
        # Check for existing linear probe epoch checkpoints
        lp_checkpoint, lp_start_epoch = find_latest_epoch_checkpoint(checkpoint_dir, 'linear_probe')
        lp_target_epochs = config['wsl']['linear_probe']['epochs']
        
        linear_probe_completed = False
        if lp_checkpoint is not None:
            if lp_start_epoch >= lp_target_epochs:
                print(f"\n[INFO] Linear Probe already completed! Found checkpoint at epoch {lp_start_epoch}/{lp_target_epochs}")
                
                # Check if final best model exists
                lp_best = checkpoint_dir / 'linear_probe_best.pth'
                if not lp_best.exists():
                    print(f"\n[INFO] Generating final best model...")
                    
                    # Try to use temp_best first (it has the best val_loss)
                    lp_temp_best = checkpoint_dir / 'linear_probe_temp_best.pth'
                    if lp_temp_best.exists():
                        temp_best = torch.load(lp_temp_best, map_location=device)
                        torch.save(temp_best['model_state_dict'], lp_best)
                        print(f"[INFO] Saved {lp_best} from temp_best (val_loss: {temp_best['val_loss']:.4f}, epoch: {temp_best['epoch']})")
                        lp_temp_best.unlink()
                        print(f"[INFO] Removed {lp_temp_best.name}")
                    else:
                        # Fallback: use last epoch checkpoint
                        import shutil
                        shutil.copy(lp_checkpoint, lp_best)
                        print(f"[INFO] Saved {lp_best} from epoch {lp_start_epoch}")
                
                linear_probe_completed = True
            else:
                print(f"\n[INFO] Resuming Linear Probe from epoch {lp_start_epoch + 1}/{lp_target_epochs}")
        else:
            lp_start_epoch = 0
            print(f"\n[INFO] Starting Linear Probe for {lp_target_epochs} epochs")
    
    # Only run linear probe if not completed
    if not linear_probe_completed:
        # Check for pretrained encoder
        mae_dir = config['mae']['checkpoint_dir']
        if not os.path.isabs(mae_dir):
            if 'output_dir' in config.get('pipeline', {}):
                mae_dir = os.path.join(config['pipeline']['output_dir'], mae_dir)
        
        encoder_path = Path(mae_dir) / 'encoder_best.pth'
        if not encoder_path.exists():
            print(f"Warning: Pretrained encoder not found at {encoder_path}")
            encoder_path = None
        
        # Create model with frozen encoder
        # Note: When using xrv encoder, encoder_path is typically None (we use pretrained xrv weights)
        model = create_model(
            config=config,
            encoder_checkpoint=str(encoder_path) if encoder_path else None,
            freeze_encoder=True,
            device=device
        )
        
        # Load checkpoint if resuming
        if lp_checkpoint is not None and lp_start_epoch > 0:
            model.load_state_dict(torch.load(lp_checkpoint, map_location=device))
            print(f"[INFO] Loaded model from epoch {lp_start_epoch}")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['wsl']['linear_probe']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['wsl']['linear_probe']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        # Create loss function based on loss type
        loss_type = config['wsl']['loss'].get('type', 'multi_class')
        use_class_weights = config['wsl']['loss'].get('use_class_weights', False)
        
        if loss_type == 'binary':
            # Binary classification: use BCE loss (without weights as requested)
            criterion = MultiLabelBCELoss(pos_weight=None)
            print(f"[INFO] Using Binary Cross-Entropy loss (BCE) without class weights")
        else:
            # Multi-class classification: use multi-class CE loss with optional weights
            if use_class_weights:
                if source_type == 'mimic_jpg':
                    class_weights = compute_class_weights_from_dataset(train_dataset, config['data']['labels'], device=device)
                else:
                    class_weights = compute_class_weights(pseudo_labels, config['data']['labels'], device=device)
                criterion = MultiLabelMultiClassLoss(class_weights=class_weights)
                print(f"[INFO] Using Multi-class Cross-Entropy loss with class weights")
            else:
                criterion = MultiLabelMultiClassLoss(class_weights=None)
                print(f"[INFO] Using Multi-class Cross-Entropy loss without class weights")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['wsl']['linear_probe']['lr']
        )
        
        # Learning rate scheduler - ReduceLROnPlateau
        scheduler_patience = config['wsl']['linear_probe'].get('scheduler_patience', 5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-7,
        )
        print(f"[INFO] Linear Probe scheduler patience: {scheduler_patience}")
        
        # Tensorboard
        writer = SummaryWriter(log_dir=checkpoint_dir / 'logs_linear_probe')
        
        # Early stopping configuration
        early_stopping_config = config['wsl']['linear_probe'].get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', False)
        early_stopping_patience = early_stopping_config.get('patience', 30)
        early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0)
        early_stopping_counter = 0
        
        if early_stopping_enabled:
            print(f"[INFO] Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        
        # Training loop
        # Load best_val_loss from temp_best if exists (for resume)
        best_val_loss = float('inf')
        lp_temp_best_path = checkpoint_dir / 'linear_probe_temp_best.pth'
        if lp_temp_best_path.exists():
            temp_best = torch.load(lp_temp_best_path, map_location=device)
            if 'val_loss' in temp_best:
                best_val_loss = temp_best['val_loss']
                print(f"[INFO] Found existing temp_best with val_loss: {best_val_loss:.4f} (epoch {temp_best.get('epoch', '?')})")
        
        for epoch in range(lp_start_epoch + 1, lp_target_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            # Step scheduler based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}/{lp_target_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            print(f"Val Metrics: {val_metrics}")
            
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('train/lr', current_lr, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            
            # Save checkpoint with epoch number (for resume)
            checkpoint_path = checkpoint_dir / f'linear_probe_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path.name}")
            
            # Save temp_best if this is the best so far
            improved = False
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                early_stopping_counter = 0  # Reset counter on improvement
                improved = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, lp_temp_best_path)
                print(f"New best val loss: {best_val_loss:.4f} - Saved to {lp_temp_best_path.name}")
            else:
                if early_stopping_enabled:
                    early_stopping_counter += 1
                    print(f"No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
            
            # Early stopping check
            if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
                print(f"\n[EARLY STOPPING] No improvement for {early_stopping_patience} epochs. Stopping Linear Probe training.")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {epoch - early_stopping_counter}")
                break
            
            # Delete old epoch checkpoints to save space (keep only last 2)
            epoch_files = sorted(checkpoint_dir.glob('linear_probe_epoch_*.pth'), 
                                key=lambda x: int(x.stem.split('_')[-1]))
            if len(epoch_files) > 2:
                for old_file in epoch_files[:-2]:
                    old_file.unlink()
                    print(f"Deleted old checkpoint: {old_file.name}")
        
        # Training completed - finalize best model
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            print(f"\n[SUCCESS] Linear Probe stopped early at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
        else:
            print(f"\n[SUCCESS] Linear Probe completed all {lp_target_epochs} epochs!")
        print(f"Finalizing best model...")
        
        # Rename temp_best to best (for phase check)
        if lp_temp_best_path.exists():
            temp_best = torch.load(lp_temp_best_path, map_location=device)
            lp_best_path = checkpoint_dir / 'linear_probe_best.pth'
            torch.save(temp_best['model_state_dict'], lp_best_path)
            print(f"Saved {lp_best_path.name} (val_loss: {temp_best['val_loss']:.4f}, epoch: {temp_best['epoch']})")
            
            # Remove temp_best
            lp_temp_best_path.unlink()
            print(f"Removed temporary file: {lp_temp_best_path.name}")
        else:
            print("[WARN] No temp_best found - this shouldn't happen")
        
        writer.close()
    
    # ========== Stage 2: Fine-tune ==========
    skip_linear_probe = config['wsl'].get('skip_linear_probe', False)
    if skip_linear_probe:
        print("\n" + "="*50)
        print("Stage 2: End-to-End Fine-tune (All Layers)")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Stage 2: Fine-tune (Unfreeze Top Layers)")
        print("="*50)
    
    # Check for existing fine-tune epoch checkpoints
    ft_checkpoint, ft_start_epoch = find_latest_epoch_checkpoint(checkpoint_dir, 'fine_tune')
    ft_target_epochs = config['wsl']['fine_tune']['epochs']
    
    fine_tune_completed = False
    if ft_checkpoint is not None:
        if ft_start_epoch >= ft_target_epochs:
            print(f"\n[INFO] Fine-tune already completed! Found checkpoint at epoch {ft_start_epoch}/{ft_target_epochs}")
            
            # Check if final best model exists
            ft_best = checkpoint_dir / 'fine_tune_best.pth'
            if not ft_best.exists():
                print(f"\n[INFO] Generating final best model...")
                
                # Try to use temp_best first (it has the best val_loss)
                ft_temp_best = checkpoint_dir / 'fine_tune_temp_best.pth'
                if ft_temp_best.exists():
                    temp_best = torch.load(ft_temp_best, map_location=device)
                    torch.save(temp_best['model_state_dict'], ft_best)
                    print(f"[INFO] Saved {ft_best} from temp_best (val_loss: {temp_best['val_loss']:.4f}, epoch: {temp_best['epoch']})")
                    ft_temp_best.unlink()
                    print(f"[INFO] Removed {ft_temp_best.name}")
                else:
                    # Fallback: use last epoch checkpoint
                    import shutil
                    shutil.copy(ft_checkpoint, ft_best)
                    print(f"[INFO] Saved {ft_best} from epoch {ft_start_epoch}")
            
            fine_tune_completed = True
        else:
            print(f"\n[INFO] Resuming Fine-tune from epoch {ft_start_epoch + 1}/{ft_target_epochs}")
    else:
        ft_start_epoch = 0
        print(f"\n[INFO] Starting Fine-tune for {ft_target_epochs} epochs")
    
    # Only run fine-tune if not completed
    if not fine_tune_completed:
        # Check for pretrained encoder if starting fresh
        mae_dir = config['mae']['checkpoint_dir']
        if not os.path.isabs(mae_dir):
            if 'output_dir' in config.get('pipeline', {}):
                mae_dir = os.path.join(config['pipeline']['output_dir'], mae_dir)
        
        encoder_path = Path(mae_dir) / 'encoder_best.pth'
        if not encoder_path.exists():
            print(f"Warning: Pretrained encoder not found at {encoder_path}")
            encoder_path = None
        
        # Create or load model
        if ft_start_epoch == 0:
            # Check if we skipped linear probe
            skip_linear_probe = config['wsl'].get('skip_linear_probe', False)
            
            if skip_linear_probe:
                # Starting fresh - load from encoder only (end-to-end training)
                print("[INFO] Skipped linear probe - initializing model from encoder")
                model = create_model(
                    config=config,
                    encoder_checkpoint=str(encoder_path) if encoder_path else None,
                    freeze_encoder=False,  # Don't freeze for end-to-end training
                    device=device
                )
            else:
                # Starting fresh - load from linear probe
                model = create_model(
                    config=config,
                    encoder_checkpoint=str(encoder_path) if encoder_path else None,
                    freeze_encoder=True,
                    device=device
                )
                model.load_state_dict(torch.load(checkpoint_dir / 'linear_probe_best.pth', map_location=device))
                print("[INFO] Loaded best linear probe model")
        else:
            # Resuming - load from fine-tune checkpoint
            model = create_model(
                config=config,
                encoder_checkpoint=str(encoder_path) if encoder_path else None,
                freeze_encoder=False,
                device=device
            )
            model.load_state_dict(torch.load(ft_checkpoint, map_location=device))
            print(f"[INFO] Loaded model from epoch {ft_start_epoch}")
        
        # Unfreeze top layers (skip if doing end-to-end training)
        skip_linear_probe = config['wsl'].get('skip_linear_probe', False)
        if not skip_linear_probe:
            model.unfreeze_top_layers(config['wsl']['fine_tune']['unfreeze_layers'])
        else:
            # Already unfrozen in model creation for end-to-end training
            print("[INFO] All layers already unfrozen for end-to-end training")
        
        # New optimizer with lower learning rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['wsl']['fine_tune']['lr']
        )
        
        # Learning rate scheduler - ReduceLROnPlateau
        scheduler_patience = config['wsl']['fine_tune'].get('scheduler_patience', 5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-7,
        )
        print(f"[INFO] Fine-tune scheduler patience: {scheduler_patience}")
        
        # New data loader with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['wsl']['fine_tune']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['wsl']['fine_tune']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
        
        # Create loss function based on loss type
        loss_type = config['wsl']['loss'].get('type', 'multi_class')
        use_class_weights = config['wsl']['loss'].get('use_class_weights', False)
        
        if loss_type == 'binary':
            # Binary classification: use BCE loss (without weights as requested)
            criterion = MultiLabelBCELoss(pos_weight=None)
            print(f"[INFO] Using Binary Cross-Entropy loss (BCE) without class weights")
        else:
            # Multi-class classification: use multi-class CE loss with optional weights
            if use_class_weights:
                if source_type == 'mimic_jpg':
                    class_weights = compute_class_weights_from_dataset(train_dataset, config['data']['labels'], device=device)
                else:
                    class_weights = compute_class_weights(pseudo_labels, config['data']['labels'], device=device)
                criterion = MultiLabelMultiClassLoss(class_weights=class_weights)
                print(f"[INFO] Using Multi-class Cross-Entropy loss with class weights")
            else:
                criterion = MultiLabelMultiClassLoss(class_weights=None)
                print(f"[INFO] Using Multi-class Cross-Entropy loss without class weights")
        
        # Tensorboard
        writer = SummaryWriter(log_dir=checkpoint_dir / 'logs_fine_tune')
        
        # Early stopping configuration
        early_stopping_config = config['wsl']['fine_tune'].get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', False)
        early_stopping_patience = early_stopping_config.get('patience', 30)
        early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0)
        early_stopping_counter = 0
        
        if early_stopping_enabled:
            print(f"[INFO] Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        
        # Training loop
        # Load best_val_loss from temp_best if exists (for resume)
        best_val_loss = float('inf')
        ft_temp_best_path = checkpoint_dir / 'fine_tune_temp_best.pth'
        if ft_temp_best_path.exists():
            temp_best = torch.load(ft_temp_best_path, map_location=device)
            if 'val_loss' in temp_best:
                best_val_loss = temp_best['val_loss']
                print(f"[INFO] Found existing temp_best with val_loss: {best_val_loss:.4f} (epoch {temp_best.get('epoch', '?')})")
        
        for epoch in range(ft_start_epoch + 1, ft_target_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            # Step scheduler based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}/{ft_target_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            print(f"Val Metrics: {val_metrics}")
            
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('train/lr', current_lr, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            
            # Save checkpoint with epoch number (for resume)
            checkpoint_path = checkpoint_dir / f'fine_tune_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path.name}")
            
            # Save temp_best if this is the best so far
            improved = False
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                early_stopping_counter = 0  # Reset counter on improvement
                improved = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, ft_temp_best_path)
                print(f"New best val loss: {best_val_loss:.4f} - Saved to {ft_temp_best_path.name}")
            else:
                if early_stopping_enabled:
                    early_stopping_counter += 1
                    print(f"No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
            
            # Early stopping check
            if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
                print(f"\n[EARLY STOPPING] No improvement for {early_stopping_patience} epochs. Stopping Fine-tune training.")
                print(f"Best val loss: {best_val_loss:.4f} at epoch {epoch - early_stopping_counter}")
                break
            
            # Delete old epoch checkpoints to save space (keep only last 2)
            epoch_files = sorted(checkpoint_dir.glob('fine_tune_epoch_*.pth'), 
                                key=lambda x: int(x.stem.split('_')[-1]))
            if len(epoch_files) > 2:
                for old_file in epoch_files[:-2]:
                    old_file.unlink()
                    print(f"Deleted old checkpoint: {old_file.name}")
        
        # Training completed - finalize best model
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            print(f"\n[SUCCESS] Fine-tune stopped early at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
        else:
            print(f"\n[SUCCESS] Fine-tune completed all {ft_target_epochs} epochs!")
        print(f"Finalizing best model...")
        
        # Rename temp_best to best (for phase check)
        if ft_temp_best_path.exists():
            temp_best = torch.load(ft_temp_best_path, map_location=device)
            ft_best_path = checkpoint_dir / 'fine_tune_best.pth'
            torch.save(temp_best['model_state_dict'], ft_best_path)
            print(f"Saved {ft_best_path.name} (val_loss: {temp_best['val_loss']:.4f}, epoch: {temp_best['epoch']})")
            
            # Remove temp_best
            ft_temp_best_path.unlink()
            print(f"Removed temporary file: {ft_temp_best_path.name}")
        else:
            print("[WARN] No temp_best found - this shouldn't happen")
        
        writer.close()
    
    # Save predictions for confident learning
    print("\nGenerating predictions for Confident Learning...")
    
    # Ensure model is loaded
    if fine_tune_completed or 'model' not in locals():
        # Need to create model and load weights
        mae_dir = config['mae']['checkpoint_dir']
        if not os.path.isabs(mae_dir):
            if 'output_dir' in config.get('pipeline', {}):
                mae_dir = os.path.join(config['pipeline']['output_dir'], mae_dir)
        
        encoder_path = Path(mae_dir) / 'encoder_best.pth'
        if not encoder_path.exists():
            encoder_path = None
        
        model = create_model(
            config=config,
            encoder_checkpoint=str(encoder_path) if encoder_path else None,
            freeze_encoder=False,
            device=device
        )
    
    model.load_state_dict(torch.load(checkpoint_dir / 'fine_tune_best.pth', map_location=device))
    
    # Ensure val_loader exists
    if 'val_loader' not in locals():
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['wsl']['fine_tune']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    
    preds, labels, indices = get_predictions(model, val_loader, device)
    
    # Save predictions
    pred_save_path = checkpoint_dir / 'val_predictions.npz'
    import numpy as np
    np.savez(
        pred_save_path,
        preds=preds,
        labels=labels,
        indices=indices
    )
    print(f"Saved predictions to {pred_save_path}")
    
    print("\nWSL training completed!")


if __name__ == '__main__':
    main()
