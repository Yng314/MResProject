"""
Phase 2: Train MAE for self-supervised pretraining
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from data.dataset import MAEDataset, get_transforms, split_dataset
from models.mae import MAE


def train_epoch(model, dataloader, optimizer, device, epoch, writer, scheduler=None, use_onecycle=False):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        
        # Forward
        reconstructed, mask, _ = model(images)
        
        # Compute loss (MSE on masked patches)
        # Reshape for patch-wise loss
        B, C, H, W = images.shape
        p = model.patch_size
        
        # Reshape to patches
        images_patches = images.reshape(B, C, H // p, p, W // p, p)
        images_patches = images_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, model.num_patches, C * p * p)
        
        recon_patches = reconstructed.reshape(B, C, H // p, p, W // p, p)
        recon_patches = recon_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, model.num_patches, C * p * p)
        
        # Loss only on masked patches
        mask_expanded = (1 - mask).unsqueeze(-1).expand_as(images_patches)
        loss = ((recon_patches - images_patches) ** 2 * mask_expanded).sum() / mask_expanded.sum()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # OneCycleLR needs step after each batch
        if use_onecycle and scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.2e}'})
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/lr', current_lr, global_step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def find_latest_epoch_checkpoint(checkpoint_dir):
    """Find the latest epoch checkpoint file"""
    epoch_files = list(checkpoint_dir.glob('mae_epoch_*.pth'))
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
    
    # Create checkpoint directory
    checkpoint_dir = config['mae']['checkpoint_dir']
    # If relative path, resolve it relative to output_dir
    if not os.path.isabs(checkpoint_dir):
        if 'output_dir' in config.get('pipeline', {}):
            checkpoint_dir = os.path.join(config['pipeline']['output_dir'], checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing epoch checkpoints
    latest_checkpoint, start_epoch = find_latest_epoch_checkpoint(checkpoint_dir)
    target_epochs = config['mae']['epochs']
    
    if latest_checkpoint is not None:
        if start_epoch >= target_epochs:
            print(f"\n[INFO] Training already completed! Found checkpoint at epoch {start_epoch}/{target_epochs}")
            print(f"[INFO] Checkpoint: {latest_checkpoint}")
            
            # Check if final best models exist
            mae_best = checkpoint_dir / 'mae_best.pth'
            encoder_best = checkpoint_dir / 'encoder_best.pth'
            
            if not mae_best.exists() or not encoder_best.exists():
                print(f"\n[INFO] Generating final best models from epoch {start_epoch}...")
                # Load the model and save as best
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                
                # Save as mae_best.pth
                torch.save(checkpoint, mae_best)
                print(f"[INFO] Saved {mae_best}")
                
                # For encoder_best, we need to load the model
                print("Creating MAE model to extract encoder...")
                model = MAE(
                    img_size=config['mae']['image_size'],
                    patch_size=config['mae']['patch_size'],
                    mask_ratio=config['mae']['mask_ratio']
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.save_encoder(str(encoder_best))
                print(f"[INFO] Saved {encoder_best}")
                
                print("\n[SUCCESS] All training completed and best models saved!")
            else:
                print(f"[INFO] Best models already exist. Training complete!")
            
            return
        else:
            print(f"\n[INFO] Resuming training from epoch {start_epoch + 1}/{target_epochs}")
            print(f"[INFO] Loading checkpoint: {latest_checkpoint}")
    else:
        start_epoch = 0
        print(f"\n[INFO] Starting fresh training for {target_epochs} epochs")
    
    # Load data
    print("Loading data...")
    
    # Check if we should use cleaned samples (from confident learning)
    use_cleaned = False
    cleaned_indices = None
    
    if 'llm' in config and 'cache_file' in config['llm']:
        cache_file = config['llm']['cache_file']
        # If relative path, resolve it relative to output_dir
        if not os.path.isabs(cache_file):
            if 'output_dir' in config.get('pipeline', {}):
                cache_file = os.path.join(config['pipeline']['output_dir'], cache_file)
        
        # Check if the cache_file points to cleaned labels
        if 'cleaned' in str(cache_file) and Path(cache_file).exists():
            import json
            print(f"Using cleaned samples from: {cache_file}")
            with open(cache_file, 'r') as f:
                cleaned_labels = json.load(f)
            cleaned_indices = sorted([int(idx) for idx in cleaned_labels.keys()])
            use_cleaned = True
            print(f"Loaded {len(cleaned_indices)} cleaned samples")
    
    # Determine total samples
    num_samples = config['data']['num_samples']
    if num_samples <= 0:
        # Count total samples
        import pandas as pd
        dfs = []
        for path in config['data']['parquet_paths']:
            if Path(path).exists():
                dfs.append(pd.read_parquet(path))
        num_samples = len(pd.concat(dfs, ignore_index=True))
    
    # Split dataset
    if use_cleaned:
        # Split using only cleaned indices
        from sklearn.model_selection import train_test_split
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        
        # First split: separate out validation
        if val_ratio > 0:
            train_indices, val_indices = train_test_split(
                cleaned_indices,
                test_size=val_ratio,
                random_state=config['data']['seed']
            )
        else:
            train_indices = cleaned_indices
            val_indices = []
        
        test_indices = []  # No test set for MAE
        print(f"[Using Cleaned Samples] Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    else:
        # Original split logic
        train_indices, val_indices, test_indices = split_dataset(
            num_samples,
            config['data']['train_ratio'],
            config['data']['val_ratio'],
            config['data']['test_ratio'],
            config['data']['seed']
        )
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create datasets
    transform = get_transforms(config['mae']['image_size'], is_train=True)
    train_dataset = MAEDataset(
        config['data']['parquet_paths'],
        indices=train_indices,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mae']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create model
    print("Creating MAE model...")
    model = MAE(
        img_size=config['mae']['image_size'],
        patch_size=config['mae']['patch_size'],
        mask_ratio=config['mae']['mask_ratio']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['mae']['lr'],
        weight_decay=config['mae']['weight_decay']
    )
    
    # Load checkpoint if resuming
    if latest_checkpoint is not None and start_epoch > 0:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] Loaded model and optimizer from epoch {start_epoch}")
    
    # Learning rate scheduler with multiple options
    scheduler_type = config['mae'].get('scheduler', 'cosine')  # default to cosine
    
    if scheduler_type == 'cosine':
        # Cosine annealing: smooth decay from lr to 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['mae']['epochs']
        )
        use_plateau = False
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau: auto reduce when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['mae'].get('scheduler_factor', 0.5),  # reduce by half
            patience=config['mae'].get('scheduler_patience', 5),  # wait 5 epochs
            min_lr=1e-7
        )
        use_plateau = True
        print(f"ReduceLROnPlateau: factor={config['mae'].get('scheduler_factor', 0.5)}, patience={config['mae'].get('scheduler_patience', 5)}")
    elif scheduler_type == 'onecycle':
        # OneCycleLR: fast training with peak lr in the middle
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['mae']['lr'] * 10,  # peak at 10x initial lr
            epochs=config['mae']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy='cos'
        )
        use_plateau = False
    elif scheduler_type == 'step':
        # StepLR: reduce lr every N epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['mae'].get('scheduler_step_size', 10),
            gamma=config['mae'].get('scheduler_gamma', 0.5)
        )
        use_plateau = False
    else:
        # No scheduler
        scheduler = None
        use_plateau = False
    
    if scheduler is not None:
        print(f"Using {scheduler_type} learning rate scheduler")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=checkpoint_dir / 'logs')
    
    # Training loop
    print("Starting MAE pretraining...")
    
    # Early stopping configuration
    early_stopping_config = config['mae'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_config.get('enabled', False)
    early_stopping_patience = early_stopping_config.get('patience', 30)
    early_stopping_min_delta = early_stopping_config.get('min_delta', 0.0)
    early_stopping_counter = 0
    
    if early_stopping_enabled:
        print(f"[INFO] Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    
    # Load best_loss from temp_best if exists (for resume)
    best_loss = float('inf')
    temp_best_path = checkpoint_dir / 'mae_temp_best.pth'
    if temp_best_path.exists():
        temp_best = torch.load(temp_best_path, map_location=device)
        best_loss = temp_best['loss']
        print(f"[INFO] Found existing temp_best with loss: {best_loss:.4f} (epoch {temp_best['epoch']})")
    
    for epoch in range(start_epoch + 1, config['mae']['epochs'] + 1):
        # Train epoch (OneCycleLR steps inside train_epoch)
        use_onecycle = scheduler_type == 'onecycle'
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer, 
                              scheduler if use_onecycle else None, use_onecycle)
        
        # Step scheduler (except OneCycleLR which steps per batch)
        if scheduler is not None and not use_onecycle:
            if use_plateau:
                # ReduceLROnPlateau needs loss value
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{config['mae']['epochs']} - Loss: {avg_loss:.4f} - LR: {current_lr:.2e}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_lr', current_lr, epoch)
        
        # Save checkpoint with epoch number (for resume)
        checkpoint_path = checkpoint_dir / f'mae_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path.name}")
        
        # Save temp_best if this is the best so far
        improved = False
        if avg_loss < best_loss - early_stopping_min_delta:
            best_loss = avg_loss
            early_stopping_counter = 0  # Reset counter on improvement
            improved = True
            temp_best_path = checkpoint_dir / 'mae_temp_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, temp_best_path)
            print(f"New best loss: {best_loss:.4f} - Saved to {temp_best_path.name}")
        else:
            if early_stopping_enabled:
                early_stopping_counter += 1
                print(f"No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
        
        # Early stopping check
        if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] No improvement for {early_stopping_patience} epochs. Stopping training.")
            print(f"Best loss: {best_loss:.4f} at epoch {epoch - early_stopping_patience}")
            break
        
        # Delete old epoch checkpoints to save space (keep only last 2)
        epoch_files = sorted(checkpoint_dir.glob('mae_epoch_*.pth'), 
                            key=lambda x: int(x.stem.split('_')[-1]))
        if len(epoch_files) > 2:
            for old_file in epoch_files[:-2]:
                old_file.unlink()
                print(f"Deleted old checkpoint: {old_file.name}")
    
    # Training completed - finalize best models
    if early_stopping_enabled and early_stopping_counter >= early_stopping_patience:
        print(f"\n[SUCCESS] Training stopped early at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
    else:
        print(f"\n[SUCCESS] All {config['mae']['epochs']} epochs completed!")
    print(f"Finalizing best models...")
    
    # Rename temp_best to best (for phase check)
    temp_best_path = checkpoint_dir / 'mae_temp_best.pth'
    if temp_best_path.exists():
        best_checkpoint = torch.load(temp_best_path, map_location=device)
        
        # Save as mae_best.pth
        mae_best_path = checkpoint_dir / 'mae_best.pth'
        torch.save(best_checkpoint, mae_best_path)
        print(f"Saved {mae_best_path.name} (loss: {best_checkpoint['loss']:.4f}, epoch: {best_checkpoint['epoch']})")
        
        # Save encoder weights
        model.load_state_dict(best_checkpoint['model_state_dict'])
        encoder_path = checkpoint_dir / 'encoder_best.pth'
        model.save_encoder(str(encoder_path))
        print(f"Saved {encoder_path.name}")
        
        # Remove temp_best (no longer needed)
        temp_best_path.unlink()
        print(f"Removed temporary file: {temp_best_path.name}")
    else:
        print("[WARN] No temp_best found - this shouldn't happen")
    
    writer.close()
    print("\nMAE pretraining completed!")


if __name__ == '__main__':
    main()
