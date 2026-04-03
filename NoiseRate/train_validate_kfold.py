"""
K-Fold Training and Validation Script

Train lightweight CNN with k-fold CV on both GT and Pseudo labels,
collect out-of-fold predictions, and perform noise rate analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import our modules
from models.lightweight_cnn import LightweightCNN
from utils.label_utils import (
    filter_by_view_position,
    handle_uncertain_labels,
    convert_to_multilabel_format,
    get_label_statistics,
    log_label_statistics
)
from utils.training_utils import (
    ChestXrayDataset,
    get_train_transforms,
    get_val_transforms,
    compute_class_weights,
    EarlyStopping,
    train_one_epoch,
    validate
)
from utils.noise_analysis import (
    run_cleanlab_multilabel_analysis,
    analyze_per_class_noise,
    compare_noise_results
)
from utils.auroc_metrics import (
    calculate_per_class_auroc,
    log_auroc_results
)
from utils.xrv_utils import get_ordered_pathology_list


def setup_logging(log_dir: Path, label_type: str):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'kfold_{label_type}_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger(f'kfold_{label_type}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str, section: str = None) -> dict:
    """
    Load configuration and merge sections for K-Fold training
    
    Args:
        config_path: Path to unified config file
        section: 'gt' or 'pseudo' to load specific kfold settings
        
    Returns:
        Merged configuration dict
    """
    with open(config_path, 'r') as f:
        unified_config = yaml.safe_load(f)
        
    # Helper for deep merge
    def update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    # Base config: common settings
    config = unified_config['common'].copy()
    
    # Merge k-fold common settings
    config['kfold'] = unified_config['kfold']['common']['kfold_settings']
    
    # Merge other k-fold common settings into top-level or sub-sections as needed
    # (data, model, training, augmentation, preprocessing, loss)
    # The structure in unified_config['kfold']['common'] matches the needed structure
    # except for 'kfold_settings' which we just handled
    
    common_kfold = unified_config['kfold']['common'].copy()
    if 'kfold_settings' in common_kfold:
        del common_kfold['kfold_settings']
        
    update_recursive(config, common_kfold)
    
    # Merge specific section (gt or pseudo)
    if section:
        if section not in unified_config['kfold']:
             raise ValueError(f"Section {section} not found in kfold config")
        update_recursive(config, unified_config['kfold'][section])
        
    return config


def prepare_data(config, logger, use_pseudo=False):
    """
    Load and prepare data for training
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        use_pseudo: If True, use pseudo labels; if False, use GT labels
    
    Returns:
        df_processed: Processed dataframe
        image_paths: List of image paths
        labels_binary: Binary label matrix (n_samples, n_classes)
    """
    # Load metadata
    metadata_path = config['data']['gt_metadata_path']
    logger.info(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Filter by ViewPosition
    allowed_views = config['data']['allowed_view_positions']
    logger.info(f"Filtering for ViewPosition: {allowed_views}")
    df = filter_by_view_position(df, allowed_views)
    logger.info(f"After filtering: {len(df)} samples\n")
    
    # Get label columns - use only CheXpert 12 pathologies
    ordered_pathologies = get_ordered_pathology_list()
    
    if use_pseudo:
        logger.info("Using PSEUDO labels for training")
        pathology_cols = ordered_pathologies  # Pseudo labels don't have 'gt_' prefix
    else:
        logger.info("Using GT labels for training")
        pathology_cols = [f'gt_{p}' for p in ordered_pathologies]
    
    # Filter to only existing columns
    pathology_cols = [col for col in pathology_cols if col in df.columns]
    logger.info(f"Using {len(pathology_cols)} CheXpert pathology columns")
    
    # Handle uncertain labels
    strategy = config['preprocessing']['uncertain_strategy']
    logger.info(f"Applying uncertain label strategy: {strategy}")
    df_processed = handle_uncertain_labels(df.copy(), pathology_cols, strategy)
    
    # Log detailed statistics
    log_label_statistics(df_processed, pathology_cols, logger, "Processed Data")
    
    # Convert to binary matrix directly from DataFrame
    labels_binary = df_processed[pathology_cols].values.astype(np.float32)
    logger.info(f"Labels shape: {labels_binary.shape}\n")
    
    # Get image paths
    image_paths = df_processed['image_path'].tolist()
    
    return df_processed, image_paths, labels_binary


def train_kfold(config, logger, label_type='gt'):
    """
    Train model using k-fold cross-validation
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    
    Returns:
        oof_predictions: Out-of-fold predictions (n_samples, n_classes)
        df_processed: Processed dataframe with labels
    """
    logger.info("\n" + "="*60)
    logger.info(f"K-FOLD TRAINING - {label_type.upper()} LABELS")
    logger.info("="*60 + "\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")
    
    # Load and prepare data
    logger.info("="*60)
    logger.info("Loading and Preparing Data")
    logger.info("="*60)
    use_pseudo = (label_type == 'pseudo')
    df_processed, image_paths, labels_binary = prepare_data(config, logger, use_pseudo)
    
    # Initialize k-fold
    logger.info("="*60)
    logger.info("K-Fold Split")
    logger.info("="*60)
    n_splits = config['kfold']['n_splits']
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config['kfold']['shuffle'],
        random_state=config['kfold']['random_state']
    )
    
    # Use first label for stratification
    stratify_labels = labels_binary[:, 0].astype(int)
    logger.info(f"K-fold: {n_splits} splits, stratified by first label\n")
    
    # Storage for OOF predictions
    oof_predictions = np.zeros_like(labels_binary, dtype=np.float32)
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(image_paths)), stratify_labels)):
        logger.info("="*60)
        logger.info(f"Fold {fold + 1}/{n_splits}")
        logger.info("="*60)
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}\n")
        
        # Create datasets
        train_transform = get_train_transforms(config)
        val_transform = get_val_transforms(config)
        
        base_path = config['data']['gt_image_base_path']
        
        full_dataset_train = ChestXrayDataset(image_paths, labels_binary, base_path=base_path, transform=train_transform)
        full_dataset_val = ChestXrayDataset(image_paths, labels_binary, base_path=base_path, transform=val_transform)
        
        train_dataset = Subset(full_dataset_train, train_idx)
        val_dataset = Subset(full_dataset_val, val_idx)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )
        
        # Model
        num_classes = labels_binary.shape[1]
        model = LightweightCNN(
            num_classes=num_classes,
            input_channels=config['model']['input_channels'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Loss function
        class_weights = compute_class_weights(labels_binary[train_idx])
        pos_weight = torch.FloatTensor(class_weights).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(config['training']['epochs_per_fold']):
            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_preds, _ = validate(model, val_loader, criterion, device)
            
            # Scheduler step
            scheduler.step()
            
            # Log
            logger.info(f"Epoch {epoch + 1}/{config['training']['epochs_per_fold']}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss:   {val_loss:.4f}")
            logger.info(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  ** New best val loss: {best_val_loss:.4f}")
                
                # Save model
                model_dir = Path(config['output']['models_dir'])
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f'fold_{fold}_{label_type}_model.pth'
                torch.save(model.state_dict(), model_path)
            
            logger.info("")
            
            # Early stopping check
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}\n")
                break
        
        # Load best model for OOF predictions
        model.load_state_dict(torch.load(model_dir / f'fold_{fold}_{label_type}_model.pth'))
        _, oof_preds_fold, _ = validate(model, val_loader, criterion, device)
        oof_predictions[val_idx] = oof_preds_fold
        
        logger.info(f"Fold {fold + 1} complete. Best val loss: {best_val_loss:.4f}\n")
    
    return oof_predictions, df_processed


def run_validation(oof_predictions, df_processed, config, logger, label_type='gt'):
    """
    Run validation: AUROC calculation and noise analysis (per-class only)
    
    Args:
        oof_predictions: Out-of-fold predictions
        df_processed: Processed dataframe
        config: Configuration dictionary
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    """
    logger.info("\n" + "="*60)
    logger.info(f"VALIDATION - {label_type.upper()} LABELS")
    logger.info("="*60 + "\n")
    
    # Get pathology names
    ordered_pathologies = get_ordered_pathology_list()
    
    # Prepare GT labels for AUROC (always use GT as ground truth)
    gt_cols = [f'gt_{p}' for p in ordered_pathologies]
    gt_cols = [col for col in gt_cols if col in df_processed.columns]
    df_gt = handle_uncertain_labels(df_processed.copy(), gt_cols, 'u_zeros')
    gt_labels_binary = df_gt[gt_cols].values.astype(np.float32)
    gt_labels_list = convert_to_multilabel_format(df_gt, gt_cols)
    
    # Calculate AUROC (using GT as ground truth)
    logger.info("="*60)
    logger.info(f"Per-Class AUROC ({label_type.upper()} model)")
    logger.info("="*60)
    auroc_dict = calculate_per_class_auroc(gt_labels_binary, oof_predictions, ordered_pathologies, logger)
    log_auroc_results(auroc_dict, f"CNN-{label_type.upper()}", logger)
    
    # Per-class noise analysis
    logger.info("="*60)
    logger.info(f"Per-Class Noise Analysis")
    logger.info("="*60  + "\n")
    
    if label_type == 'gt':
        # For GT model: analyze ONLY GT labels (trained on GT, evaluate on GT)
        gt_per_class = analyze_per_class_noise(
            gt_labels_list, oof_predictions, ordered_pathologies, logger, 'gt'
        )
        
        # Save results to CSV
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        dicom_ids = df_processed['dicom_id'].values
        
        results_data = []
        issue_lines = []
        
        # Get strategy description
        strategy = config['preprocessing']['uncertain_strategy']
        if strategy == 'u_ones':
            strategy_desc = "Uncertain->1, Missing->0"
        elif strategy == 'u_zeros':
            strategy_desc = "Uncertain->0, Missing->0"
        elif strategy == 'u_ignore':
            strategy_desc = "Uncertain->Ignored, Missing->0"
        else:
            strategy_desc = strategy
        
        for pathology in ordered_pathologies:
            auroc = auroc_dict.get(pathology, np.nan)
            gt_res = gt_per_class.get(pathology)
            if gt_res:
                # Collect issues
                issue_mask = gt_res['label_issues']
                issue_ids = dicom_ids[issue_mask]
                if len(issue_ids) > 0:
                    issue_lines.append(f"=== {pathology} ({len(issue_ids)} issues) ===")
                    issue_lines.extend(issue_ids)
                    issue_lines.append("") # Empty line separator
                
                results_data.append({
                    'Pathology': pathology,
                    'AUROC': auroc,
                    'GT_Noise_Rate': f"{gt_res['issue_rate']:.2f}%",
                    'GT_Issues': gt_res['num_issues'],
                    'Positive_Count': gt_res['num_positive'],
                    'Negative_Count': gt_res['num_negative'],
                    'Positive_Rate': f"{gt_res['num_positive'] / (gt_res['num_positive'] + gt_res['num_negative']) * 100:.2f}%"
                })
        
        # Save consolidated issues
        if issue_lines:
            issue_file = results_dir / f'issues_gt_{timestamp}.txt'
            with open(issue_file, 'w') as f:
                f.write(f"# Strategy: {strategy_desc}\n\n")
                f.write('\n'.join(issue_lines))
            logger.info(f"Issue IDs saved to: {issue_file}")
        
        results_df = pd.DataFrame(results_data)
        csv_path = results_dir / f'cnn_gt_results_{timestamp}.csv'
        with open(csv_path, 'w') as f:
            f.write(f"# Strategy: {strategy_desc}\n")
            results_df.to_csv(f, index=False)
        logger.info(f"Results saved to: {csv_path}\n")
        
    else:  # pseudo
        # For Pseudo model: analyze only Pseudo labels
        pseudo_cols = ordered_pathologies
        pseudo_cols = [col for col in pseudo_cols if col in df_processed.columns]
        df_pseudo = handle_uncertain_labels(df_processed.copy(), pseudo_cols, 'u_zeros')
        pseudo_labels_list = convert_to_multilabel_format(df_pseudo, pseudo_cols)
        
        pseudo_per_class = analyze_per_class_noise(
            pseudo_labels_list, oof_predictions, ordered_pathologies, logger, 'pseudo'
        )
        
        # Save results to CSV and issues to TXT
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        dicom_ids = df_processed['dicom_id'].values
        
        results_data = []
        issue_lines = []
        
        # Get strategy description
        strategy = config['preprocessing']['uncertain_strategy']
        if strategy == 'u_ones':
            strategy_desc = "Uncertain->1, Missing->0"
        elif strategy == 'u_zeros':
            strategy_desc = "Uncertain->0, Missing->0"
        elif strategy == 'u_ignore':
            strategy_desc = "Uncertain->Ignored, Missing->0"
        else:
            strategy_desc = strategy
        
        for pathology in ordered_pathologies:
            auroc = auroc_dict.get(pathology, np.nan)
            pseudo_res = pseudo_per_class.get(pathology)
            if pseudo_res:
                # Collect issues
                issue_mask = pseudo_res['label_issues']
                issue_ids = dicom_ids[issue_mask]
                if len(issue_ids) > 0:
                    issue_lines.append(f"=== {pathology} ({len(issue_ids)} issues) ===")
                    issue_lines.extend(issue_ids)
                    issue_lines.append("")
                    
                results_data.append({
                    'Pathology': pathology,
                    'AUROC': auroc,
                    'Pseudo_Noise_Rate': f"{pseudo_res['issue_rate']:.2f}%",
                    'Pseudo_Issues': pseudo_res['num_issues'],
                    'Positive_Count': pseudo_res['num_positive'],
                    'Negative_Count': pseudo_res['num_negative'],
                    'Positive_Rate': f"{pseudo_res['num_positive'] / (pseudo_res['num_positive'] + pseudo_res['num_negative']) * 100:.2f}%"
                })
        
        # Save consolidated issues
        if issue_lines:
            issue_file = results_dir / f'issues_pseudo_{timestamp}.txt'
            with open(issue_file, 'w') as f:
                f.write(f"# Strategy: {strategy_desc}\n\n")
                f.write('\n'.join(issue_lines))
            logger.info(f"Issue IDs saved to: {issue_file}")
        
        results_df = pd.DataFrame(results_data)
        csv_path = results_dir / f'cnn_pseudo_results_{timestamp}.csv'
        with open(csv_path, 'w') as f:
            f.write(f"# Strategy: {strategy_desc}\n")
            results_df.to_csv(f, index=False)
        logger.info(f"Results saved to: {csv_path}")



def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    default_config = script_dir / 'config' / 'unified_config.yaml'
    
    parser = argparse.ArgumentParser(description='K-Fold Training and Validation')
    parser.add_argument('--config', type=str,
                       default=str(default_config),
                       help='Path to unified configuration file')
    
    args = parser.parse_args()
    
    # ===== TRAIN AND VALIDATE GT MODEL =====
    print("\n" + "="*80)
    print("STEP 1: TRAINING WITH GT LABELS")
    print("="*80)
    
    config_gt = load_config(args.config, section='gt')
    logger_gt = setup_logging(Path(config_gt['output']['logs_dir']), 'gt')
    
    # Train
    oof_predictions_gt, df_processed_gt = train_kfold(config_gt, logger_gt, label_type='gt')
    
    # Save OOF predictions
    results_dir_gt = Path(config_gt['output']['results_dir'])
    results_dir_gt.mkdir(parents=True, exist_ok=True)
    oof_path_gt = results_dir_gt / config_gt['output']['oof_predictions_file']
    np.save(oof_path_gt, oof_predictions_gt)
    logger_gt.info(f"OOF predictions saved to: {oof_path_gt}")
    logger_gt.info(f"Shape: {oof_predictions_gt.shape}\n")
    
    # Validate
    run_validation(oof_predictions_gt, df_processed_gt, config_gt, logger_gt, label_type='gt')
    
    logger_gt.info("\n" + "="*60)
    logger_gt.info("GT TRAINING AND VALIDATION - COMPLETE")
    logger_gt.info("="*60 + "\n")
    
    # ===== TRAIN AND VALIDATE PSEUDO MODEL =====
    print("\n" + "="*80)
    print("STEP 2: TRAINING WITH PSEUDO LABELS")
    print("="*80)
    
    config_pseudo = load_config(args.config, section='pseudo')
    logger_pseudo = setup_logging(Path(config_pseudo['output']['logs_dir']), 'pseudo')
    
    # Train
    oof_predictions_pseudo, df_processed_pseudo = train_kfold(config_pseudo, logger_pseudo, label_type='pseudo')
    
    # Save OOF predictions
    results_dir_pseudo = Path(config_pseudo['output']['results_dir'])
    results_dir_pseudo.mkdir(parents=True, exist_ok=True)
    oof_path_pseudo = results_dir_pseudo / config_pseudo['output']['oof_predictions_file']
    np.save(oof_path_pseudo, oof_predictions_pseudo)
    logger_pseudo.info(f"OOF predictions saved to: {oof_path_pseudo}")
    logger_pseudo.info(f"Shape: {oof_predictions_pseudo.shape}\n")
    
    # Validate
    run_validation(oof_predictions_pseudo, df_processed_pseudo, config_pseudo, logger_pseudo, label_type='pseudo')
    
    logger_pseudo.info("\n" + "="*60)
    logger_pseudo.info("PSEUDO TRAINING AND VALIDATION - COMPLETE")
    logger_pseudo.info("="*60 + "\n")
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("ALL TRAINING AND VALIDATION COMPLETE")
    print("="*80)
    print(f"\nGT OOF Predictions:     {oof_path_gt}")
    print(f"Pseudo OOF Predictions: {oof_path_pseudo}")
    print("\nCheck log files for detailed results.")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
