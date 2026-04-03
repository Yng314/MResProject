"""
Noise Rate Validation Script
End-to-end pipeline for estimating noise rate using cleanlab
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from cleanlab.multilabel_classification import rank
from cleanlab.filter import find_label_issues

# Import our utilities
from utils.label_utils import (
    filter_by_view_position,
    handle_uncertain_labels,
    convert_to_multilabel_format,
    get_label_statistics,
    validate_cleanlab_format,
    log_label_statistics
)
from utils.auroc_metrics import calculate_per_class_auroc
from utils.xrv_utils import (
    load_pretrained_model,
    get_ordered_pathology_list,
    create_dataloader,
    extract_predictions,
    validate_predictions
)


def setup_logging(log_dir: Path, log_level: str = 'INFO'):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'noise_validation_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from unified YAML file and merge common + torchxray settings
    """
    with open(config_path, 'r') as f:
        unified_config = yaml.safe_load(f)
    
    # Merge common and torchxray specific settings
    config = unified_config['common'].copy()
    
    # helper for deep merge
    def update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    update_recursive(config, unified_config['torchxray'])
    
    return config


def load_and_preprocess_data(config: dict, logger: logging.Logger, max_samples: int = None):
    """
    Load GT data and preprocess labels
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        max_samples: Maximum number of samples to use (for testing)
    
    Returns:
        DataFrame with preprocessed data
    """
    logger.info("="*60)
    logger.info("STEP 1: Load and Preprocess Data")
    logger.info("="*60)
    
    # Load metadata
    metadata_path = config['data']['gt_metadata_path']
    logger.info(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Filter by ViewPosition
    allowed_views = config['data']['allowed_view_positions']
    logger.info(f"Filtering for ViewPosition: {allowed_views}")
    df = filter_by_view_position(df, allowed_views)
    
    # Limit samples for testing
    if max_samples is not None and len(df) > max_samples:
        logger.info(f"Limiting to {max_samples} samples for testing")
        df = df.head(max_samples).copy()
    
    logger.info(f"Final dataset: {len(df)} samples\n")
    
    return df


def prepare_labels(df: pd.DataFrame, config: dict, logger: logging.Logger, label_type: str = 'gt'):
    """
    Prepare labels for cleanlab analysis
    
    Args:
        df: DataFrame with labels
        config: Configuration dictionary
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    
    Returns:
        Tuple of (labels_list, pathology_cols)
    """
    logger.info("="*60)
    logger.info(f"STEP 2: Prepare {label_type.upper()} Labels")
    logger.info("="*60)
    
    # Get pathology columns
    ordered_pathologies = get_ordered_pathology_list()
    
    if label_type == 'gt':
        pathology_cols = [f'gt_{p}' for p in ordered_pathologies]
    else:
        pathology_cols = ordered_pathologies
    
    # Check available columns
    available_cols = [col for col in pathology_cols if col in df.columns]
    logger.info(f"Available pathology columns: {len(available_cols)}/{len(pathology_cols)}")
    
    # Handle uncertain labels
    strategy = config['preprocessing']['uncertain_strategy']
    logger.info(f"Applying uncertain label strategy: {strategy}")
    df_processed = handle_uncertain_labels(df.copy(), available_cols, strategy)
    
    # Convert to cleanlab format
    logger.info("Converting to cleanlab multi-label format...")
    labels_list = convert_to_multilabel_format(df_processed, available_cols)
    
    # Show statistics
    log_label_statistics(df_processed, available_cols, logger, f"{label_type.upper()}")
    return labels_list, available_cols


def run_model_inference(df: pd.DataFrame, config: dict, logger: logging.Logger):
    """
    Run model inference to get predictions
    
    Args:
        df: DataFrame with image paths
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Mapped predictions array (n_samples, 12)
    """
    logger.info("="*60)
    logger.info("STEP 3: Model Inference")
    logger.info("="*60)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['model']['device'] != 'auto': 
         device = config['model']['device']
         
    logger.info(f"Using device: {device}")
    model = load_pretrained_model(device)
    
    # Create dataloader
    image_paths = df['image_path'].tolist()
    base_path = config['data']['gt_image_base_path']
    batch_size = config['inference']['batch_size']
    num_workers = config['inference']['num_workers']
    
    logger.info(f"Creating dataloader: batch_size={batch_size}, num_workers={num_workers}")
    dataloader = create_dataloader(
        image_paths,
        base_path=base_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Extract predictions
    full_preds, mapped_preds = extract_predictions(model, dataloader, device)
    
    # Validate predictions
    ordered_pathologies = get_ordered_pathology_list()
    validate_predictions(mapped_preds, ordered_pathologies)
    
    logger.info("")
    return mapped_preds


def analyze_per_class_noise(
    labels_list,
    pred_probs: np.ndarray,
    pathology_names,
    logger: logging.Logger,
    label_type: str = 'gt'
):
    """
    Analyze noise rate for each individual pathology class
    
    Args:
        labels_list: Ground truth labels in cleanlab format (list of lists)
        pred_probs: Prediction probabilities (n_samples, n_classes)
        pathology_names: List of pathology names
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    
    Returns:
        Dictionary with per-class noise rates
    """
    logger.info("="*60)
    logger.info(f"Per-Class Noise Analysis ({label_type.upper()} labels)")
    logger.info("="*60)
    
    n_samples = len(labels_list)
    n_classes = len(pathology_names)
    
    per_class_results = {}
    
    logger.info(f"\nAnalyzing {n_classes} pathology classes individually...")
    
    for class_idx, pathology in enumerate(pathology_names):
        # Convert multi-label to binary for this class
        binary_labels = np.array([1 if class_idx in sample_labels else 0 
                                  for sample_labels in labels_list])
        
        # Get prediction probabilities for this class
        class_probs = pred_probs[:, class_idx]
        
        # Create pred_probs in format expected by cleanlab (n_samples, 2)
        # [prob_negative, prob_positive]
        binary_pred_probs = np.column_stack([1 - class_probs, class_probs])
        
        try:
            # Find label issues using binary classification
            # Returns INDICES of issues, not boolean array
            issue_indices = find_label_issues(
                labels=binary_labels,
                pred_probs=binary_pred_probs,
                return_indices_ranked_by='self_confidence'
            )
            
            # Convert indices to boolean array
            label_issues = np.zeros(n_samples, dtype=bool)
            label_issues[issue_indices] = True
            
            num_issues = label_issues.sum()
            issue_rate = num_issues / n_samples * 100
            
            # Count positive/negative samples
            num_positive = binary_labels.sum()
            num_negative = n_samples - num_positive
            
            # Issues in positive vs negative
            issues_in_positive = label_issues[binary_labels == 1].sum()
            issues_in_negative = label_issues[binary_labels == 0].sum()
            
            per_class_results[pathology] = {
                'num_issues': int(num_issues),
                'issue_rate': float(issue_rate),
                'num_positive': int(num_positive),
                'num_negative': int(num_negative),
                'issues_in_positive': int(issues_in_positive),
                'issues_in_negative': int(issues_in_negative),
                'label_issues': label_issues
            }
            
        except Exception as e:
            logger.warning(f"  {pathology}: Analysis failed - {e}")
            per_class_results[pathology] = None
    
    # Display results
    logger.info(f"\nPer-class noise rates:")
    logger.info(f"{'Pathology':<30} {'Positive':>8} {'Negative':>8} {'Issues':>7} {'Rate':>7}")
    logger.info("-" * 70)
    
    for pathology in pathology_names:
        result = per_class_results.get(pathology)
        if result:
            logger.info(
                f"{pathology:<30} "
                f"{result['num_positive']:>8} "
                f"{result['num_negative']:>8} "
                f"{result['num_issues']:>7} "
                f"{result['issue_rate']:>6.1f}%"
            )
    
    logger.info("")
    return per_class_results


def main():
    """Main function"""
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    default_config = script_dir / 'config' / 'unified_config.yaml'
    
    parser = argparse.ArgumentParser(description='Noise Rate Validation')
    parser.add_argument('--config', type=str, default=str(default_config),
                       help='Path to configuration file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (for testing)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(Path(config['logging']['log_dir']), args.log_level)
    
    logger.info("\n" + "="*60)
    logger.info("NOISE RATE VALIDATION - START")
    logger.info("="*60 + "\n")
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(config, logger, args.max_samples)
        
        # Step 2a: Prepare GT labels
        gt_labels, gt_cols = prepare_labels(df, config, logger, label_type='gt')
        
        # Step 2b: Prepare pseudo labels
        pseudo_labels, pseudo_cols = prepare_labels(df, config, logger, label_type='pseudo')
        
        # Step 3: Model inference
        pred_probs = run_model_inference(df, config, logger)
        
        # Step 4: Per-class noise analysis (GT and Pseudo)
        ordered_pathologies = get_ordered_pathology_list()
        
        # Calculate AUROCs (using GT labels as ground truth)
        # Re-get GT labels in binary format for AUROC calculation
        gt_cols = [f'gt_{p}' for p in ordered_pathologies]
        available_gt_cols = [col for col in gt_cols if col in df.columns]
        # We need to filter df again or assume prepare_labels didn't change row order/count
        # prepare_labels returns labels_list, but we need binary matrix for AUROC
        # So let's re-process df locally to get binary matrix
        strategy = config['preprocessing']['uncertain_strategy']
        df_gt_for_auroc = handle_uncertain_labels(df.copy(), available_gt_cols, strategy)
        gt_labels_binary = df_gt_for_auroc[available_gt_cols].values.astype(np.float32)
        
        # Calculate AUROC
        logger.info("\nCalculating AUROC...")
        auroc_dict = calculate_per_class_auroc(gt_labels_binary, pred_probs, ordered_pathologies, logger)
        
        logger.info("\n" + "="*60)
        logger.info("PER-CLASS NOISE ANALYSIS")
        logger.info("="*60 + "\n")
        
        gt_per_class = analyze_per_class_noise(gt_labels, pred_probs, ordered_pathologies, logger, label_type='gt')
        pseudo_per_class = analyze_per_class_noise(pseudo_labels, pred_probs, ordered_pathologies, logger, label_type='pseudo')
        
        # Step 5: Compare per-class results
        logger.info("\n" + "="*60)
        logger.info("Per-Class Noise Rate Comparison")
        logger.info("="*60)
        logger.info(f"\n{'Pathology':<30} {'GT Rate':>10} {'Pseudo Rate':>12} {'Diff':>8}")
        logger.info("-" * 70)
        
        for pathology in ordered_pathologies:
            gt_res = gt_per_class.get(pathology)
            pseudo_res = pseudo_per_class.get(pathology)
            
            if gt_res and pseudo_res:
                diff = pseudo_res['issue_rate'] - gt_res['issue_rate']
                logger.info(
                    f"{pathology:<30} "
                    f"{gt_res['issue_rate']:>9.1f}% "
                    f"{pseudo_res['issue_rate']:>11.1f}% "
                    f"{diff:>+7.1f}%"
                )
        logger.info("")
        
        # Step 6: Save results
        logger.info("="*60)
        logger.info("Saving Results")
        logger.info("="*60)
        
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save per-class results as CSV
        dicom_ids = df['dicom_id'].values
        
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
            gt_res = gt_per_class.get(pathology)
            pseudo_res = pseudo_per_class.get(pathology)
            
            # Save issue IDs to TXT buffer
            if gt_res and len(dicom_ids[gt_res['label_issues']]) > 0:
                issue_lines.append(f"=== GT - {pathology} ({gt_res['num_issues']} issues) ===")
                issue_lines.extend(dicom_ids[gt_res['label_issues']])
                issue_lines.append("")
                
            if pseudo_res and len(dicom_ids[pseudo_res['label_issues']]) > 0:
                issue_lines.append(f"=== Pseudo - {pathology} ({pseudo_res['num_issues']} issues) ===")
                issue_lines.extend(dicom_ids[pseudo_res['label_issues']])
                issue_lines.append("")
            
            if gt_res and pseudo_res:
                results_data.append({
                    'Pathology': pathology,
                    'AUROC': auroc_dict.get(pathology, np.nan),
                    'GT_Positive': gt_res['num_positive'],
                    'GT_Negative': gt_res['num_negative'],
                    'GT_Issues': gt_res['num_issues'],
                    'GT_Noise_Rate': f"{gt_res['issue_rate']:.2f}%",
                    'GT_Pos_Rate': f"{gt_res['num_positive'] / (gt_res['num_positive'] + gt_res['num_negative']) * 100:.2f}%",
                    'Pseudo_Issues': pseudo_res['num_issues'],
                    'Pseudo_Noise_Rate': f"{pseudo_res['issue_rate']:.2f}%",
                    'Pseudo_Pos_Rate': f"{pseudo_res['num_positive'] / (pseudo_res['num_positive'] + pseudo_res['num_negative']) * 100:.2f}%",
                    'Diff': f"{(pseudo_res['issue_rate'] - gt_res['issue_rate']):.2f}%"
                })
        
        # Save consolidated issues
        if issue_lines:
            issue_file = results_dir / f'issues_{timestamp}.txt'
            # Save issue IDs
            with open(issue_file, 'w') as f:
                f.write(f"# Strategy: {strategy_desc}\n\n")
                f.write('\n'.join(issue_lines))
            logger.info(f"Issue IDs saved to: {issue_file}")
        
        results_df = pd.DataFrame(results_data)
        csv_path = results_dir / f'perclass_noise_rates_{timestamp}.csv'
        with open(csv_path, 'w') as f:
            f.write(f"# Strategy: {strategy_desc}\n")
            results_df.to_csv(f, index=False)
        logger.info(f"Per-class results saved to: {csv_path}")
        
        # Save predictions and per-class issues
        npz_path = results_dir / f'noise_analysis_{timestamp}.npz'
        np.savez(
            npz_path,
            pred_probs=pred_probs,
            **{f'gt_{p}_issues': gt_per_class[p]['label_issues'] 
               for p in ordered_pathologies if p in gt_per_class and gt_per_class[p]},
            **{f'pseudo_{p}_issues': pseudo_per_class[p]['label_issues'] 
               for p in ordered_pathologies if p in pseudo_per_class and pseudo_per_class[p]}
        )
        logger.info(f"Detailed data saved to: {npz_path}")
        logger.info("")
        
        logger.info("\n" + "="*60)
        logger.info("NOISE RATE VALIDATION - COMPLETE")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
