"""
AUROC Metrics Calculation Utilities

This module provides functions for calculating AUROC (Area Under ROC Curve)
for multi-label classification tasks.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Dict
import logging


def calculate_per_class_auroc(
    gt_labels: np.ndarray,
    predictions: np.ndarray,
    pathology_names: List[str],
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    Calculate AUROC for each class
    
    Args:
        gt_labels: Ground truth binary matrix (n_samples, n_classes)
        predictions: Prediction probabilities (n_samples, n_classes)
        pathology_names: List of pathology names
        logger: Optional logger instance
    
    Returns:
        Dictionary with per-class AUROC values
    """
    auroc_dict = {}
    
    for i, pathology in enumerate(pathology_names):
        try:
            y_true = gt_labels[:, i]
            y_pred = predictions[:, i]
            
            # Check if class has both positive and negative samples
            if len(np.unique(y_true)) < 2:
                auroc_dict[pathology] = np.nan
                if logger:
                    logger.warning(f"{pathology}: Only one class present, AUROC = NaN")
            else:
                auroc = roc_auc_score(y_true, y_pred)
                auroc_dict[pathology] = auroc
        except Exception as e:
            if logger:
                logger.error(f"Error calculating AUROC for {pathology}: {e}")
            auroc_dict[pathology] = np.nan
    
    return auroc_dict


def log_auroc_results(
    auroc_dict: Dict[str, float],
    model_name: str,
    logger: logging.Logger
) -> None:
    """
    Log AUROC results in a formatted table
    
    Args:
        auroc_dict: Dictionary with per-class AUROC values
        model_name: Name of the model (for logging)
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info(f"{model_name} - Per-Class AUROC")
    logger.info("="*60)
    
    valid_aurocs = [v for v in auroc_dict.values() if not np.isnan(v)]
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else np.nan
    
    logger.info(f"\n{'Pathology':<30} {'AUROC':>10}")
    logger.info("-" * 45)
    
    for pathology, auroc in auroc_dict.items():
        if not np.isnan(auroc):
            logger.info(f"{pathology:<30} {auroc:>10.3f}")
        else:
            logger.info(f"{pathology:<30} {'N/A':>10}")
    
    logger.info("-" * 45)
    logger.info(f"{'Mean AUROC':<30} {mean_auroc:>10.3f}")
    logger.info("")
