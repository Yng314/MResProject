"""
Noise Analysis Utilities for Cleanlab Multi-label Classification

This module provides functions for analyzing label noise using Cleanlab,
specifically designed for multi-label chest X-ray classification.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from cleanlab.multilabel_classification import rank
from cleanlab.filter import find_label_issues


def run_cleanlab_multilabel_analysis(
    labels_list: List[List[int]],
    pred_probs: np.ndarray,
    pathology_names: List[str],
    logger: logging.Logger,
    label_type: str = 'gt'
) -> Dict:
    """
    Run cleanlab noise detection for multi-label classification
    
    Args:
        labels_list: Ground truth labels in cleanlab format (list of lists)
        pred_probs: Prediction probabilities (n_samples, n_classes)
        pathology_names: List of pathology names
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("="*60)
    logger.info(f"Cleanlab Analysis ({label_type.upper()} labels)")
    logger.info("="*60)
    
    # Validate format
    logger.info("Validating cleanlab format...")
    n_samples = len(labels_list)
    n_classes = pred_probs.shape[1]
    logger.info(f"  Samples: {n_samples}, Classes: {n_classes}")
    
    # Use multilabel_classification.rank for multi-label classification
    logger.info("\nRunning cleanlab.multilabel_classification.rank for multi-label...")
    try:
        # Get label quality scores
        label_quality_scores = rank.get_label_quality_scores(
            labels=labels_list,
            pred_probs=pred_probs
        )
        
        # Threshold for identifying label issues
        threshold = np.median(label_quality_scores) - 1.0 * np.std(label_quality_scores)
        label_issues = label_quality_scores < threshold
        
        num_issues = label_issues.sum()
        issue_rate = num_issues / len(labels_list) * 100
        
        logger.info(f"\nCleanlab analysis complete!")
        logger.info(f"  Total samples: {len(labels_list)}")
        logger.info(f"  Quality score threshold: {threshold:.4f}")
        logger.info(f"  Label issues detected: {num_issues} ({issue_rate:.2f}%)")
        
        results = {
            'label_type': label_type,
            'total_samples': len(labels_list),
            'num_issues': int(num_issues),
            'issue_rate': float(issue_rate),
            'label_issues': label_issues,
            'label_scores': label_quality_scores,
            'threshold': float(threshold),
            'issue_indices': np.where(label_issues)[0]
        }
        
        logger.info("")
        return results
        
    except Exception as e:
        logger.error(f"Cleanlab analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_per_class_noise(
    labels_list: List[List[int]],
    pred_probs: np.ndarray,
    pathology_names: List[str],
    logger: logging.Logger,
    label_type: str = 'gt'
) -> Dict[str, Dict]:
    """
    Analyze noise rate for each pathology class individually using binary classification
    
    Args:
        labels_list: Multi-label format labels
        pred_probs: Prediction probabilities (n_samples, n_classes)
        pathology_names: List of pathology names
        logger: Logger instance
        label_type: 'gt' or 'pseudo'
    
    Returns:
        Dictionary with per-class results
    """
    logger.info("="*60)
    logger.info(f"Per-Class Noise Analysis ({label_type.upper()} labels)")
    logger.info("="*60)
    
    n_samples = len(labels_list)
    num_classes = len(pathology_names)
    
    logger.info(f"\nAnalyzing {num_classes} pathology classes individually...")
    
    per_class_results = {}
    
    for class_idx, pathology in enumerate(pathology_names):
        try:
            # Convert multi-label to binary for this class
            binary_labels = np.array([1 if class_idx in sample_labels else 0 
                                      for sample_labels in labels_list])
            
            # Get probabilities for this class
            class_probs = pred_probs[:, class_idx]
            
            # Create binary pred_probs format for cleanlab
            binary_pred_probs = np.column_stack([1 - class_probs, class_probs])
            
            # Find label issues using binary classification
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
                'num_positive': int(num_positive),
                'num_negative': int(num_negative),
                'num_issues': int(num_issues),
                'issue_rate': float(issue_rate),
                'issues_in_positive': int(issues_in_positive),
                'issues_in_negative': int(issues_in_negative),
                'label_issues': label_issues
            }
            
        except Exception as e:
            logger.warning(f"  {pathology}: Analysis failed - {e}")
            per_class_results[pathology] = None
    
    # Print summary table
    logger.info(f"\nPer-class noise rates:")
    logger.info(f"{'Pathology':<30} {'Positive':>8} {'Negative':>8}  {'Issues':>6}    {'Rate':>5}")
    logger.info("-" * 70)
    
    for pathology in pathology_names:
        result = per_class_results.get(pathology)
        if result:
            logger.info(
                f"{pathology:<30} {result['num_positive']:>8} {result['num_negative']:>8} "
                f"{result['num_issues']:>6}   {result['issue_rate']:>5.1f}%"
            )
    
    logger.info("")
    return per_class_results


def compare_noise_results(
    gt_results: Dict,
    pseudo_results: Dict,
    gt_per_class: Dict,
    pseudo_per_class: Dict,
    logger: logging.Logger
) -> None:
    """
    Compare GT vs Pseudo label noise rates and log results
    
    Args:
        gt_results: Overall GT analysis results
        pseudo_results: Overall Pseudo analysis results
        gt_per_class: Per-class GT results
        pseudo_per_class: Per-class Pseudo results
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("Noise Rate Comparison")
    logger.info("="*60)
    
    logger.info("\nOverall Noise Rate:")
    logger.info(f"  GT labels:     {gt_results['issue_rate']:.2f}% ({gt_results['num_issues']}/{gt_results['total_samples']})")
    logger.info(f"  Pseudo labels: {pseudo_results['issue_rate']:.2f}% ({pseudo_results['num_issues']}/{pseudo_results['total_samples']})")
    
    rate_diff = pseudo_results['issue_rate'] - gt_results['issue_rate']
    logger.info(f"\n  Difference: {rate_diff:.2f}% (Pseudo - GT)")
    
    # Per-class comparison
    logger.info("\n" + "="*60)
    logger.info("Per-Class Noise Rate Comparison")
    logger.info("="*60)
    logger.info(f"\n{'Pathology':<30} {'GT Rate':>10} {'Pseudo Rate':>12} {'Diff':>8}")
    logger.info("-" * 70)
    
    for pathology in sorted(gt_per_class.keys()):
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
