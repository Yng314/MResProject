"""
Estimate Label Noise Rate using Confident Learning

This script computes estimated noise rates for multi-label classification.
It uses the confident joint matrix from cleanlab to estimate how much label noise
exists in the dataset.

Usage:
    # After WSL training (need val_predictions.npz)
    python estimate_noise_rate.py --config configs/config.yaml
    
    # Or specify experiment directly
    python estimate_noise_rate.py --experiment outputs/my_experiment
"""
import os
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Cleanlab imports
from cleanlab.count import compute_confident_joint, estimate_latent


def load_predictions(config: dict, use_test: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions and labels from WSL training output
    
    Args:
        config: Configuration dict
        use_test: If True, load test_predictions.npz instead of val_predictions.npz
    
    Returns:
        preds: (N, num_labels) predicted probabilities for positive class
        labels: (N, num_labels) binary labels (0 or 1, with NaN for missing)
        indices: (N,) sample indices
    """
    checkpoint_dir = config['wsl']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir):
        if 'output_dir' in config.get('pipeline', {}):
            checkpoint_dir = os.path.join(config['pipeline']['output_dir'], checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Select prediction file
    pred_filename = 'test_predictions.npz' if use_test else 'val_predictions.npz'
    pred_file = checkpoint_dir / pred_filename
    
    if not pred_file.exists():
        raise FileNotFoundError(
            f"Predictions not found at {pred_file}.\n"
            f"Run {'generate_test_predictions.py' if use_test else 'train_wsl.py'} first to generate predictions."
        )
    
    print(f"Loading predictions from: {pred_file}")
    data = np.load(pred_file)
    preds = data['preds']
    labels = data['labels']
    indices = data['indices']
    
    # Handle multi-class predictions (N, num_labels, num_classes_per_label)
    # Convert to binary probabilities
    if len(preds.shape) == 3:
        print(f"Detected multi-class predictions with shape {preds.shape}")
        # Apply softmax to convert logits to probabilities
        # preds shape: (N, num_labels, 4) where 4 = [negative, positive, uncertain, unlabeled]
        preds_softmax = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)
        
        # Binary mapping: P(positive) = P(class1) + P(class2) (positive + uncertain)
        # P(negative) = P(class0)
        preds_binary = preds_softmax[:, :, 1] + preds_softmax[:, :, 2]  # positive + uncertain
        preds = preds_binary
        print(f"Converted to binary probabilities (positive+uncertain): shape {preds.shape}")
    
    # Convert labels to binary: positive(1) + uncertain(-1) → 1, negative(0) → 0, NaN stays NaN
    # Original labels: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), NaN (missing)
    labels_binary = labels.copy()
    labels_binary[labels == -1.0] = 1.0  # uncertain → positive (1)
    # 0.0 stays 0.0, 1.0 stays 1.0, NaN stays NaN
    print(f"Converted labels to binary: positive(1)+uncertain(-1)→1, negative(0)→0")
    
    return preds, labels_binary, indices


def compute_confident_joint_for_label(
    preds: np.ndarray, 
    labels: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Compute confident joint matrix for a single binary label
    
    Args:
        preds: (N,) predicted probabilities for positive class
        labels: (N,) binary labels (0 or 1)
    
    Returns:
        confident_joint: (2, 2) confident joint matrix
        noise_rate: Estimated overall noise rate
        flip_rate_0_to_1: Rate of 0 labels that are actually 1
        flip_rate_1_to_0: Rate of 1 labels that are actually 0
    """
    # Convert probabilities to 2-class format for cleanlab
    pred_probs = np.stack([1 - preds, preds], axis=1)  # (N, 2)
    
    # Compute confident joint
    confident_joint = compute_confident_joint(
        labels=labels.astype(int),
        pred_probs=pred_probs,
        calibrate=True
    )
    
    # Estimate noise transition matrix (py, noise_matrix, inv_noise_matrix)
    try:
        py, noise_matrix, inv_noise_matrix = estimate_latent(
            confident_joint=confident_joint,
            labels=labels.astype(int)
        )
        
        # Noise rate = off-diagonal elements / total
        # noise_matrix[i][j] = P(given label = j | true label = i)
        # Off-diagonal = P(mislabeled)
        flip_rate_0_to_1 = noise_matrix[0, 1]  # P(labeled 1 | true 0)
        flip_rate_1_to_0 = noise_matrix[1, 0]  # P(labeled 0 | true 1)
        
        # Overall noise rate
        noise_rate = py[0] * flip_rate_0_to_1 + py[1] * flip_rate_1_to_0
        
    except Exception as e:
        print(f"    Warning: Could not estimate noise matrix: {e}")
        # Fallback: estimate from confident joint directly
        cj_normalized = confident_joint / confident_joint.sum()
        noise_rate = 1 - np.trace(cj_normalized)
        flip_rate_0_to_1 = 0.0
        flip_rate_1_to_0 = 0.0
    
    return confident_joint, noise_rate, flip_rate_0_to_1, flip_rate_1_to_0


def estimate_noise_rates(
    preds: np.ndarray, 
    labels: np.ndarray, 
    label_names: List[str],
    min_samples_per_class: int = 20
) -> Dict:
    """Estimate noise rates for all labels in multi-label classification
    
    Args:
        preds: (N, num_labels) predicted probabilities
        labels: (N, num_labels) labels (0, 1, -1, or NaN)
        label_names: List of label names
        min_samples_per_class: Minimum samples per class to compute noise rate
    
    Returns:
        Dictionary with per-label and overall noise statistics
    """
    num_labels = preds.shape[1]
    
    results = {
        'per_label': {},
        'summary': {}
    }
    
    all_noise_rates = []
    all_flip_0_to_1 = []
    all_flip_1_to_0 = []
    valid_labels = []
    
    print("\n" + "="*70)
    print("Estimating Noise Rate per Label")
    print("="*70)
    
    for i in range(num_labels):
        label_name = label_names[i] if i < len(label_names) else f"Label_{i}"
        
        # Filter valid samples (only 0 and 1, exclude uncertain -1 and NaN)
        valid_mask = (labels[:, i] == 0) | (labels[:, i] == 1)
        valid_count = valid_mask.sum()
        
        if valid_count < min_samples_per_class * 2:
            print(f"\n[{label_name}] Skipped - insufficient samples ({valid_count})")
            results['per_label'][label_name] = {
                'status': 'skipped',
                'reason': f'insufficient samples ({valid_count})',
                'valid_samples': int(valid_count)
            }
            continue
        
        valid_preds = preds[valid_mask, i]
        valid_labels_binary = labels[valid_mask, i]
        
        # Check class balance
        n_positive = (valid_labels_binary == 1).sum()
        n_negative = (valid_labels_binary == 0).sum()
        
        if n_positive < min_samples_per_class or n_negative < min_samples_per_class:
            print(f"\n[{label_name}] Skipped - imbalanced (pos: {n_positive}, neg: {n_negative})")
            results['per_label'][label_name] = {
                'status': 'skipped',
                'reason': f'imbalanced (pos: {n_positive}, neg: {n_negative})',
                'valid_samples': int(valid_count),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative)
            }
            continue
        
        # Compute noise rate
        try:
            cj, noise_rate, flip_0_to_1, flip_1_to_0 = compute_confident_joint_for_label(
                valid_preds, valid_labels_binary
            )
            
            print(f"\n[{label_name}]")
            print(f"  Samples: {valid_count} (pos: {n_positive}, neg: {n_negative})")
            print(f"  Estimated Noise Rate: {noise_rate*100:.2f}%")
            print(f"  Flip Rate (0→1): {flip_0_to_1*100:.2f}%")
            print(f"  Flip Rate (1→0): {flip_1_to_0*100:.2f}%")
            print(f"  Confident Joint:\n{cj}")
            
            results['per_label'][label_name] = {
                'status': 'success',
                'valid_samples': int(valid_count),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative),
                'estimated_noise_rate': float(noise_rate),
                'flip_rate_0_to_1': float(flip_0_to_1),
                'flip_rate_1_to_0': float(flip_1_to_0),
                'confident_joint': cj.tolist()
            }
            
            all_noise_rates.append(noise_rate)
            all_flip_0_to_1.append(flip_0_to_1)
            all_flip_1_to_0.append(flip_1_to_0)
            valid_labels.append(label_name)
            
        except Exception as e:
            print(f"\n[{label_name}] Error: {e}")
            results['per_label'][label_name] = {
                'status': 'error',
                'error': str(e),
                'valid_samples': int(valid_count)
            }
    
    # Compute summary statistics
    if all_noise_rates:
        results['summary'] = {
            'num_labels_analyzed': len(all_noise_rates),
            'labels_analyzed': valid_labels,
            'mean_noise_rate': float(np.mean(all_noise_rates)),
            'std_noise_rate': float(np.std(all_noise_rates)),
            'min_noise_rate': float(np.min(all_noise_rates)),
            'max_noise_rate': float(np.max(all_noise_rates)),
            'median_noise_rate': float(np.median(all_noise_rates)),
            'mean_flip_rate_0_to_1': float(np.mean(all_flip_0_to_1)),
            'mean_flip_rate_1_to_0': float(np.mean(all_flip_1_to_0))
        }
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Labels analyzed: {len(all_noise_rates)}/{num_labels}")
        print(f"Mean Noise Rate: {results['summary']['mean_noise_rate']*100:.2f}% ± {results['summary']['std_noise_rate']*100:.2f}%")
        print(f"Noise Rate Range: [{results['summary']['min_noise_rate']*100:.2f}%, {results['summary']['max_noise_rate']*100:.2f}%]")
        print(f"Median Noise Rate: {results['summary']['median_noise_rate']*100:.2f}%")
        print(f"Mean Flip Rate (0→1): {results['summary']['mean_flip_rate_0_to_1']*100:.2f}%")
        print(f"Mean Flip Rate (1→0): {results['summary']['mean_flip_rate_1_to_0']*100:.2f}%")
    else:
        results['summary'] = {
            'num_labels_analyzed': 0,
            'error': 'No labels could be analyzed'
        }
        print("\n[WARNING] No labels could be analyzed!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Estimate label noise rate using Confident Learning')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Path to experiment directory (alternative to config)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: <experiment>/noise_rate_estimate.json)')
    parser.add_argument('--min-samples', type=int, default=20,
                       help='Minimum samples per class to compute noise rate (default: 20)')
    parser.add_argument('--use-test', action='store_true',
                       help='Use test set predictions instead of validation set')
    args = parser.parse_args()
    
    # Load config
    if args.experiment:
        config_path = Path(args.experiment) / 'config.yaml'
        if not config_path.exists():
            # Try to find config in the experiment directory
            config_path = Path(args.experiment) / 'config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found in {args.experiment}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    print("="*70)
    print("Noise Rate Estimation using Confident Learning")
    print("="*70)
    
    # Load predictions
    print(f"\nLoading {'test' if args.use_test else 'validation'} predictions...")
    preds, labels, indices = load_predictions(config, use_test=args.use_test)
    print(f"Loaded {len(indices)} samples, {preds.shape[1]} labels")
    
    # Get label names
    label_names = config['data']['labels']
    
    # Estimate noise rates
    results = estimate_noise_rates(
        preds, labels, label_names, 
        min_samples_per_class=args.min_samples
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to experiment directory or confident_learning output dir
        output_dir = config.get('confident_learning', {}).get('output_dir', 'confident_learning')
        if not os.path.isabs(output_dir):
            if 'output_dir' in config.get('pipeline', {}):
                output_dir = os.path.join(config['pipeline']['output_dir'], output_dir)
        output_path = Path(output_dir) / 'noise_rate_estimate.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "="*70)
    print("USAGE GUIDE")
    print("="*70)
    print("""
To compare LLM pseudo labels vs ground truth labels:

1. Train model with GROUND TRUTH labels:
   - Set data.source_type: 'mimic_jpg' in config
   - Run: python train_wsl.py
   - Run: python estimate_noise_rate.py
   - Save result as 'noise_rate_ground_truth.json'

2. Train model with LLM PSEUDO labels:
   - Set data.source_type: 'parquet' in config
   - Run: python train_wsl.py
   - Run: python estimate_noise_rate.py
   - Save result as 'noise_rate_llm_pseudo.json'

3. Compare mean_noise_rate from both results.
   Lower noise rate = better label quality!
""")


if __name__ == '__main__':
    main()
