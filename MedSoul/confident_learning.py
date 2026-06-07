"""
Phase 4: Confident Learning for label noise detection and cleaning
"""
import os
import yaml
import json
import numpy as np
from pathlib import Path
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores


def load_data(config):
    """Load predictions and pseudo labels"""
    # Resolve checkpoint directory
    checkpoint_dir = config['wsl']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir):
        if 'output_dir' in config.get('pipeline', {}):
            checkpoint_dir = os.path.join(config['pipeline']['output_dir'], checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    pred_file = checkpoint_dir / 'val_predictions.npz'
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found at {pred_file}. Run train_wsl.py first.")
    
    data = np.load(pred_file)
    preds = data['preds']  # (N, num_classes)
    labels = data['labels']  # (N, num_classes)
    indices = data['indices']  # (N,)
    
    return preds, labels, indices


def identify_noise_multi_label(preds, labels, label_names):
    """Identify noisy labels for multi-label classification
    
    Returns:
        noise_mask: (N,) boolean array, True for noisy samples
        quality_scores: (N,) quality score for each sample
    """
    N, num_classes = preds.shape
    
    # For each sample, compute average quality across all labels
    sample_quality = np.zeros(N)
    
    for i in range(num_classes):
        # Filter valid labels (not uncertain)
        valid_mask = (labels[:, i] == 0) | (labels[:, i] == 1)
        
        if valid_mask.sum() < 10:  # Skip if too few samples
            continue
        
        valid_labels = labels[valid_mask, i].astype(int)
        valid_preds = preds[valid_mask, i]
        
        # Get quality scores for this label
        try:
            quality = get_label_quality_scores(
                labels=valid_labels,
                pred_probs=np.stack([1 - valid_preds, valid_preds], axis=1)
            )
            
            # Map back to full array
            sample_quality[valid_mask] += quality
        except Exception as e:
            print(f"Warning: Failed to compute quality for label {label_names[i]}: {e}")
            continue
    
    # Normalize by number of valid labels
    valid_label_counts = ((labels == 0) | (labels == 1)).sum(axis=1)
    valid_label_counts = np.maximum(valid_label_counts, 1)  # Avoid division by zero
    sample_quality /= valid_label_counts
    
    # Identify noisy samples (bottom 20% quality)
    threshold = np.percentile(sample_quality, 20)
    noise_mask = sample_quality < threshold
    
    return noise_mask, sample_quality


def clean_labels(config, noise_indices):
    """
    Remove noisy samples from pseudo labels
    
    NOTE: This function creates a COPY of the labels.
    Original labels are preserved at config['llm']['cache_file']
    Cleaned labels are saved to config['confident_learning']['output_dir']/pseudo_labels_cleaned.json
    """
    # Load original pseudo labels (read-only)
    pseudo_labels_path = config['llm']['cache_file']
    if not os.path.isabs(pseudo_labels_path):
        if 'output_dir' in config.get('pipeline', {}):
            pseudo_labels_path = os.path.join(config['pipeline']['output_dir'], pseudo_labels_path)
    
    with open(pseudo_labels_path, 'r') as f:
        pseudo_labels = json.load(f)
    
    # Remove noisy samples
    cleaned_labels = {}
    removed_count = 0
    
    for idx_str, labels in pseudo_labels.items():
        idx = int(idx_str)
        if idx not in noise_indices:
            cleaned_labels[idx_str] = labels
        else:
            removed_count += 1
    
    print(f"Removed {removed_count} noisy samples")
    print(f"Remaining samples: {len(cleaned_labels)}")
    
    # Save cleaned labels
    output_dir = config['confident_learning']['output_dir']
    if not os.path.isabs(output_dir):
        if 'output_dir' in config.get('pipeline', {}):
            output_dir = os.path.join(config['pipeline']['output_dir'], output_dir)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = output_dir / 'pseudo_labels_cleaned.json'
    with open(cleaned_path, 'w') as f:
        json.dump(cleaned_labels, f, indent=2)
    
    print(f"Saved cleaned labels to {cleaned_path}")
    
    return cleaned_labels


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
    
    if not config['confident_learning']['enabled']:
        print("Confident Learning is disabled in config")
        return
    
    print("="*50)
    print("Phase 4: Confident Learning")
    print("="*50)
    
    # Load predictions and labels
    print("\nLoading predictions...")
    preds, labels, indices = load_data(config)
    print(f"Loaded {len(indices)} samples")
    
    # Identify noisy labels
    print("\nIdentifying noisy labels...")
    noise_mask, quality_scores = identify_noise_multi_label(
        preds, labels, config['data']['labels']
    )
    
    noise_indices = indices[noise_mask]
    print(f"Identified {len(noise_indices)} noisy samples ({100*len(noise_indices)/len(indices):.1f}%)")
    
    # Save quality scores
    output_dir = config['confident_learning']['output_dir']
    if not os.path.isabs(output_dir):
        if 'output_dir' in config.get('pipeline', {}):
            output_dir = os.path.join(config['pipeline']['output_dir'], output_dir)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / 'label_quality.npz',
        indices=indices,
        quality_scores=quality_scores,
        noise_mask=noise_mask
    )
    
    # Clean labels
    print("\nCleaning pseudo labels...")
    cleaned_labels = clean_labels(config, noise_indices)
    
    print("\nConfident Learning completed!")
    print(f"You can now retrain WSL with cleaned labels: {output_dir / 'pseudo_labels_cleaned.json'}")


if __name__ == '__main__':
    main()
