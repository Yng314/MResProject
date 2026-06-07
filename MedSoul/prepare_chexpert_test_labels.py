"""
Compute noise rate using CheXpert labels instead of Ground Truth labels
for validating Confident Learning effectiveness
"""
import os
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load existing test predictions
pred_file = 'outputs/slurm_experiment/wsl_train/test_predictions.npz'
data = np.load(pred_file)
preds = data['preds']  # (1029, 14, 4)
gt_labels = data['labels']  # (1029, 14) - Ground Truth
indices = data['indices']

print(f"Loaded predictions: {preds.shape}")
print(f"GT labels: {gt_labels.shape}")

# Load test set to get study_ids
test_csv = 'datasets/mimic-cxr-2.1.0-test-set-labeled.csv'
split_csv = 'datasets/mimic-cxr-2.0.0-split.csv.gz'

test_df = pd.read_csv(test_csv)
split_df = pd.read_csv(split_csv, compression='gzip')
split_df = split_df[split_df['split'] == 'test']
merged_df = split_df.merge(test_df, on='study_id', how='inner')

print(f"Test samples in merged df: {len(merged_df)}")

# Load CheXpert labels
chexpert_csv = 'datasets/mimic-cxr-2.0.0-chexpert.csv.gz'
chexpert_df = pd.read_csv(chexpert_csv, compression='gzip')

# Merge CheXpert labels with test set
test_with_chexpert = merged_df.merge(chexpert_df, on=['study_id', 'subject_id'], how='left', suffixes=('_gt', '_chexpert'))

print(f"Test with CheXpert: {len(test_with_chexpert)}")

# Load config for label names
with open('slurm_jobs/config_slurm.yaml') as f:
    config = yaml.safe_load(f)

label_names = config['data']['labels']
print(f"Labels: {label_names}")

# Build CheXpert labels array (same order as predictions)
chexpert_labels = np.full((len(test_with_chexpert), 14), np.nan)

# Map label names (handle Airspace Opacity vs Lung Opacity)
chexpert_col_map = {}
for i, name in enumerate(label_names):
    if name + '_chexpert' in test_with_chexpert.columns:
        chexpert_col_map[i] = name + '_chexpert'
    elif name == 'Airspace Opacity' and 'Lung Opacity' in test_with_chexpert.columns:
        chexpert_col_map[i] = 'Lung Opacity'
    elif name in test_with_chexpert.columns:
        chexpert_col_map[i] = name

print(f"CheXpert column mapping: {chexpert_col_map}")

for i, name in enumerate(label_names):
    col = chexpert_col_map.get(i)
    if col and col in test_with_chexpert.columns:
        chexpert_labels[:, i] = test_with_chexpert[col].values

print(f"CheXpert labels shape: {chexpert_labels.shape}")

# Align predictions with labels (same sample order)
# Note: predictions were generated in the same order as merged_df
aligned_preds = preds[:len(chexpert_labels)]
aligned_gt = gt_labels[:len(chexpert_labels)]

print(f"Aligned preds: {aligned_preds.shape}")
print(f"Aligned GT: {aligned_gt.shape}")

# Save for noise rate calculation
np.savez(
    'outputs/slurm_experiment/wsl_train/test_predictions_chexpert.npz',
    preds=aligned_preds,
    labels=chexpert_labels,  # Use CheXpert labels instead of GT
    indices=indices[:len(chexpert_labels)]
)

print("\nSaved test_predictions_chexpert.npz with CheXpert labels")

# Also compare GT vs CheXpert label agreement
print("\n" + "="*60)
print("Label Agreement: Ground Truth vs CheXpert")
print("="*60)

for i, name in enumerate(label_names):
    gt_col = aligned_gt[:, i]
    cx_col = chexpert_labels[:, i]
    
    # Only compare where both have values (not NaN)
    valid_mask = ~np.isnan(gt_col) & ~np.isnan(cx_col)
    
    if valid_mask.sum() < 20:
        print(f"[{name}] Skipped - insufficient samples ({valid_mask.sum()})")
        continue
    
    gt_valid = gt_col[valid_mask]
    cx_valid = cx_col[valid_mask]
    
    # Convert to binary
    gt_binary = (gt_valid >= 0.5).astype(int)
    cx_binary = (cx_valid >= 0.5).astype(int)
    
    agreement = (gt_binary == cx_binary).mean() * 100
    
    # Count disagreements
    gt_pos_cx_neg = ((gt_binary == 1) & (cx_binary == 0)).sum()
    gt_neg_cx_pos = ((gt_binary == 0) & (cx_binary == 1)).sum()
    
    print(f"[{name}] Agreement: {agreement:.1f}%, GT+CX-: {gt_pos_cx_neg}, GT-CX+: {gt_neg_cx_pos}")
