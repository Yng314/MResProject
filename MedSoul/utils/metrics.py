"""Metrics for evaluation"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Dict


def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute multi-label classification metrics
    
    Args:
        preds: (N, num_classes) predicted probabilities
        labels: (N, num_classes) ground truth labels (0 or 1)
        threshold: threshold for binary prediction
    
    Returns:
        Dictionary of metrics
    """
    # Filter out uncertain labels (-1.0, 0.5, etc.)
    valid_mask = (labels == 0) | (labels == 1)
    
    metrics = {}
    
    # AUC-ROC (per class and macro average)
    try:
        aucs = []
        for i in range(labels.shape[1]):
            mask = valid_mask[:, i]
            if mask.sum() > 0 and len(np.unique(labels[mask, i])) > 1:
                auc = roc_auc_score(labels[mask, i], preds[mask, i])
                aucs.append(auc)
        
        if aucs:
            metrics['auc_macro'] = np.mean(aucs)
    except Exception as e:
        print(f"AUC computation failed: {e}")
        metrics['auc_macro'] = 0.0
    
    # Average Precision (mAP)
    try:
        aps = []
        for i in range(labels.shape[1]):
            mask = valid_mask[:, i]
            if mask.sum() > 0 and len(np.unique(labels[mask, i])) > 1:
                ap = average_precision_score(labels[mask, i], preds[mask, i])
                aps.append(ap)
        
        if aps:
            metrics['map'] = np.mean(aps)
    except Exception as e:
        print(f"mAP computation failed: {e}")
        metrics['map'] = 0.0
    
    # F1 Score
    try:
        preds_binary = (preds > threshold).astype(int)
        
        # Macro F1
        f1_scores = []
        for i in range(labels.shape[1]):
            mask = valid_mask[:, i]
            if mask.sum() > 0:
                f1 = f1_score(labels[mask, i], preds_binary[mask, i], zero_division=0)
                f1_scores.append(f1)
        
        if f1_scores:
            metrics['f1_macro'] = np.mean(f1_scores)
    except Exception as e:
        print(f"F1 computation failed: {e}")
        metrics['f1_macro'] = 0.0
    
    return metrics


def get_predictions(model: torch.nn.Module, dataloader, device: torch.device) -> tuple:
    """Get predictions from model
    
    Returns:
        preds: (N, num_classes) probabilities
        labels: (N, num_classes) ground truth
        indices: (N,) sample indices
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch.get('labels')
            indices = batch['idx']
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            if labels is not None:
                all_labels.append(labels.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    indices = np.concatenate(all_indices, axis=0)
    
    return preds, labels, indices

