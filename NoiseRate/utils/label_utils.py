"""
Label Utility Functions
Provides functions for label preprocessing and format conversion
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging



def filter_by_view_position(df: pd.DataFrame, allowed_views: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame by ViewPosition
    
    Args:
        df: DataFrame with 'ViewPosition' column
        allowed_views: List of allowed view positions (e.g., ['AP', 'PA'])
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> df_filtered = filter_by_view_position(df, ['AP', 'PA'])
    """
    if 'ViewPosition' not in df.columns:
        raise ValueError("DataFrame must have 'ViewPosition' column")
    
    initial_count = len(df)
    df_filtered = df[df['ViewPosition'].isin(allowed_views)].copy()
    filtered_count = len(df_filtered)
    
    print(f"ViewPosition filter: {initial_count} -> {filtered_count} samples "
          f"({filtered_count/initial_count*100:.1f}% retained)")
    
    return df_filtered


def handle_uncertain_labels(
    df: pd.DataFrame, 
    pathology_cols: List[str], 
    strategy: str = 'u_zeros'
) -> pd.DataFrame:
    """
    Handle uncertain labels (-1.0) and missing values (NaN)
    
    Args:
        df: DataFrame with pathology columns
        pathology_cols: List of pathology column names
        strategy: How to handle uncertain values
            - 'u_zeros': Treat -1.0 and NaN as 0 (negative)
            - 'u_ones': Treat -1.0 as 1 (positive), NaN as 0
            - 'u_ignore': Remove samples with any -1.0 values
    
    Returns:
        Processed DataFrame
    
    Example:
        >>> df_clean = handle_uncertain_labels(df, pathology_cols, 'u_zeros')
    """
    df_processed = df.copy()
    
    if strategy == 'u_zeros':
        # -1.0 and NaN -> 0
        for col in pathology_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0.0)
                df_processed[col] = df_processed[col].replace(-1.0, 0.0)
        
        print(f"Strategy 'u_zeros': Converted -1.0 and NaN to 0")
    
    elif strategy == 'u_ones':
        # -1.0 -> 1, NaN -> 0
        for col in pathology_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0.0)
                df_processed[col] = df_processed[col].replace(-1.0, 1.0)
        
        print(f"Strategy 'u_ones': Converted -1.0 to 1, NaN to 0")
    
    elif strategy == 'u_ignore':
        # Remove rows with any -1.0
        initial_count = len(df_processed)
        
        # Create mask for rows without -1.0
        mask = pd.Series([True] * len(df_processed), index=df_processed.index)
        for col in pathology_cols:
            if col in df_processed.columns:
                mask &= (df_processed[col] != -1.0)
        
        df_processed = df_processed[mask].copy()
        removed_count = initial_count - len(df_processed)
        
        print(f"Strategy 'u_ignore': Removed {removed_count} samples with -1.0 "
              f"({removed_count/initial_count*100:.1f}%)")
        
        # Still fill NaN with 0
        for col in pathology_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0.0)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Choose from 'u_zeros', 'u_ones', 'u_ignore'")
    
    return df_processed


def convert_to_multilabel_format(
    df: pd.DataFrame, 
    pathology_cols: List[str]
) -> List[List[int]]:
    """
    Convert binary label matrix to cleanlab multi-label format (list of lists)
    
    Each sample is represented as a list of class indices where the label is 1.
    
    Args:
        df: DataFrame with binary pathology columns (0 or 1)
        pathology_cols: Ordered list of pathology column names
    
    Returns:
        List of lists, where each inner list contains class indices for that sample
    
    Example:
        >>> # Sample with labels [0, 1, 0, 1, 0] becomes [1, 3]
        >>> labels = convert_to_multilabel_format(df, pathology_cols)
        >>> print(labels[0])  # [1, 3] - positive for classes 1 and 3
    """
    labels_list = []
    
    for _, row in df.iterrows():
        positive_classes = []
        for idx, col in enumerate(pathology_cols):
            if col in df.columns and row[col] == 1.0:
                positive_classes.append(idx)
        labels_list.append(positive_classes)
    
    return labels_list


def convert_to_binary_matrix(
    labels_list: List[List[int]], 
    num_classes: int
) -> np.ndarray:
    """
    Convert cleanlab multi-label format (list of lists) back to binary matrix
    
    This is the inverse operation of convert_to_multilabel_format
    
    Args:
        labels_list: List of lists, each containing positive class indices
        num_classes: Total number of classes
    
    Returns:
        Binary numpy array of shape (n_samples, num_classes)
    
    Example:
        >>> labels_list = [[1, 3], [0], [2, 4]]
        >>> matrix = convert_to_binary_matrix(labels_list, num_classes=5)
        >>> print(matrix.shape)  # (3, 5)
        >>> print(matrix[0])  # [0, 1, 0, 1, 0]
    """
    n_samples = len(labels_list)
    binary_matrix = np.zeros((n_samples, num_classes), dtype=int)
    
    for i, positive_classes in enumerate(labels_list):
        for class_idx in positive_classes:
            if 0 <= class_idx < num_classes:
                binary_matrix[i, class_idx] = 1
    
    return binary_matrix


def get_label_statistics(
    labels_list: List[List[int]], 
    pathology_names: List[str]
) -> pd.DataFrame:
    """
    Get statistics about multi-label distribution
    
    Args:
        labels_list: Multi-label format labels
        pathology_names: Names of pathology classes
    
    Returns:
        DataFrame with statistics per class
    """
    num_classes = len(pathology_names)
    binary_matrix = convert_to_binary_matrix(labels_list, num_classes)
    
    stats = []
    for idx, name in enumerate(pathology_names):
        positive_count = binary_matrix[:, idx].sum()
        stats.append({
            'pathology': name,
            'positive_count': positive_count,
            'positive_rate': positive_count / len(labels_list) * 100
        })
    
    return pd.DataFrame(stats)


def validate_cleanlab_format(
    labels_list: List[List[int]], 
    pred_probs: np.ndarray,
    verbose: bool = True
) -> bool:
    """
    Validate that labels and predictions are in correct cleanlab format
    
    Args:
        labels_list: Multi-label format labels (list of lists)
        pred_probs: Prediction probabilities (n_samples, n_classes)
        verbose: Whether to print validation messages
    
    Returns:
        True if valid, False otherwise
    """
    valid = True
    
    # Check shapes
    n_samples = len(labels_list)
    if pred_probs.shape[0] != n_samples:
        if verbose:
            print(f"❌ Shape mismatch: labels has {n_samples} samples, "
                  f"pred_probs has {pred_probs.shape[0]}")
        valid = False
    
    # Check labels_list format
    if not isinstance(labels_list, list):
        if verbose:
            print(f"❌ labels_list must be a list, got {type(labels_list)}")
        valid = False
    
    # Check each sample
    for i, sample_labels in enumerate(labels_list):
        if not isinstance(sample_labels, list):
            if verbose:
                print(f"❌ labels_list[{i}] must be a list, got {type(sample_labels)}")
            valid = False
            break
        
        # Check class indices are valid
        for class_idx in sample_labels:
            if not isinstance(class_idx, int):
                if verbose:
                    print(f"❌ Class index must be int, got {type(class_idx)} at sample {i}")
                valid = False
                break
            if class_idx < 0 or class_idx >= pred_probs.shape[1]:
                if verbose:
                    print(f"❌ Invalid class index {class_idx} at sample {i} "
                          f"(must be 0-{pred_probs.shape[1]-1})")
                valid = False
                break
    
    # Check pred_probs format
    if not isinstance(pred_probs, np.ndarray):
        if verbose:
            print(f"❌ pred_probs must be numpy array, got {type(pred_probs)}")
        valid = False
    elif len(pred_probs.shape) != 2:
        if verbose:
            print(f"❌ pred_probs must be 2D, got shape {pred_probs.shape}")
        valid = False
    elif not np.allclose(pred_probs.min(), 0, atol=0.1) or not np.allclose(pred_probs.max(), 1, atol=0.1):
        if verbose:
            print(f"⚠️  Warning: pred_probs should be in range [0, 1], "
                  f"got range [{pred_probs.min():.3f}, {pred_probs.max():.3f}]")
    
    if valid and verbose:
        print(f"✅ Format validation passed!")
        print(f"   Samples: {n_samples}, Classes: {pred_probs.shape[1]}")
    

def log_label_statistics(
    df: pd.DataFrame,
    pathology_cols: List[str],
    logger: logging.Logger = None,
    prefix: str = ""
):
    """
    Log detailed statistics about label distribution
    
    Args:
        df: DataFrame with labels
        pathology_cols: List of pathology columns
        logger: Logger instance (if None, uses print)
        prefix: Prefix for log messages (e.g., "GT" or "Pseudo")
    """
    log_func = logger.info if logger else print
    
    n_samples = len(df)
    log_func(f"{prefix} Label Statistics (Total Samples: {n_samples}):")
    log_func(f"{'Pathology':<30} {'Positive':<10} {'Negative':<10} {'Pos Rate':<10}")
    log_func("-" * 65)
    
    for col in pathology_cols:
        if col in df.columns:
            n_pos = (df[col] == 1.0).sum()
            n_neg = (df[col] == 0.0).sum()
            pos_rate = n_pos / n_samples * 100
            
            log_func(f"{col:<30} {n_pos:<10} {n_neg:<10} {pos_rate:>9.1f}%")
        else:
            log_func(f"{col:<30} {'MISSING':<10} {'MISSING':<10} {'N/A':>9}")
    log_func("")

