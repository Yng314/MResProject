"""
Test script for label_utils.py
Tests all utility functions with small sample data
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from utils.label_utils import (
    filter_by_view_position,
    handle_uncertain_labels,
    convert_to_multilabel_format,
    convert_to_binary_matrix,
    get_label_statistics,
    validate_cleanlab_format
)

def print_test_header(test_name):
    """Print test section header"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


def test_filter_by_view_position():
    """Test ViewPosition filtering"""
    print_test_header("filter_by_view_position")
    
    # Create sample data
    df = pd.DataFrame({
        'ViewPosition': ['AP', 'PA', 'LATERAL', 'AP', 'LL', 'PA'],
        'data': [1, 2, 3, 4, 5, 6]
    })
    
    print("Original data:")
    print(df)
    
    # Test filtering
    df_filtered = filter_by_view_position(df, ['AP', 'PA'])
    
    print("\nFiltered data:")
    print(df_filtered)
    
    assert len(df_filtered) == 4, "Should have 4 samples (2 AP + 2 PA)"
    assert list(df_filtered['ViewPosition']) == ['AP', 'PA', 'AP', 'PA']
    print("✅ PASSED")


def test_handle_uncertain_labels():
    """Test uncertain label handling with different strategies"""
    print_test_header("handle_uncertain_labels")
    
    # Create sample data with uncertain labels
    df = pd.DataFrame({
        'Pathology1': [1.0, 0.0, -1.0, np.nan, 1.0],
        'Pathology2': [0.0, 1.0, -1.0, 0.0, np.nan],
        'Pathology3': [-1.0, -1.0, 1.0, 0.0, 1.0]
    })
    pathology_cols = ['Pathology1', 'Pathology2', 'Pathology3']
    
    print("Original data:")
    print(df)
    print()
    
    # Test u_zeros strategy
    print("\n--- Strategy: u_zeros ---")
    df_zeros = handle_uncertain_labels(df.copy(), pathology_cols, 'u_zeros')
    print(df_zeros)
    assert (df_zeros == -1.0).sum().sum() == 0, "Should have no -1.0 values"
    assert df_zeros.isna().sum().sum() == 0, "Should have no NaN values"
    print("✅ u_zeros PASSED")
    
    # Test u_ones strategy
    print("\n--- Strategy: u_ones ---")
    df_ones = handle_uncertain_labels(df.copy(), pathology_cols, 'u_ones')
    print(df_ones)
    assert (df_ones == -1.0).sum().sum() == 0, "Should have no -1.0 values"
    assert df_ones.isna().sum().sum() == 0, "Should have no NaN values"
    # Check that -1.0 was converted to 1.0
    assert df_ones.loc[0, 'Pathology3'] == 1.0, "-1.0 should become 1.0"
    print("✅ u_ones PASSED")
    
    # Test u_ignore strategy
    print("\n--- Strategy: u_ignore ---")
    df_ignore = handle_uncertain_labels(df.copy(), pathology_cols, 'u_ignore')
    print(df_ignore)
    assert len(df_ignore) < len(df), "Should have fewer samples"
    assert (df_ignore == -1.0).sum().sum() == 0, "Should have no -1.0 values"
    print("✅ u_ignore PASSED")


def test_multilabel_conversion():
    """Test conversion between binary matrix and list of lists format"""
    print_test_header("Multilabel Format Conversion")
    
    # Create sample binary data
    df = pd.DataFrame({
        'Pathology1': [1, 0, 1, 0],
        'Pathology2': [0, 1, 1, 0],
        'Pathology3': [1, 0, 0, 1]
    })
    pathology_cols = ['Pathology1', 'Pathology2', 'Pathology3']
    
    print("Binary matrix:")
    print(df)
    
    # Convert to list of lists
    labels_list = convert_to_multilabel_format(df, pathology_cols)
    print("\nList of lists format:")
    for i, labels in enumerate(labels_list):
        print(f"  Sample {i}: {labels}")
    
    # Verify
    assert labels_list[0] == [0, 2], "Sample 0 should be [0, 2]"
    assert labels_list[1] == [1], "Sample 1 should be [1]"
    assert labels_list[2] == [0, 1], "Sample 2 should be [0, 1]"
    assert labels_list[3] == [2], "Sample 3 should be [2]"
    print("✅ convert_to_multilabel_format PASSED")
    
    # Convert back to binary matrix
    binary_matrix = convert_to_binary_matrix(labels_list, num_classes=3)
    print("\nConverted back to binary matrix:")
    print(binary_matrix)
    
    # Verify round-trip
    assert np.array_equal(binary_matrix, df.values), "Round-trip conversion should match"
    print("✅ convert_to_binary_matrix PASSED")


def test_label_statistics():
    """Test label statistics calculation"""
    print_test_header("Label Statistics")
    
    labels_list = [
        [0, 1],     # Positive for classes 0 and 1
        [1],        # Positive for class 1
        [0, 2],     # Positive for classes 0 and 2
        [],         # Negative for all
        [1, 2]      # Positive for classes 1 and 2
    ]
    pathology_names = ['Pathology A', 'Pathology B', 'Pathology C']
    
    stats = get_label_statistics(labels_list, pathology_names)
    print(stats)
    
    # Verify counts
    # Class 0: appears in samples 0, 2 -> 2 positives
    # Class 1: appears in samples 0, 1, 4 -> 3 positives  
    # Class 2: appears in samples 2, 4 -> 2 positives
    assert stats.loc[0, 'positive_count'] == 2, "Class 0 should have 2 positives"
    assert stats.loc[1, 'positive_count'] == 3, "Class 1 should have 3 positives"
    assert stats.loc[2, 'positive_count'] == 2, "Class 2 should have 2 positives"
    print("✅ PASSED")


def test_cleanlab_format_validation():
    """Test cleanlab format validation"""
    print_test_header("Cleanlab Format Validation")
    
    # Valid format
    labels_list = [[0, 1], [1], [0, 2], [], [1, 2]]
    pred_probs = np.random.rand(5, 3)  # 5 samples, 3 classes
    
    print("Testing valid format...")
    is_valid = validate_cleanlab_format(labels_list, pred_probs, verbose=True)
    assert is_valid, "Should be valid"
    print()
    
    # Invalid: shape mismatch
    print("Testing invalid format (shape mismatch)...")
    pred_probs_wrong = np.random.rand(4, 3)  # Wrong number of samples
    is_valid = validate_cleanlab_format(labels_list, pred_probs_wrong, verbose=True)
    assert not is_valid, "Should be invalid"
    print()
    
    # Invalid: wrong class index
    print("Testing invalid format (wrong class index)...")
    labels_list_wrong = [[0, 1], [1], [0, 5], [], [1, 2]]  # Class 5 doesn't exist
    is_valid = validate_cleanlab_format(labels_list_wrong, pred_probs, verbose=True)
    assert not is_valid, "Should be invalid"
    print("✅ PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("LABEL UTILS TEST SUITE".center(60))
    print("="*60)
    
    try:
        test_filter_by_view_position()
        test_handle_uncertain_labels()
        test_multilabel_conversion()
        test_label_statistics()
        test_cleanlab_format_validation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!".center(60))
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
