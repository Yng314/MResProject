"""
Test script for xrv_utils.py
Tests model loading, prediction, and pathology mapping on small sample
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from pathlib import Path
from utils.xrv_utils import (
    load_pretrained_model,
    get_pathology_mapping,
    get_ordered_pathology_list,
    create_dataloader,
    extract_predictions,
    validate_predictions
)


def test_pathology_mapping():
    """Test pathology mapping function"""
    print("\n" + "="*60)
    print("TEST: Pathology Mapping")
    print("="*60)
    
    mapping = get_pathology_mapping()
    ordered_pathologies = get_ordered_pathology_list()
    
    print(f"\nMappable pathologies: {len(mapping)}")
    print("\nMapping (CheXpert -> TorchXRayVision index):")
    for chexpert_name in ordered_pathologies:
        if chexpert_name in mapping:
            print(f"  {chexpert_name}: {mapping[chexpert_name]}")
    
    # Verify we have 12 mappable pathologies
    assert len(mapping) == 12, f"Should have 12 mappable pathologies, got {len(mapping)}"
    assert len(ordered_pathologies) == 12, f"Should have 12 ordered pathologies, got {len(ordered_pathologies)}"
    
    print("\n✅ PASSED")
    return mapping, ordered_pathologies


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("TEST: Model Loading")
    print("="*60)
    
    model = load_pretrained_model()
    
    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode"
    
    print("✅ PASSED")
    return model


def test_small_sample_inference():
    """Test inference on small sample (5-10 images)"""
    print("\n" + "="*60)
    print("TEST: Small Sample Inference (10 samples)")
    print("="*60)
    
    # Load GT metadata and filter
    df = pd.read_csv("d:/workspace/MRes/datasets/mimic-cxr-clean/gt/metadata.csv")
    df = df[df['ViewPosition'].isin(['AP', 'PA'])]
    
    # Take first 10 samples
    df_sample = df.head(10).copy()
    image_paths = df_sample['image_path'].tolist()
    
    print(f"\nLoaded {len(image_paths)} sample images")
    
    # Create dataloader
    base_path = "d:/workspace/MRes/datasets/mimic-cxr-clean/gt"
    dataloader = create_dataloader(
        image_paths,
        base_path=base_path,
        batch_size=4,
        num_workers=0
    )
    
    # Load model
    model = load_pretrained_model()
    
    # Extract predictions
    full_preds, mapped_preds = extract_predictions(model, dataloader)
    
    print(f"\nPrediction shapes:")
    print(f"  Full (18 classes): {full_preds.shape}")
    print(f"  Mapped (12 classes): {mapped_preds.shape}")
    
    # Verify shapes
    assert full_preds.shape == (10, 18), f"Full predictions shape should be (10, 18), got {full_preds.shape}"
    assert mapped_preds.shape == (10, 12), f"Mapped predictions shape should be (10, 12), got {mapped_preds.shape}"
    
    print("✅ Shape validation PASSED")
    
    # Validate predictions
    ordered_pathologies = get_ordered_pathology_list()
    is_valid = validate_predictions(mapped_preds, ordered_pathologies)
    
    assert is_valid, "Prediction validation failed"
    print("✅ Prediction validation PASSED")
    
    # Show sample predictions
    print("\n" + "="*60)
    print("Sample Predictions (first 3 samples)")
    print("="*60)
    for i in range(min(3, len(mapped_preds))):
        print(f"\nSample {i} ({Path(image_paths[i]).name}):")
        top_3_indices = np.argsort(mapped_preds[i])[-3:][::-1]
        for idx in top_3_indices:
            prob = mapped_preds[i, idx]
            pathology = ordered_pathologies[idx]
            print(f"  {pathology}: {prob:.3f}")
    
    print("\n✅ PASSED")
    return full_preds, mapped_preds


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("XRV UTILS TEST SUITE".center(60))
    print("="*60)
    
    try:
        # Test 1: Pathology mapping
        mapping, ordered_pathologies = test_pathology_mapping()
        
        # Test 2: Model loading
        model = test_model_loading()
        
        # Test 3: Small sample inference
        full_preds, mapped_preds = test_small_sample_inference()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!".center(60))
        print("="*60)
        
        print("\nSummary:")
        print(f"  ✅ Model loaded successfully")
        print(f"  ✅ Pathology mapping: 12 classes")
        print(f"  ✅ Inference on 10 samples completed")
        print(f"  ✅ Output shapes validated")
        print(f"  ✅ Prediction probabilities in valid range")
        print()
        
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
