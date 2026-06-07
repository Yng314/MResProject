"""Quick test to verify setup and imports"""
import sys
import yaml
from pathlib import Path

print("Testing MedSoul Setup...")
print("="*50)

# Test 1: Config file
print("\n1. Testing config file...")
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("[OK] Config loaded successfully")
    print(f"  - Data samples: {config['data']['num_samples']}")
    print(f"  - MAE epochs: {config['mae']['epochs']}")
    print(f"  - WSL batch size: {config['wsl']['linear_probe']['batch_size']}")
except Exception as e:
    print(f"[ERROR] Failed to load config: {e}")
    sys.exit(1)

# Test 2: Data files
print("\n2. Testing data files...")
try:
    found_files = []
    for path in config['data']['parquet_paths']:
        if Path(path).exists():
            found_files.append(path)
            print(f"[OK] Found: {path}")
        else:
            print(f"[ERROR] Missing: {path}")
    
    if not found_files:
        print("  [WARN] No parquet files found!")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# Test 3: Import modules
print("\n3. Testing module imports...")
try:
    from data.dataset import MIMICDataset, MAEDataset
    print("[OK] data.dataset")
    
    from models.mae import MAE
    print("[OK] models.mae")
    
    from models.resnet import ResNetClassifier
    print("[OK] models.resnet")
    
    from utils.qwen_api import QwenLabeler
    print("[OK] utils.qwen_api")
    
    from utils.metrics import compute_metrics
    print("[OK] utils.metrics")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: PyTorch and CUDA
print("\n4. Testing PyTorch and CUDA...")
try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"[ERROR] PyTorch test failed: {e}")

# Test 5: Environment file
print("\n5. Testing environment file...")
try:
    from dotenv import load_dotenv
    import os
    
    if Path('.env').exists():
        load_dotenv()
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if api_key:
            print(f"[OK] API key found (length: {len(api_key)})")
        else:
            print("[ERROR] API key not set in .env")
    else:
        print("[WARN] .env file not found (needed for Phase 1)")
except Exception as e:
    print(f"[ERROR] Error: {e}")

# Test 6: Create a small model
print("\n6. Testing model creation...")
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test MAE
    mae = MAE(img_size=512, patch_size=32).to(device)
    print(f"[OK] MAE model created on {device}")
    
    # Test classifier
    classifier = ResNetClassifier(num_classes=12).to(device)
    print(f"[OK] ResNet classifier created")
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        mae_out, _, _ = mae(dummy_input)
        clf_out = classifier(dummy_input)
    print(f"[OK] Forward pass successful")
    print(f"  - MAE output shape: {mae_out.shape}")
    print(f"  - Classifier output shape: {clf_out.shape}")
    
except Exception as e:
    print(f"[ERROR] Model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Setup verification complete!")
print("\nNext steps:")
print("1. Make sure .env file exists with DASHSCOPE_API_KEY")
print("2. Run: python main.py")
print("   Or run individual phases:")
print("   - python generate_labels.py")
print("   - python train_mae.py")
print("   - python train_wsl.py")
print("   - python confident_learning.py")

