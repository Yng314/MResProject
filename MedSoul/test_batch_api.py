"""
Test script for Batch API functionality
Tests with a small sample to verify the setup
"""
import os
import yaml
from dotenv import load_dotenv
from utils.qwen_batch_api import QwenBatchLabeler


def main():
    print("="*70)
    print("  Batch API Test")
    print("="*70)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv(config['llm']['api_key_env'])
    if not api_key:
        print("❌ API key not found!")
        print(f"Please set {config['llm']['api_key_env']} in .env file")
        return
    
    print("✅ API key found")
    
    # Create test reports
    test_reports = [
        "No acute cardiopulmonary process. Heart size is normal. Lungs are clear.",
        "Cardiomegaly with mild pulmonary edema. Possible pleural effusion on the right.",
        "Pneumonia in the right lower lobe. No pleural effusion or pneumothorax.",
    ]
    
    print(f"\n📝 Testing with {len(test_reports)} sample reports")
    
    # Initialize batch labeler (min 24 hours required by API)
    labeler = QwenBatchLabeler(
        api_key=api_key,
        base_url=config['llm']['base_url'],
        model_name=config['llm']['model_name'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens'],
        label_names=config['data']['labels'],
        max_wait_hours=24,  # Minimum allowed by Qwen API
        poll_interval=30   # Poll every 30 seconds for test
    )
    
    print("\n🚀 Starting batch test...")
    print("This will:")
    print("  1. Create JSONL file")
    print("  2. Upload to Qwen")
    print("  3. Create batch job")
    print("  4. Poll for completion (max 24 hours)")
    print("  5. Download results")
    
    try:
        results = labeler.batch_extract(
            reports=test_reports,
            save_path="test_batch_results.json",
            temp_dir="test_temp_batch"
        )
        
        print("\n" + "="*70)
        print("✅ TEST PASSED")
        print("="*70)
        print(f"Successfully processed {len(results)} reports")
        print("\nSample result:")
        if results:
            first_key = list(results.keys())[0]
            import json
            print(json.dumps(results[first_key], indent=2))
        
        print("\n💡 Batch API is working correctly!")
        print("You can now use it for full-scale label generation.")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        print("\nPlease check:")
        print("  1. API key is correct and has sufficient quota")
        print("  2. Network connection is stable")
        print("  3. Model name is supported for batch API")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
