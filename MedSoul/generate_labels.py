"""
Phase 1: Generate pseudo labels using Qwen LLM
Supports both realtime and batch (50% cost savings) modes
"""
import os
import yaml
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from utils.qwen_api import QwenLabeler
from utils.qwen_batch_api import QwenBatchLabeler
from utils.qwen_preview_api import QwenPreviewLabeler


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume-batch', type=str, default=None,
                       help='Resume a previous batch job using batch_id')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv(config['llm']['api_key_env'])
    if not api_key:
        raise ValueError(f"API key not found in environment variable {config['llm']['api_key_env']}")
    
    # Determine cache file path
    cache_file = config['llm']['cache_file']
    if not os.path.isabs(cache_file):
        # If relative path, make it relative to experiment dir if specified
        if 'output_dir' in config.get('pipeline', {}):
            cache_file = os.path.join(config['pipeline']['output_dir'], cache_file)
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Check if resuming a batch job
    if args.resume_batch:
        print(f"Resuming batch job: {args.resume_batch}")
        
        batch_labeler = QwenBatchLabeler(
            api_key=api_key,
            base_url=config['llm']['base_url'],
            model_name=config['llm'].get('model_name', 'qwen-max'),
            temperature=config['llm'].get('temperature', 0.0),
            max_tokens=config['llm'].get('max_tokens', 500),
            label_names=config['data']['labels'],
            max_wait_hours=config['llm'].get('batch', {}).get('max_wait_hours', 24),
            poll_interval=config['llm'].get('batch', {}).get('poll_interval', 300)
        )
        
        pseudo_labels = batch_labeler.resume_batch(
            batch_id=args.resume_batch,
            save_path=cache_file
        )
        
        print(f"\n✅ Resumed and completed batch job")
        print(f"Generated labels for {len(pseudo_labels)} samples")
        print(f"Saved to {cache_file}")
        return
    
    # Load data
    print("Loading data...")
    dfs = []
    for path in config['data']['parquet_paths']:
        if Path(path).exists():
            dfs.append(pd.read_parquet(path))
    df = pd.concat(dfs, ignore_index=True)
    
    # Use subset if specified
    num_samples = config['data']['num_samples']
    if num_samples > 0:
        df = df.iloc[:num_samples]
    
    print(f"Total samples: {len(df)}")
    
    # Extract reports (prefer impression over findings)
    reports = []
    for idx, row in df.iterrows():
        impression = row.get('impression', '')
        findings = row.get('findings', '')
        report = impression if pd.notna(impression) and impression.strip() else findings
        reports.append(report if pd.notna(report) else '')
    
    # Determine model type and mode
    model_type = config['llm'].get('model_type', 'qwen-max')
    mode = config['llm'].get('mode', 'realtime')
    
    # Preview model does not support batch mode
    if model_type == 'qwen-preview' and mode == 'batch':
        print("\n⚠️  Warning: qwen-preview model does NOT support batch mode")
        print("   Falling back to realtime mode\n")
        mode = 'realtime'
    
    if model_type == 'qwen-preview':
        print("\n" + "="*70)
        print("  QWEN3-MAX-PREVIEW MODE (with Thinking Capability)")
        print("="*70)
        print(f"Processing {len(reports)} reports using Preview API")
        print("Note: Preview model does not support batch mode")
        print("="*70 + "\n")
        
        # Initialize preview labeler
        labeler = QwenPreviewLabeler(
            api_key=api_key,
            base_url=config['llm']['base_url'],
            model_name='qwen3-max-preview',
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            label_names=config['data']['labels'],
            top_p=config['llm'].get('preview', {}).get('top_p', 0.8),
            thinking_budget=config['llm'].get('preview', {}).get('thinking_budget', 500)
        )
        
        # Generate labels
        pseudo_labels = labeler.batch_extract(
            reports=reports,
            batch_size=config['llm'].get('batch_size', 10),
            save_path=cache_file,
            max_workers=config['llm'].get('max_workers', 50)
        )
    
    elif mode == 'batch':
        print("\n" + "="*70)
        print("  BATCH MODE (50% Cost Savings)")
        print("="*70)
        print(f"Processing {len(reports)} reports using Batch API")
        print("This will create an async job and poll for completion.")
        print("="*70 + "\n")
        
        # Initialize batch labeler
        batch_labeler = QwenBatchLabeler(
            api_key=api_key,
            base_url=config['llm']['base_url'],
            model_name=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            label_names=config['data']['labels'],
            max_wait_hours=config['llm'].get('batch', {}).get('max_wait_hours', 24),
            poll_interval=config['llm'].get('batch', {}).get('poll_interval', 300)
        )
        
        # Determine temp directory
        temp_dir = config['llm'].get('batch', {}).get('temp_dir', 'temp_batch')
        if not os.path.isabs(temp_dir):
            if 'output_dir' in config.get('pipeline', {}):
                temp_dir = os.path.join(config['pipeline']['output_dir'], temp_dir)
        
        # Run batch extraction
        pseudo_labels = batch_labeler.batch_extract(
            reports=reports,
            save_path=cache_file,
            temp_dir=temp_dir
        )
        
    else:  # realtime mode with qwen-max
        print("\n" + "="*70)
        print("  REALTIME MODE")
        print("="*70)
        print(f"Processing {len(reports)} reports using Realtime API")
        print("💡 Tip: Use 'batch' mode for >1000 samples to save 50% cost")
        print("="*70 + "\n")
        
        # Initialize realtime labeler
        labeler = QwenLabeler(
            api_key=api_key,
            base_url=config['llm']['base_url'],
            model_name=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            label_names=config['data']['labels']
        )
        
        # Generate labels
        pseudo_labels = labeler.batch_extract(
            reports=reports,
            batch_size=config['llm']['batch_size'],
            save_path=cache_file
        )
    
    print(f"\nGenerated labels for {len(pseudo_labels)}/{len(reports)} samples")
    print(f"Saved to {cache_file}")
    
    # Print sample
    if pseudo_labels:
        sample_idx = list(pseudo_labels.keys())[0]
        print(f"\nSample label (index {sample_idx}):")
        print(json.dumps(pseudo_labels[sample_idx], indent=2))


if __name__ == '__main__':
    main()
