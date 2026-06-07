"""
Generate LLM pseudo labels for test set
"""
import os
import sys
import json
import yaml
import zipfile
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.qwen_api import QwenLabeler
from utils.qwen_preview_api import QwenPreviewLabeler
from utils.gpt5_batch_api import GPT5BatchLabeler
from utils.deepseek_batch_api import DeepSeekBatchLabeler
from utils.deepseek_api import DeepSeekLabeler


def extract_reports_from_zip(zip_path, study_ids):
    """Extract reports for given study_ids from mimic-cxr-reports.zip"""
    print(f"Extracting reports from {zip_path}...")
    
    # MIMIC-CXR report structure: files/pXX/pXXXXXXXX/sXXXXXXXX.txt
    reports = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get all file names in the zip
        all_files = zf.namelist()
        
        for study_id in study_ids:
            # Find matching report file
            study_str = f"s{study_id}.txt"
            matching_files = [f for f in all_files if f.endswith(study_str)]
            
            if matching_files:
                # Read the report as-is, no parsing or section extraction
                with zf.open(matching_files[0]) as f:
                    content = f.read().decode('utf-8')
                    reports[study_id] = content.strip()
            else:
                print(f"Warning: Report not found for study_id {study_id}")
                reports[study_id] = ""
    
    return reports


def load_ground_truth(csv_path):
    """Load ground truth labels from test set CSV
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1.0 (present) -> 1 (positive mention)
    - 0.0 (explicitly absent) -> 1 (positive mention, explicitly said "no")
    - -1.0 (uncertain) -> 1 (positive mention, just uncertain)
    - NaN (not mentioned) -> 0 (negative, not mentioned at all)
    
    Only NaN means the finding was not mentioned at all, so only NaN is negative.
    """
    print(f"Loading ground truth from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Label columns (matching the CSV structure)
    label_cols = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
        "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation", 
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    ground_truth = {}
    for idx, row in df.iterrows():
        study_id = int(row['study_id'])
        labels = {}
        for label in label_cols:
            val = row[label]
            # For mention detection: 
            # - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
            # - NaN -> negative (not mentioned) -> 0
            if pd.isna(val):
                labels[label] = 0
            else:  # val == 1.0 or val == 0.0 or val == -1.0
                labels[label] = 1
        ground_truth[study_id] = labels
    
    return ground_truth, label_cols


def generate_llm_labels(reports, label_names, config, use_batch_mode=False):
    """Generate labels using LLM
    
    Args:
        reports: Dict of study_id -> report text
        label_names: List of label names
        config: Config dict
        use_batch_mode: If True, use batch API (50% cheaper, slower). If False, use realtime API.
                        Note: Qwen-preview does NOT support batch mode.
                        Note: GPT-5 currently ONLY supports batch mode.
                        Note: DeepSeek does NOT support batch API, will use realtime mode regardless.
    """
    print(f"\nGenerating LLM labels for {len(reports)} reports...")
    
    # Get project root for temp directory paths
    project_root = Path(__file__).parent.parent
    
    # Load API key
    load_dotenv()
    api_key = os.getenv(config['llm']['api_key_env'])
    if not api_key:
        raise ValueError(f"API key not found in environment variable {config['llm']['api_key_env']}")
    
    sorted_study_ids = sorted(reports.keys())
    report_list = [reports[sid] for sid in sorted_study_ids]
    
    # Determine which model to use
    model_type = config['llm'].get('model_type', 'qwen-max')
    
    if model_type == 'qwen-preview':
        # Use Preview model (with thinking capability, no batch support)
        print("Using QWEN3-MAX-PREVIEW MODEL (with thinking capability)")
        
        labeler = QwenPreviewLabeler(
            api_key=api_key,
            base_url=config['llm']['base_url'],
            model_name='qwen3-max-preview',
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            label_names=label_names,
            top_p=config['llm'].get('preview', {}).get('top_p', 0.8),
            thinking_budget=config['llm'].get('preview', {}).get('thinking_budget', 500)
        )
        
        # Use parallel processing with max_workers
        max_workers = config['llm'].get('max_workers', 20)
        pseudo_labels_dict = labeler.batch_extract(
            reports=report_list,
            batch_size=10,  # Not used in parallel mode
            save_path=None,
            max_workers=max_workers
        )
    
    elif model_type == 'qwen-max':
        # Use standard Qwen model (supports batch)
        if use_batch_mode:
            from utils.qwen_batch_api import QwenBatchLabeler
            print("Using QWEN-MAX in BATCH MODE (50% cost savings, will take longer)")
            
            batch_labeler = QwenBatchLabeler(
                api_key=api_key,
                base_url=config['llm']['base_url'],
                model_name=config['llm']['model_name'],
                temperature=config['llm']['temperature'],
                max_tokens=config['llm']['max_tokens'],
                label_names=label_names,
                max_wait_hours=config['llm'].get('batch', {}).get('max_wait_hours', 24),
                poll_interval=config['llm'].get('batch', {}).get('poll_interval', 300)
            )
            
            pseudo_labels_dict = batch_labeler.batch_extract(
                reports=report_list,
                save_path=None,
                temp_dir=str(project_root / 'temp_eval_batch')
            )
        else:
            print("Using QWEN-MAX in REALTIME MODE (faster, more expensive)")
            
            labeler = QwenLabeler(
                api_key=api_key,
                base_url=config['llm']['base_url'],
                model_name=config['llm']['model_name'],
                temperature=config['llm']['temperature'],
                max_tokens=config['llm']['max_tokens'],
                label_names=label_names
            )
            
            pseudo_labels_dict = labeler.batch_extract(
                reports=report_list,
                batch_size=10,
                save_path=None
            )
    elif model_type == 'gpt5' or model_type == 'gpt-5':
        # Use GPT-5 model
        if use_batch_mode:
            print("Using GPT-5 in BATCH MODE (50% cost savings, will take longer)")
            
            batch_labeler = GPT5BatchLabeler(
                api_key=None,  # Will read from OPENAI_API_KEY env variable
                model_name='gpt-5-2025-08-07',
                temperature=config['llm']['temperature'],
                max_tokens=config['llm']['max_tokens'],
                label_names=label_names,
                max_wait_hours=config['llm'].get('batch', {}).get('max_wait_hours', 24),
                poll_interval=config['llm'].get('batch', {}).get('poll_interval', 300)
            )
            
            pseudo_labels_dict = batch_labeler.batch_extract(
                reports=report_list,
                save_path=None,
                temp_dir=str(project_root / 'temp_eval_batch_gpt5')
            )
        else:
            raise ValueError("GPT-5 currently only supports batch mode. Please use --batch-mode flag.")
    
    elif model_type == 'deepseek' or model_type == 'deepseek-v3.2-exp':
        # Use DeepSeek model (realtime API only, batch not supported by DeepSeek)
        if use_batch_mode:
            print("⚠️  WARNING: DeepSeek does NOT support Batch API. Falling back to REALTIME mode.")
        
        # Get model name from config, default to deepseek-chat
        # Options: 'deepseek-chat' (standard) or 'deepseek-reasoner' (with thinking)
        deepseek_model = config['llm'].get('model_name', 'deepseek-chat')
        print(f"Using DeepSeek ({deepseek_model}) in REALTIME MODE")
        
        labeler = DeepSeekLabeler(
            api_key=None,  # Will read from DEEPSEEK_API_KEY env variable
            base_url="https://api.deepseek.com",
            model_name=deepseek_model,
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
            label_names=label_names
        )
        
        # Use parallel processing with max_workers
        max_workers = config['llm'].get('max_workers', 20)
        pseudo_labels_dict = labeler.batch_extract(
            reports=report_list,
            batch_size=10,  # Not used in parallel mode
            save_path=None,
            max_workers=max_workers
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'qwen-max', 'qwen-preview', 'gpt5', or 'deepseek'")
    
    # Convert indices to study_ids
    pseudo_labels = {}
    for idx_str, labels in pseudo_labels_dict.items():
        idx = int(idx_str)
        if idx < len(sorted_study_ids):
            study_id = sorted_study_ids[idx]
            pseudo_labels[study_id] = labels
    
    return pseudo_labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-mode', action='store_true', 
                       help='Use batch API mode (50%% cost savings, slower)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: pseudo_label_evaluation/test_set_pseudo_labels.json)')
    args = parser.parse_args()
    
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent
    test_csv = project_root / "datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv"
    reports_zip = project_root / "datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-reports.zip"
    config_path = project_root / "configs/config.yaml"
    
    # Load config
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)
    
    # Load ground truth
    ground_truth, label_names = load_ground_truth(str(test_csv))
    print(f"Loaded {len(ground_truth)} test samples with {len(label_names)} labels")
    
    # Extract reports
    study_ids = list(ground_truth.keys())
    reports = extract_reports_from_zip(str(reports_zip), study_ids)
    print(f"Extracted {len(reports)} reports")
    
    # Filter out empty reports
    valid_study_ids = [sid for sid in study_ids if reports.get(sid, "").strip()]
    print(f"Valid reports (non-empty): {len(valid_study_ids)}")
    
    # Generate LLM labels
    valid_reports = {sid: reports[sid] for sid in valid_study_ids}
    pseudo_labels = generate_llm_labels(valid_reports, label_names, config, use_batch_mode=args.batch_mode)
    
    # Set default output path if not provided
    if args.output is None:
        args.output = str(project_root / 'pseudo_label_evaluation/test_set_pseudo_labels.json')
    else:
        # If relative path, make it relative to project root
        if not os.path.isabs(args.output):
            args.output = str(project_root / args.output)
    
    # Save pseudo labels
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Convert study_id keys to strings for JSON serialization
    pseudo_labels_str_keys = {str(k): v for k, v in pseudo_labels.items()}
    
    output_data = {
        'num_samples': len(pseudo_labels),
        'pseudo_labels': pseudo_labels_str_keys
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nPseudo labels saved to {args.output}")


if __name__ == '__main__':
    main()

