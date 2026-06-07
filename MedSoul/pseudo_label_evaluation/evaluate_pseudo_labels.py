"""
Evaluate LLM-generated pseudo labels against ground truth test set
"""
import os
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


def load_labels_from_comparison_csv(csv_path):
    """Load ground truth and pseudo labels from detailed comparison CSV
    
    CSV format: Each finding has a column with "GT\LLM" format (e.g., "null\0", "1\1")
    - GT values: null, 1, 0, -1 (where null means not mentioned)
    - LLM values: null, 1, 0, -1 (where null means not mentioned)
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
    - null -> negative (not mentioned) -> 0
    """
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Label columns (matching the CSV structure)
    label_cols = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
        "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation", 
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    ground_truth = {}
    pseudo_labels = {}
    
    for idx, row in df.iterrows():
        study_id = int(row['study_id'])
        gt_labels = {}
        llm_labels = {}
        
        for label in label_cols:
            # Parse "GT\LLM" format
            combined_val = str(row[label])
            
            if '\\' in combined_val:
                parts = combined_val.split('\\')
                gt_val_str = parts[0].strip()
                llm_val_str = parts[1].strip() if len(parts) > 1 else 'null'
            else:
                # Fallback if format is different
                gt_val_str = combined_val
                llm_val_str = 'null'
            
            # Parse GT value for mention detection
            # - "null" -> 0 (not mentioned)
            # - "1", "0", "-1" -> 1 (mentioned)
            if gt_val_str == 'null' or gt_val_str == 'nan':
                gt_labels[label] = 0
            else:
                gt_labels[label] = 1
            
            # Parse LLM value for mention detection
            # - "null" -> 0 (not mentioned)
            # - "1", "0", "-1" -> 1 (mentioned)
            if llm_val_str == 'null' or llm_val_str == 'nan':
                llm_labels[label] = 0
            else:
                llm_labels[label] = 1
        
        ground_truth[study_id] = gt_labels
        pseudo_labels[study_id] = llm_labels
    
    return ground_truth, pseudo_labels, label_cols


def calculate_metrics(ground_truth, pseudo_labels, label_names):
    """Calculate Precision, Recall, F1 for each label
    
    Args:
        ground_truth: Dict[study_id -> Dict[label -> 0/1]]
        pseudo_labels: Dict[study_id -> Dict[label -> 0/1]]
        label_names: List of label names
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1 = mentioned (positive)
    - 0 = not mentioned (negative)
    """
    print("\nCalculating metrics...")
    
    results = []
    
    for label in label_names:
        y_true = []
        y_pred = []
        
        for study_id in ground_truth.keys():
            if study_id in pseudo_labels:
                gt_val = ground_truth[study_id][label]
                pred_val = pseudo_labels[study_id][label]
                
                y_true.append(gt_val)
                y_pred.append(pred_val)
        
        # Calculate metrics
        if len(y_true) > 0 and sum(y_true) > 0:  # Only if there are samples and positive cases
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            num_positive = sum(y_true)
            num_evaluated = len(y_true)
        else:
            precision, recall, f1, num_positive, num_evaluated = 0, 0, 0, 0, 0
        
        results.append({
            'Finding': label,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Positive Cases': num_positive,
            'Total Cases': num_evaluated
        })
    
    return results


def print_results(results):
    """Print results in a formatted table"""
    print("\n" + "="*110)
    print("  LLM Pseudo Label Evaluation Results (CheXpert Mention Detection)")
    print("="*110)
    print(f"{'Finding':<30} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Positive Cases':>15} {'Total Cases':>15}")
    print("-"*110)
    
    for r in results:
        print(f"{r['Finding']:<30} {r['Precision']:>12.3f} {r['Recall']:>12.3f} {r['F1']:>12.3f} {r['Positive Cases']:>15} {r['Total Cases']:>15}")
    
    print("="*110)
    
    # Calculate average metrics (macro average)
    avg_precision = np.mean([r['Precision'] for r in results if r['Positive Cases'] > 0])
    avg_recall = np.mean([r['Recall'] for r in results if r['Positive Cases'] > 0])
    avg_f1 = np.mean([r['F1'] for r in results if r['Positive Cases'] > 0])
    
    print(f"{'MACRO AVERAGE':<30} {avg_precision:>12.3f} {avg_recall:>12.3f} {avg_f1:>12.3f}")
    print("="*110)


def main():
    import argparse
    
    # Get project root for default paths
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-csv', type=str, 
                       default=None,
                       help='Path to detailed comparison CSV file (default: pseudo_label_evaluation/test_set_detailed_comparison.csv)')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Output JSON file path (default: pseudo_label_evaluation/test_set_metrics.json)')
    args = parser.parse_args()
    
    # Set default paths relative to project root
    if args.comparison_csv is None:
        args.comparison_csv = str(project_root / 'pseudo_label_evaluation/test_set_detailed_comparison.csv')
    elif not os.path.isabs(args.comparison_csv):
        args.comparison_csv = str(project_root / args.comparison_csv)
    
    if args.output is None:
        args.output = str(project_root / 'pseudo_label_evaluation/test_set_metrics.json')
    elif not os.path.isabs(args.output):
        args.output = str(project_root / args.output)
    
    # Load ground truth and pseudo labels from comparison CSV
    ground_truth, pseudo_labels, label_names = load_labels_from_comparison_csv(args.comparison_csv)
    print(f"Loaded {len(ground_truth)} samples with {len(label_names)} labels")
    
    # Calculate metrics
    results = calculate_metrics(ground_truth, pseudo_labels, label_names)
    
    # Print results
    print_results(results)
    
    # Save results (only metrics, no pseudo labels or ground truth)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        'num_samples': len(ground_truth),
        'results': results
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
