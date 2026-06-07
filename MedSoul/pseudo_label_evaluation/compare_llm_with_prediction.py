"""
Compare model predictions with ground truth using mention detection
"""
import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


def load_ground_truth_from_json(json_path):
    """Load ground truth labels from test_set_2_1_0.json
    
    JSON format: Array of objects, each with:
    {
        "study_id": int,
        "labels": {
            "Finding": value (1.0, 0.0, -1.0, or null)
        }
    }
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
    - null -> negative (not mentioned) -> 0
    """
    print(f"Loading ground truth from {json_path}...")
    with open(json_path, 'r') as f:
        test_samples = json.load(f)
    
    labels_dict = {}
    label_names = None
    
    for sample in test_samples:
        study_id = int(sample['study_id'])
        labels = {}
        
        # Get label names from first sample
        if label_names is None:
            label_names = list(sample['labels'].keys())
        
        for label in label_names:
            val = sample['labels'].get(label)
            # For mention detection:
            # - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
            # - null -> negative (not mentioned) -> 0
            if val is None or (isinstance(val, float) and np.isnan(val)):
                labels[label] = 0
            else:  # val == 1.0 or val == 0.0 or val == -1.0
                labels[label] = 1
        
        labels_dict[study_id] = labels
    
    return labels_dict, label_names


def load_predictions_from_json(json_path):
    """Load model predictions from JSON file
    
    JSON format: 
    {
        "num_samples": int,
        "pseudo_labels": {
            "study_id": {
                "Finding": value (1.0, 0.0, -1.0, or null)
            }
        }
    }
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
    - null -> negative (not mentioned) -> 0
    """
    print(f"Loading model predictions from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    labels_dict = {}
    label_names = None
    
    for study_id_str, findings in data['pseudo_labels'].items():
        study_id = int(study_id_str)
        labels = {}
        
        # Get label names from first sample
        if label_names is None:
            label_names = list(findings.keys())
        
        for label in label_names:
            val = findings.get(label)
            # For mention detection:
            # - 1.0, 0.0, or -1.0 -> positive (mentioned) -> 1
            # - null -> negative (not mentioned) -> 0
            if val is None or (isinstance(val, float) and np.isnan(val)):
                labels[label] = 0
            else:  # val == 1.0 or val == 0.0 or val == -1.0
                labels[label] = 1
        
        labels_dict[study_id] = labels
    
    return labels_dict, label_names


def calculate_metrics(ground_truth, predictions, label_names):
    """Calculate Precision, Recall, F1 for each label
    
    Args:
        ground_truth: Dict[study_id -> Dict[label -> 0/1]] (ground truth)
        predictions: Dict[study_id -> Dict[label -> 0/1]] (model predictions)
        label_names: List of label names
    
    For mention detection (CheXpert Table 4 evaluation):
    "detect any utterance of the finding, regardless of uncertainty"
    - 1 = mentioned (positive)
    - 0 = not mentioned (negative)
    
    Metrics are calculated with ground truth as reference and model predictions as predictions.
    """
    print("\nCalculating metrics (Ground Truth as reference, Model Predictions as predictions)...")
    
    results = []
    
    for label in label_names:
        y_true = []  # Ground truth labels
        y_pred = []  # Model predictions
        
        # Only evaluate on samples that exist in both
        common_study_ids = set(ground_truth.keys()) & set(predictions.keys())
        
        for study_id in common_study_ids:
            gt_val = ground_truth[study_id][label]
            pred_val = predictions[study_id][label]
            
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
            'Positive Cases (GT)': num_positive,
            'Total Cases': num_evaluated
        })
    
    return results


def print_results(results):
    """Print results in a formatted table"""
    print("\n" + "="*120)
    print("  Model Predictions vs Ground Truth (CheXpert Mention Detection)")
    print("="*120)
    print(f"{'Finding':<30} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Positive Cases (GT)':>20} {'Total Cases':>15}")
    print("-"*120)
    
    for r in results:
        print(f"{r['Finding']:<30} {r['Precision']:>12.3f} {r['Recall']:>12.3f} {r['F1']:>12.3f} {r['Positive Cases (GT)']:>20} {r['Total Cases']:>15}")
    
    print("="*120)
    
    # Calculate average metrics (macro average)
    results_with_positives = [r for r in results if r['Positive Cases (GT)'] > 0]
    if results_with_positives:
        avg_precision = np.mean([r['Precision'] for r in results_with_positives])
        avg_recall = np.mean([r['Recall'] for r in results_with_positives])
        avg_f1 = np.mean([r['F1'] for r in results_with_positives])
        
        print(f"{'MACRO AVERAGE':<30} {avg_precision:>12.3f} {avg_recall:>12.3f} {avg_f1:>12.3f}")
        print("="*120)


def list_experiments(outputs_dir):
    """List all experiment folders in outputs directory"""
    experiments = []
    if not os.path.exists(outputs_dir):
        return experiments
    
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if os.path.isdir(item_path):
            # Check if test_set_predictions.json exists in this folder
            predictions_file = os.path.join(item_path, 'test_set_predictions.json')
            if os.path.exists(predictions_file):
                experiments.append(item)
    
    return sorted(experiments)


def select_experiment(experiments):
    """Let user select an experiment from the list"""
    if not experiments:
        print("No experiments found in outputs directory with test_set_predictions.json")
        return None
    
    print("\n" + "="*80)
    print("Available Experiments:")
    print("="*80)
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp}")
    print("="*80)
    
    while True:
        try:
            choice = input(f"\nSelect experiment (1-{len(experiments)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return None


def list_llm_metrics(metrics_dir):
    """List all LLM metrics JSON files in pseudo_label_evaluation directory"""
    metrics_files = []
    if not os.path.exists(metrics_dir):
        return metrics_files
    
    for item in os.listdir(metrics_dir):
        if item.endswith('.json') and 'metrics' in item.lower():
            metrics_files.append(item)
    
    return sorted(metrics_files)


def select_llm_metrics(metrics_files):
    """Let user select an LLM metrics file from the list"""
    if not metrics_files:
        print("No LLM metrics JSON files found in pseudo_label_evaluation directory")
        return None
    
    print("\n" + "="*80)
    print("Available LLM Metrics Files:")
    print("="*80)
    print("0. Skip comparison with LLM")
    for i, metrics_file in enumerate(metrics_files, 1):
        print(f"{i}. {metrics_file}")
    print("="*80)
    
    while True:
        try:
            choice = input(f"\nSelect LLM metrics (0-{len(metrics_files)}, 0 to skip): ").strip()
            idx = int(choice)
            if idx == 0:
                return None
            elif 1 <= idx <= len(metrics_files):
                return metrics_files[idx - 1]
            else:
                print(f"Please enter a number between 0 and {len(metrics_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return None


def load_llm_metrics(json_path):
    """Load LLM metrics from JSON file"""
    print(f"Loading LLM metrics from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert results list to dict for easy lookup
    metrics_dict = {}
    for result in data['results']:
        finding = result['Finding']
        metrics_dict[finding] = {
            'Precision': result['Precision'],
            'Recall': result['Recall'],
            'F1': result['F1']
        }
    
    return metrics_dict, data.get('num_samples', 0)


def print_comparison_results(model_results, llm_metrics, llm_name):
    """Print comparison between model and LLM metrics"""
    print("\n" + "="*150)
    print(f"  Model vs LLM Comparison (LLM: {llm_name})")
    print("="*150)
    print(f"{'Finding':<30} {'Model P':>10} {'LLM P':>10} {'Diff':>10} {'Model R':>10} {'LLM R':>10} {'Diff':>10} {'Model F1':>10} {'LLM F1':>10} {'Diff':>10}")
    print("-"*150)
    
    model_wins = {'precision': 0, 'recall': 0, 'f1': 0}
    llm_wins = {'precision': 0, 'recall': 0, 'f1': 0}
    
    for r in model_results:
        finding = r['Finding']
        if finding in llm_metrics:
            llm = llm_metrics[finding]
            
            p_diff = r['Precision'] - llm['Precision']
            r_diff = r['Recall'] - llm['Recall']
            f1_diff = r['F1'] - llm['F1']
            
            # Track wins
            if p_diff > 0:
                model_wins['precision'] += 1
            elif p_diff < 0:
                llm_wins['precision'] += 1
            
            if r_diff > 0:
                model_wins['recall'] += 1
            elif r_diff < 0:
                llm_wins['recall'] += 1
            
            if f1_diff > 0:
                model_wins['f1'] += 1
            elif f1_diff < 0:
                llm_wins['f1'] += 1
            
            output_line = f"{finding:<30} {r['Precision']:>10.3f} {llm['Precision']:>10.3f} {p_diff:>+10.3f} {r['Recall']:>10.3f} {llm['Recall']:>10.3f} {r_diff:>+10.3f} {r['F1']:>10.3f} {llm['F1']:>10.3f} {f1_diff:>+10.3f}"
            print(output_line)
        else:
            output_line = f"{finding:<30} {r['Precision']:>10.3f} {'N/A':>10} {'N/A':>10} {r['Recall']:>10.3f} {'N/A':>10} {'N/A':>10} {r['F1']:>10.3f} {'N/A':>10} {'N/A':>10}"
            print(output_line)
    
    print("="*150)
    
    # Calculate average metrics
    results_with_positives = [r for r in model_results if r['Positive Cases (GT)'] > 0]
    if results_with_positives:
        model_avg_p = np.mean([r['Precision'] for r in results_with_positives])
        model_avg_r = np.mean([r['Recall'] for r in results_with_positives])
        model_avg_f1 = np.mean([r['F1'] for r in results_with_positives])
        
        # Calculate LLM averages for matching findings
        llm_p_list = []
        llm_r_list = []
        llm_f1_list = []
        for r in results_with_positives:
            if r['Finding'] in llm_metrics:
                llm = llm_metrics[r['Finding']]
                llm_p_list.append(llm['Precision'])
                llm_r_list.append(llm['Recall'])
                llm_f1_list.append(llm['F1'])
        
        if llm_p_list:
            llm_avg_p = np.mean(llm_p_list)
            llm_avg_r = np.mean(llm_r_list)
            llm_avg_f1 = np.mean(llm_f1_list)
            
            p_diff_avg = model_avg_p - llm_avg_p
            r_diff_avg = model_avg_r - llm_avg_r
            f1_diff_avg = model_avg_f1 - llm_avg_f1
            
            macro_avg_line = f"{'MACRO AVERAGE':<30} {model_avg_p:>10.3f} {llm_avg_p:>10.3f} {p_diff_avg:>+10.3f} {model_avg_r:>10.3f} {llm_avg_r:>10.3f} {r_diff_avg:>+10.3f} {model_avg_f1:>10.3f} {llm_avg_f1:>10.3f} {f1_diff_avg:>+10.3f}"
            print(macro_avg_line)
            print("="*150)
    
    # Print summary
    print(f"\nComparison Summary:")
    print(f"  Model wins - Precision: {model_wins['precision']}, Recall: {model_wins['recall']}, F1: {model_wins['f1']}")
    print(f"  LLM wins   - Precision: {llm_wins['precision']}, Recall: {llm_wins['recall']}, F1: {llm_wins['f1']}")
    print("="*150)


def main():
    import argparse
    
    # Get project root for default paths
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=str, 
                       default=None,
                       help='Path to ground truth JSON file (default: outputs/test_set_2_1_0.json)')
    parser.add_argument('--experiment', type=str,
                       default=None,
                       help='Experiment folder name (if not provided, will prompt for selection)')
    args = parser.parse_args()
    
    # Set ground truth path
    if args.ground_truth is None:
        args.ground_truth = str(project_root / 'outputs/test_set_2_1_0.json')
    elif not os.path.isabs(args.ground_truth):
        args.ground_truth = str(project_root / args.ground_truth)
    
    # Select experiment
    outputs_dir = project_root / 'outputs'
    if args.experiment is None:
        # List and let user select
        experiments = list_experiments(str(outputs_dir))
        selected_exp = select_experiment(experiments)
        if selected_exp is None:
            return
    else:
        selected_exp = args.experiment
    
    # Set paths based on selected experiment
    experiment_dir = outputs_dir / selected_exp
    args.model_predictions = str(experiment_dir / 'test_set_predictions.json')
    args.output = str(experiment_dir / 'model_vs_gt_comparison.json')
    
    # Verify predictions file exists
    if not os.path.exists(args.model_predictions):
        print(f"Error: test_set_predictions.json not found in {experiment_dir}")
        return
    
    print(f"\nSelected experiment: {selected_exp}")
    print(f"Model predictions: {args.model_predictions}")
    print(f"Output will be saved to: {args.output}")
    
    # Load ground truth
    ground_truth, label_names = load_ground_truth_from_json(args.ground_truth)
    print(f"Loaded {len(ground_truth)} ground truth samples with {len(label_names)} labels")
    
    # Load model predictions
    predictions, pred_label_names = load_predictions_from_json(args.model_predictions)
    print(f"Loaded {len(predictions)} model prediction samples with {len(pred_label_names)} labels")
    
    # Verify label names match
    if set(label_names) != set(pred_label_names):
        print("\nWarning: Label names don't match between ground truth and predictions!")
        print(f"Ground truth labels: {label_names}")
        print(f"Prediction labels: {pred_label_names}")
        # Use intersection
        label_names = [l for l in label_names if l in pred_label_names]
        print(f"Using intersection: {label_names}")
    
    # Calculate metrics
    results = calculate_metrics(ground_truth, predictions, label_names)
    
    # Print results
    print_results(results)
    
    # Select LLM metrics for comparison
    llm_metrics_dir = project_root / 'pseudo_label_evaluation'
    metrics_files = list_llm_metrics(str(llm_metrics_dir))
    selected_llm_metrics = select_llm_metrics(metrics_files)
    
    llm_comparison_data = None
    if selected_llm_metrics:
        llm_metrics_path = llm_metrics_dir / selected_llm_metrics
        llm_metrics, llm_num_samples = load_llm_metrics(str(llm_metrics_path))
        print(f"Loaded LLM metrics with {llm_num_samples} samples\n")
        
        # Print comparison
        print_comparison_results(results, llm_metrics, selected_llm_metrics)
        
        # Prepare comparison data for JSON output
        llm_comparison_data = {
            'llm_metrics_file': selected_llm_metrics,
            'llm_num_samples': llm_num_samples,
            'comparison': []
        }
        
        for r in results:
            finding = r['Finding']
            if finding in llm_metrics:
                llm = llm_metrics[finding]
                llm_comparison_data['comparison'].append({
                    'Finding': finding,
                    'Model': {
                        'Precision': r['Precision'],
                        'Recall': r['Recall'],
                        'F1': r['F1']
                    },
                    'LLM': {
                        'Precision': llm['Precision'],
                        'Recall': llm['Recall'],
                        'F1': llm['F1']
                    },
                    'Difference': {
                        'Precision': r['Precision'] - llm['Precision'],
                        'Recall': r['Recall'] - llm['Recall'],
                        'F1': r['F1'] - llm['F1']
                    }
                })
    
    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    output_data = {
        'num_gt_samples': len(ground_truth),
        'num_pred_samples': len(predictions),
        'num_common_samples': len(set(ground_truth.keys()) & set(predictions.keys())),
        'results': results
    }
    
    if llm_comparison_data:
        output_data['llm_comparison'] = llm_comparison_data
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

