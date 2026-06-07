"""
Evaluation script for MIMIC-CXR classification model
Evaluates on the MIMIC-CXR 2.1.0 test set with ground truth labels
"""
import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import pandas as pd

from models.resnet import ResNetClassifier
from models.densenet import DenseNetClassifier
from torchvision import transforms


def load_config(config_path='configs/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_set(test_set_json='outputs/test_set_2_1_0.json'):
    """Load test set metadata and ground truth labels from JSON"""
    with open(test_set_json, 'r') as f:
        test_samples = json.load(f)
    return test_samples


def scan_test_sets(datasets_dir='datasets'):
    """Scan datasets directory for test label CSV files
    
    Returns:
        List of tuples: (display_name, file_path)
    """
    datasets_path = Path(datasets_dir)
    test_sets = []
    
    if not datasets_path.exists():
        return test_sets
    
    # Look for CSV files that might be test sets
    # Common patterns: *test*label*.csv, *label*.csv in datasets/
    patterns = [
        '*test*label*.csv',
        '*test*.csv',
        '*label*.csv'
    ]
    
    found_files = set()
    for pattern in patterns:
        for csv_file in datasets_path.glob(pattern):
            if csv_file.is_file() and csv_file.name not in found_files:
                found_files.add(csv_file.name)
                # Create display name from filename
                display_name = csv_file.stem.replace('_', ' ').title()
                test_sets.append((display_name, str(csv_file)))
    
    # Also check for the default JSON test set
    default_json = Path('outputs/test_set_2_1_0.json')
    if default_json.exists():
        test_sets.append(('Default JSON Test Set', str(default_json)))
    
    # Sort by display name
    test_sets.sort(key=lambda x: x[0])
    
    return test_sets


def load_test_set_from_csv(csv_path, split_csv='datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz', 
                           image_root='datasets/mimic-cxr-jpg-224', label_names=None):
    """Load test set from CSV file and convert to JSON format
    
    Args:
        csv_path: Path to CSV file with columns: study_id, and label columns
        split_csv: Path to split CSV file (contains subject_id, study_id, dicom_id mapping)
        image_root: Root directory containing MIMIC-CXR images (e.g., datasets/mimic-cxr-jpg-224)
        label_names: List of label names (if None, will infer from CSV columns)
    
    Returns:
        List of test samples in JSON format: [{'study_id': int, 'subject_id': int, 'images': [{'dicom_id': str}], 'labels': {label: value}}]
    """
    print(f"Loading test set from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get label names (exclude study_id column)
    if label_names is None:
        label_names = [col for col in df.columns if col != 'study_id']
    else:
        # Filter to only labels that exist in CSV
        label_names = [col for col in label_names if col in df.columns]
    
    print(f"Found {len(df)} test samples with {len(label_names)} labels")
    
    # Convert study_id to int
    df['study_id'] = df['study_id'].astype(int)
    test_study_ids = set(df['study_id'].unique())
    
    # Load split CSV to get subject_id and dicom_id for each study_id
    print(f"Loading split file to get image metadata: {split_csv}")
    split_df = pd.read_csv(split_csv)
    
    # Filter to only test study_ids
    split_test = split_df[split_df['study_id'].isin(test_study_ids)].copy()
    print(f"Found {len(split_test)} images for {len(test_study_ids)} test studies")
    
    # Group by study_id to get all images for each study
    study_images = {}
    for _, row in split_test.iterrows():
        study_id = int(row['study_id'])
        subject_id = int(row['subject_id'])
        dicom_id = row['dicom_id']
        
        if study_id not in study_images:
            study_images[study_id] = {
                'subject_id': subject_id,
                'images': []
            }
        study_images[study_id]['images'].append({'dicom_id': dicom_id})
    
    # Convert each row to test sample format
    test_samples = []
    missing_images = []
    
    for _, row in df.iterrows():
        study_id = int(row['study_id'])
        
        # Get images for this study_id
        if study_id in study_images:
            images_info = study_images[study_id]['images']
            subject_id = study_images[study_id]['subject_id']
        else:
            missing_images.append(study_id)
            # Skip studies without image metadata
            continue
        
        # Extract labels
        labels = {}
        for label_name in label_names:
            val = row.get(label_name)
            if pd.notna(val):
                labels[label_name] = float(val)
            else:
                labels[label_name] = None
        
        test_samples.append({
            'study_id': study_id,
            'subject_id': subject_id,
            'images': images_info,
            'labels': labels
        })
    
    if missing_images:
        print(f"Warning: {len(missing_images)} studies have no image metadata in split file (skipped)")
    
    print(f"Converted {len(test_samples)} samples to test set format")
    return test_samples


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    # Get model type (resnet or densenet)
    model_type = config['wsl'].get('model_type', 'resnet').lower()  # Default to resnet
    
    # Get encoder config
    encoder_config = config['wsl'].get('encoder', {})
    encoder_type = encoder_config.get('type', 'resnet50' if model_type == 'resnet' else 'densenet121')
    xrv_weights = encoder_config.get('xrv_weights', 'resnet50-res512-all' if model_type == 'resnet' else 'densenet121-res224-all')
    
    print(f"[INFO] Loading {model_type} model with encoder type: {encoder_type}")
    if encoder_type == 'xrv':
        print(f"[INFO] Using TorchXRayVision weights: {xrv_weights}")
    
    # Get number of labels
    num_labels = config['wsl'].get('num_labels', config['wsl'].get('num_classes', 14))
    
    # Determine num_classes_per_label based on loss type (must match training config)
    loss_type = config['wsl']['loss'].get('type', 'multi_class')
    if loss_type == 'binary':
        num_classes_per_label = 1  # Binary classification: 1 logit per label
        print(f"[INFO] Using binary classification mode (num_classes_per_label=1)")
    else:
        num_classes_per_label = config['wsl'].get('num_classes_per_label', 4)  # Multi-class: 4 classes
        print(f"[INFO] Using multi-class mode (num_classes_per_label={num_classes_per_label})")
    
    # Create model based on model_type
    if model_type == 'densenet':
        model = DenseNetClassifier(
            num_labels=num_labels,
            num_classes_per_label=num_classes_per_label,
            encoder_type=encoder_type,
            xrv_weights=xrv_weights,
            encoder_checkpoint=None,  # Don't load MAE checkpoint for evaluation
            freeze_encoder=False
        )
    else:  # Default to resnet
        model = ResNetClassifier(
            num_labels=num_labels,
            num_classes_per_label=num_classes_per_label,
            encoder_type=encoder_type,
            xrv_weights=xrv_weights,
            encoder_checkpoint=None,  # Don't load MAE checkpoint for evaluation
            freeze_encoder=False
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_image_transform(image_size=224):
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(
    model,
    test_samples,
    label_names,
    image_root='datasets/mimic-cxr-jpg-224',
    device='cuda',
    batch_size=16,
    image_size=224,
    is_binary_mode=False
):
    """
    Evaluate model on test set
    
    Note: Ground truth is sparse - only evaluate on samples where labels exist
    
    Args:
        image_size: Image input size (default: 224)
        is_binary_mode: If True, model outputs [B, num_labels] for binary classification
                       If False, model outputs [B, num_labels, num_classes] for multi-class
    """
    transform = get_image_transform(image_size)
    
    # Prepare data structures for each disease
    predictions = {label: [] for label in label_names}
    ground_truths = {label: [] for label in label_names}
    study_ids = {label: [] for label in label_names}
    
    # Also collect all predictions organized by study_id (for saving predictions file)
    all_predictions_by_study = {}
    
    print(f"\nEvaluating on {len(test_samples)} test samples...")
    
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Evaluating"):
            study_id = sample['study_id']
            labels = sample['labels']
            images_info = sample['images']
            
            # Load and preprocess image
            # Use the first image if multiple exist
            img_info = images_info[0]
            dicom_id = img_info['dicom_id']
            
            # Get subject_id from sample (if available) or try to infer
            subject_id = sample.get('subject_id')
            if subject_id is None:
                # Try to infer from study_id (this is a fallback, should not happen)
                print(f"Warning: No subject_id for study {study_id}, skipping")
                continue
            
            # Construct image path following MIMIC-CXR structure
            # Format: p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
            subject_prefix = str(subject_id)[:2]  # First 2 digits
            image_path = Path(image_root) / f'p{subject_prefix}' / f'p{subject_id}' / f's{study_id}' / f'{dicom_id}.jpg'
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get model prediction
            output = model(image_tensor)
            
            # Store all predictions for this study_id
            all_predictions_by_study[str(study_id)] = {}
            
            if is_binary_mode:
                # Binary mode: output shape [1, num_labels]
                # Apply sigmoid to get probabilities
                pred_probs = torch.sigmoid(output).cpu().numpy()[0]  # [num_labels]
                
                # Store predictions for each label
                for i, label_name in enumerate(label_names):
                    prob = pred_probs[i]
                    # Map to ground truth format: 1.0 (positive) or 0.0 (negative)
                    pred_value = 1.0 if prob > 0.5 else 0.0
                    all_predictions_by_study[str(study_id)][label_name] = pred_value
                
                # Store predictions and ground truth for metrics computation
                for i, label_name in enumerate(label_names):
                    if label_name in labels:
                        gt_value = labels[label_name]
                        
                        # Only include samples where ground truth exists
                        if gt_value is not None:
                            # For binary mode, use sigmoid probability directly
                            pos_prob = pred_probs[i]
                            predictions[label_name].append(pos_prob)
                            
                            # Convert ground truth to binary for metrics
                            if gt_value == 1.0:
                                ground_truths[label_name].append(1)
                            elif gt_value == 0.0:
                                ground_truths[label_name].append(0)
                            elif gt_value == -1.0:
                                # For uncertain cases: in binary mode we treat as positive
                                # So compare prediction against 1
                                ground_truths[label_name].append(1)
                            else:
                                # null case, exclude from metrics
                                predictions[label_name].pop()
                                continue
                            
                            study_ids[label_name].append(study_id)
            else:
                # Multi-class mode: output shape [1, num_labels, num_classes_per_label]
                # Apply softmax to get class probabilities
                pred_probs = torch.softmax(output, dim=-1).cpu().numpy()[0]  # [num_labels, num_classes_per_label]
                
                # Store predictions for each label
                for i, label_name in enumerate(label_names):
                    # Get predicted class (argmax)
                    pred_class = np.argmax(pred_probs[i])
                    
                    # Convert class index to label value
                    if pred_class == 0:
                        pred_value = 0.0  # Class 0: negative
                    elif pred_class == 1:
                        pred_value = 1.0  # Class 1: positive
                    elif pred_class == 2:
                        pred_value = -1.0  # Class 2: uncertain
                    else:  # pred_class == 3
                        pred_value = None  # Class 3: unlabeled
                    
                    all_predictions_by_study[str(study_id)][label_name] = pred_value
                
                # Store predictions and ground truth for metrics computation
                for i, label_name in enumerate(label_names):
                    if label_name in labels:
                        gt_value = labels[label_name]
                        
                        # Only include samples where ground truth exists
                        if gt_value is not None:
                            # Get probability of positive class (class 1) for metrics
                            pos_prob = pred_probs[i][1]  # Probability of positive class
                            predictions[label_name].append(pos_prob)
                            
                            # Convert ground truth to binary for metrics
                            # 1.0 -> 1 (positive)
                            # 0.0 -> 0 (negative)
                            # -1.0 -> exclude from binary metrics (uncertain)
                            # null -> exclude from metrics
                            if gt_value == 1.0:
                                ground_truths[label_name].append(1)
                            elif gt_value == 0.0:
                                ground_truths[label_name].append(0)
                            elif gt_value == -1.0:
                                # For uncertain cases, exclude from binary metrics
                                predictions[label_name].pop()  # Remove the prediction we just added
                                continue
                            else:
                                # null case, exclude from metrics
                                predictions[label_name].pop()
                                continue
                            
                            study_ids[label_name].append(study_id)
    
    return predictions, ground_truths, study_ids, all_predictions_by_study


def compute_metrics(predictions, ground_truths, label_names, threshold=0.5):
    """
    Compute evaluation metrics for each disease
    
    Args:
        predictions: Dict of {label_name: [pred_probs]}
        ground_truths: Dict of {label_name: [gt_labels]}
        label_names: List of disease names
        threshold: Classification threshold for binary predictions
    """
    results = {}
    
    print("\n" + "="*70)
    print("Per-Disease Evaluation Metrics")
    print("="*70)
    
    for label_name in label_names:
        preds = np.array(predictions[label_name])
        gts = np.array(ground_truths[label_name])
        
        if len(preds) == 0:
            print(f"\n{label_name}: No ground truth samples available")
            results[label_name] = {
                'n_samples': 0,
                'note': 'No ground truth available'
            }
            continue
        
        # Binary predictions
        pred_binary = (preds >= threshold).astype(int)
        
        # Check if we have both classes
        unique_gts = np.unique(gts)
        n_positive = np.sum(gts == 1)
        n_negative = np.sum(gts == 0)
        
        metrics = {
            'n_samples': len(preds),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'positive_rate': float(n_positive / len(preds)) if len(preds) > 0 else 0.0
        }
        
        # Compute metrics
        try:
            metrics['accuracy'] = float(accuracy_score(gts, pred_binary))
            
            # Precision, Recall, F1 (handle cases where one class is missing)
            if len(unique_gts) > 1:
                metrics['precision'] = float(precision_score(gts, pred_binary, zero_division=0))
                metrics['recall'] = float(recall_score(gts, pred_binary, zero_division=0))
                metrics['f1'] = float(f1_score(gts, pred_binary, zero_division=0))
                
                # AUC-ROC and Average Precision
                try:
                    metrics['auc_roc'] = float(roc_auc_score(gts, preds))
                    metrics['avg_precision'] = float(average_precision_score(gts, preds))
                except:
                    metrics['auc_roc'] = None
                    metrics['avg_precision'] = None
            else:
                metrics['precision'] = None
                metrics['recall'] = None
                metrics['f1'] = None
                metrics['auc_roc'] = None
                metrics['avg_precision'] = None
                metrics['note'] = f'Only one class present: {unique_gts[0]}'
            
            # Confusion matrix
            if len(unique_gts) > 1:
                cm = confusion_matrix(gts, pred_binary)
                metrics['confusion_matrix'] = cm.tolist()
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['true_negative'] = int(tn)
                    metrics['false_positive'] = int(fp)
                    metrics['false_negative'] = int(fn)
                    metrics['true_positive'] = int(tp)
        
        except Exception as e:
            print(f"Error computing metrics for {label_name}: {e}")
            metrics['error'] = str(e)
        
        results[label_name] = metrics
        
        # Print results
        print(f"\n{label_name}:")
        print(f"  Samples: {metrics['n_samples']} (Pos: {metrics['n_positive']}, Neg: {metrics['n_negative']})")
        if metrics.get('accuracy') is not None:
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if metrics.get('precision') is not None:
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('auc_roc') is not None:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
        if 'note' in metrics:
            print(f"  Note: {metrics['note']}")
    
    return results


def compute_overall_metrics(results, label_names):
    """Compute overall metrics across all diseases"""
    # Collect valid metrics
    valid_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc_roc': [],
        'avg_precision': []
    }
    
    total_samples = 0
    total_positive = 0
    
    for label_name in label_names:
        if label_name not in results:
            continue
        
        metrics = results[label_name]
        total_samples += metrics.get('n_samples', 0)
        total_positive += metrics.get('n_positive', 0)
        
        for metric_name in valid_metrics.keys():
            value = metrics.get(metric_name)
            if value is not None:
                valid_metrics[metric_name].append(value)
    
    # Compute averages
    overall = {
        'total_samples': total_samples,
        'total_positive': total_positive,
        'macro_avg': {}
    }
    
    for metric_name, values in valid_metrics.items():
        if len(values) > 0:
            overall['macro_avg'][metric_name] = float(np.mean(values))
    
    print("\n" + "="*70)
    print("Overall Metrics (Macro Average)")
    print("="*70)
    print(f"Total evaluation samples: {total_samples}")
    print(f"Total positive samples: {total_positive}")
    for metric_name, value in overall['macro_avg'].items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    return overall


def save_results(results, overall, output_path='outputs/evaluation_results.json'):
    """Save evaluation results to JSON file"""
    output = {
        'per_disease_metrics': results,
        'overall_metrics': overall
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def save_predictions(all_predictions_by_study, output_path='outputs/predictions.json'):
    """Save predictions in the format similar to pseudo_labels format
    
    Format:
    {
        "num_samples": int,
        "pseudo_labels": {
            "study_id": {
                "label_name": 0.0 or 1.0
            }
        }
    }
    """
    output = {
        'num_samples': len(all_predictions_by_study),
        'pseudo_labels': all_predictions_by_study
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nPredictions saved to: {output_path}")


def main():
    """Main evaluation function"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--test-set-index', type=int, default=None,
                       help='Index of test set to use (from auto-scanned list). Use --list-test-sets to see available options.')
    parser.add_argument('--test-set-path', type=str, default=None,
                       help='Direct path to test set file (CSV or JSON). Overrides auto-scan.')
    parser.add_argument('--list-test-sets', action='store_true',
                       help='List all available test sets and exit')
    args = parser.parse_args()
    
    # Scan for available test sets
    available_test_sets = scan_test_sets()
    
    # If --list-test-sets, print and exit
    if args.list_test_sets:
        print("="*70)
        print("Available Test Sets")
        print("="*70)
        if not available_test_sets:
            print("No test sets found in datasets/ directory")
        else:
            for idx, (display_name, file_path) in enumerate(available_test_sets, 1):
                print(f"{idx}. {display_name}")
                print(f"   Path: {file_path}")
                # Try to get file size
                if Path(file_path).exists():
                    size = Path(file_path).stat().st_size
                    if file_path.endswith('.csv'):
                        try:
                            df = pd.read_csv(file_path, nrows=0)
                            print(f"   Columns: {len(df.columns)} (including study_id)")
                        except:
                            pass
                    print(f"   Size: {size / 1024:.1f} KB")
                print()
        return
    
    print("="*70)
    print("MIMIC-CXR Model Evaluation")
    print("="*70)
    
    # Load configuration
    config = load_config(args.config)
    label_names = config['data']['labels']
    num_classes = len(label_names)
    device = config['training']['device']
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Labels: {', '.join(label_names)}")
    
    # Load test set
    print("\n" + "="*70)
    print("Available Test Sets:")
    print("="*70)
    
    if not available_test_sets:
        print("No test sets found! Please check datasets/ directory.")
        return
    
    for idx, (display_name, file_path) in enumerate(available_test_sets, 1):
        print(f"{idx}. {display_name} ({file_path})")
    
    # Determine which test set to use
    if args.test_set_path:
        # Direct path provided
        test_set_path = args.test_set_path
        print(f"\nUsing test set from direct path: {test_set_path}")
    elif args.test_set_index:
        # Index provided
        if 1 <= args.test_set_index <= len(available_test_sets):
            display_name, test_set_path = available_test_sets[args.test_set_index - 1]
            print(f"\nUsing test set [{args.test_set_index}]: {display_name}")
        else:
            print(f"\nError: Invalid test set index {args.test_set_index}")
            print(f"Please choose between 1 and {len(available_test_sets)}")
            return
    else:
        # No selection provided - prompt user for interactive selection
        print(f"\nPlease select a test set (1-{len(available_test_sets)}):")
        
        while True:
            try:
                user_input = input("Enter test set number: ").strip()
                if not user_input:
                    # If user just presses Enter, use first CSV test set as default
                    csv_test_sets = [(name, path) for name, path in available_test_sets if path.endswith('.csv')]
                    if csv_test_sets:
                        display_name, test_set_path = csv_test_sets[0]
                        print(f"Using default (first CSV test set): {display_name}")
                    elif available_test_sets:
                        display_name, test_set_path = available_test_sets[0]
                        print(f"Using default (first available): {display_name}")
                    else:
                        print("\nError: No test sets found!")
                        return
                    break
                
                selected_index = int(user_input)
                if 1 <= selected_index <= len(available_test_sets):
                    display_name, test_set_path = available_test_sets[selected_index - 1]
                    print(f"\nSelected: {display_name}")
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(available_test_sets)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nCancelled by user.")
                return
    
    # Load the selected test set
    print(f"\nLoading test set: {test_set_path}")
    if test_set_path.endswith('.csv'):
        # Get split CSV and image root from config
        split_csv = config.get('data', {}).get('mimic_jpg', {}).get('split_csv', 
                   'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz')
        image_root = config.get('data', {}).get('mimic_jpg', {}).get('image_root',
                   'datasets/mimic-cxr-jpg-224')
        
        test_samples = load_test_set_from_csv(
            test_set_path,
            split_csv=split_csv,
            image_root=image_root,
            label_names=label_names
        )
    else:
        test_samples = load_test_set(test_set_path)
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Find the latest model checkpoint
    # Try fine-tuned model first, then linear probe
    checkpoint_dir = config['wsl']['checkpoint_dir']
    # If relative path, resolve it relative to output_dir
    if not os.path.isabs(checkpoint_dir):
        if 'output_dir' in config.get('pipeline', {}):
            checkpoint_dir = os.path.join(config['pipeline']['output_dir'], checkpoint_dir)
    
    checkpoint_dir = Path(checkpoint_dir)
    
    possible_checkpoints = [
        checkpoint_dir / 'fine_tune_best.pth',
        checkpoint_dir / 'fine_tune_last.pth',
        checkpoint_dir / 'linear_probe_best.pth',
        checkpoint_dir / 'linear_probe_last.pth'
    ]
    
    checkpoint_path = None
    for cp in possible_checkpoints:
        if cp.exists():
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        print("\nError: No trained model checkpoint found!")
        print(f"Searched in: {checkpoint_dir}")
        print("Please train the model first using main.py")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    print("Model loaded successfully")
    
    # Get image size from config (prefer wsl.image_size, fallback to mae.image_size, default 224)
    image_size = config.get('wsl', {}).get('image_size', 
                                           config.get('mae', {}).get('image_size', 224))
    print(f"[INFO] Using image size: {image_size}")
    
    # Determine if model is in binary mode
    loss_type = config['wsl']['loss'].get('type', 'multi_class')
    is_binary_mode = (loss_type == 'binary')
    print(f"[INFO] Evaluation mode: {'binary' if is_binary_mode else 'multi-class'}")
    
    # Get image root from config
    image_root = config.get('data', {}).get('mimic_jpg', {}).get('image_root',
               'datasets/mimic-cxr-jpg-224')
    
    # Evaluate
    predictions, ground_truths, study_ids, all_predictions_by_study = evaluate_model(
        model=model,
        test_samples=test_samples,
        label_names=label_names,
        image_root=image_root,
        device=device,
        image_size=image_size,
        is_binary_mode=is_binary_mode
    )
    
    # Compute metrics
    results = compute_metrics(predictions, ground_truths, label_names)
    overall = compute_overall_metrics(results, label_names)
    
    # Save results to experiment directory
    output_path = Path(config['pipeline']['output_dir']) / 'evaluation_results.json'
    save_results(results, overall, str(output_path))
    
    # Save predictions in pseudo_labels format
    predictions_path = Path(config['pipeline']['output_dir']) / 'test_set_predictions.json'
    save_predictions(all_predictions_by_study, str(predictions_path))
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

