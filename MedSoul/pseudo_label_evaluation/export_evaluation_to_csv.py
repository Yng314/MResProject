"""
Export evaluation results to CSV for detailed comparison
Includes: study_id, report text, pseudo labels, ground truth labels
"""
import json
import zipfile
import csv
import pandas as pd
from pathlib import Path
import os


def extract_reports_from_zip(zip_path, study_ids):
    """Extract reports for given study_ids from mimic-cxr-reports.zip"""
    print(f"Extracting reports from {zip_path}...")
    
    reports = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = zf.namelist()
        
        for study_id in study_ids:
            study_str = f"s{study_id}.txt"
            matching_files = [f for f in all_files if f.endswith(study_str)]
            
            if matching_files:
                # Read the report as-is, no parsing or section extraction
                with zf.open(matching_files[0]) as f:
                    content = f.read().decode('utf-8')
                    reports[study_id] = content.strip()
            else:
                reports[study_id] = ""
    
    return reports


def format_label_value(val):
    """Format label value for display: 1.0 -> 1, -1.0 -> -1, 0.0 -> 0, None -> 'null'"""
    if val is None:
        return 'null'
    elif val == 1.0:
        return '1'
    elif val == -1.0:
        return '-1'
    elif val == 0.0:
        return '0'
    else:
        return str(val)


def check_match(gt_val, pseudo_val):
    """Check if GT and LLM labels match (simple value comparison)
    
    Returns True only if GT and LLM values are exactly the same.
    - 1.0 == 1.0 -> True
    - 1.0 == -1.0 -> False
    - None == None -> True
    - 1.0 == None -> False
    """
    # Simple value comparison: exact match only
    return gt_val == pseudo_val


def load_ground_truth_from_csv(csv_path, study_ids):
    """Load original ground truth values from CSV file (1.0, 0.0, -1.0, NaN)"""
    print(f"Loading original ground truth from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    label_cols = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
        "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation", 
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    ground_truth = {}
    study_ids_set = set(study_ids)
    
    for idx, row in df.iterrows():
        study_id = int(row['study_id'])
        if study_id in study_ids_set:
            labels = {}
            for label in label_cols:
                val = row[label]
                # Keep original value: 1.0, 0.0, -1.0, or NaN
                if pd.isna(val):
                    labels[label] = None
                else:
                    labels[label] = float(val)
            ground_truth[study_id] = labels
    
    return ground_truth, label_cols


def create_comparison_csv(results_json, reports_zip, gt_csv_path, output_csv):
    """Create a CSV with study_id, report, pseudo labels, and ground truth
    
    Format: | GT/LLM | study_id | No Finding | Matching | Enlarged Cardiomediastinum | Matching | ... | Report |
    """
    
    # Load evaluation results
    print(f"Loading evaluation results from {results_json}...")
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    pseudo_labels = data['pseudo_labels']
    
    # Convert study_id strings to integers
    study_ids = [int(sid) for sid in pseudo_labels.keys()]
    
    # Load original ground truth from CSV (with original values: 1.0, 0.0, -1.0, NaN)
    ground_truth, label_names = load_ground_truth_from_csv(gt_csv_path, study_ids)
    
    # Extract reports
    reports = extract_reports_from_zip(reports_zip, study_ids)
    
    print(f"Creating CSV with {len(study_ids)} studies and {len(label_names)} labels...")
    
    # Build rows for CSV
    rows = []
    for study_id in sorted(study_ids):
        sid_str = str(study_id)
        
        row = {
            'GT/LLM': 'GT/LLM',  # Identifier row
            'study_id': study_id,
        }
        
        # For each label, add GT/LLM value column and Matching column
        for label in label_names:
            gt_val = ground_truth.get(study_id, {}).get(label, None)
            pseudo_val_raw = pseudo_labels[sid_str].get(label, None)
            
            # Convert JSON string "null" to None, and string numbers to float
            if pseudo_val_raw == "null" or pseudo_val_raw is None:
                pseudo_val = None
            elif isinstance(pseudo_val_raw, str):
                # Try to convert string to float (e.g., "1.0" -> 1.0, "-1.0" -> -1.0)
                try:
                    pseudo_val = float(pseudo_val_raw)
                except (ValueError, TypeError):
                    pseudo_val = None
            else:
                pseudo_val = pseudo_val_raw
            
            # Format GT\LLM value: "GT_val\LLM_val" (using backslash to avoid Excel date format)
            gt_str = format_label_value(gt_val)
            pseudo_str = format_label_value(pseudo_val)
            
            # Always show both values separated by backslash
            row[label] = f"{gt_str}\\{pseudo_str}"
            
            # Check if they match for mention detection
            matches = check_match(gt_val, pseudo_val)
            row[f'{label}_Matching'] = 'Yes' if matches else 'No'
        
        # Add report at the end
        row['Report'] = reports.get(study_id, "")
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns: GT/LLM, study_id, then for each label: label, label_Matching, then Report
    ordered_cols = ['GT/LLM', 'study_id']
    for label in label_names:
        ordered_cols.extend([
            label,
            f'{label}_Matching'
        ])
    ordered_cols.append('Report')
    
    df = df[ordered_cols]
    
    # Rename columns to match template format
    # First column should be empty, then study_id, then label pairs
    new_cols = ['', 'study_id']
    for label in label_names:
        new_cols.extend([label, 'Matching'])
    new_cols.append('Report')
    
    # Note: pandas doesn't support duplicate column names, so we'll use unique names
    # but write CSV manually to allow "Matching" as repeated column name
    df.columns = ordered_cols  # Keep original unique names for processing
    
    # Write CSV manually to allow "Matching" as repeated column name
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with repeated "Matching"
        header = ['', 'study_id']
        for label in label_names:
            header.extend([label, 'Matching'])
        header.append('Report')
        writer.writerow(header)
        
        # Write data rows
        for _, row in df.iterrows():
            values = [row['GT/LLM'], str(row['study_id'])]
            for label in label_names:
                values.append(str(row[label]))
                values.append(str(row[f'{label}_Matching']))
            # Keep report text as-is (csv.writer will handle escaping)
            values.append(str(row['Report']))
            writer.writerow(values)
    print(f"\nSaved comparison CSV to: {output_csv}")
    print(f"   - {len(df)} studies")
    print(f"   - {len(label_names)} labels")
    print(f"   - Format: GT/LLM | study_id | [Label | Matching]* | Report")
    
    # Print some statistics
    print("\nMatch Statistics:")
    for label in label_names:
        match_col = f'{label}_Matching'
        match_count = (df[match_col] == 'Yes').sum()
        match_rate = match_count / len(df) * 100
        print(f"   {label:<30} {match_rate:>6.2f}% match ({match_count}/{len(df)})")
    
    # Calculate overall match rate
    match_cols = [f'{label}_Matching' for label in label_names]
    overall_match = df[match_cols].apply(lambda row: (row == 'Yes').sum() / len(row), axis=1).mean() * 100
    print(f"\n   {'OVERALL':<30} {overall_match:>6.2f}% match")


def main():
    # Get project root for paths
    project_root = Path(__file__).parent.parent
    
    results_json = str(project_root / "pseudo_label_evaluation/test_set_pseudo_labels.json")
    reports_zip = str(project_root / "datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-reports.zip")
    gt_csv_path = str(project_root / "datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv")
    output_csv = str(project_root / "pseudo_label_evaluation/test_set_detailed_comparison.csv")
    
    create_comparison_csv(results_json, reports_zip, gt_csv_path, output_csv)


if __name__ == '__main__':
    main()

