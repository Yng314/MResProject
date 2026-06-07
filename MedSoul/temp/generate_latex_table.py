"""
Generate LaTeX table from evaluation results
"""
import json
from pathlib import Path

# Data from terminal output
data = {
    "No Finding": {
        "n_samples": 30,
        "n_positive": 30,
        "n_negative": 0,
        "accuracy": 1.0000,
        "precision": None,
        "recall": None,
        "f1": None,
        "auc_roc": None,
        "avg_precision": None
    },
    "Enlarged Cardiomediastinum": {
        "n_samples": 114,
        "n_positive": 84,
        "n_negative": 30,
        "accuracy": 0.7632,
        "precision": 0.7664,
        "recall": 0.9762,
        "f1": 0.8586,
        "auc_roc": 0.8266,
        "avg_precision": 0.9220
    },
    "Cardiomegaly": {
        "n_samples": 279,
        "n_positive": 190,
        "n_negative": 89,
        "accuracy": 0.8065,
        "precision": 0.7881,
        "recall": 0.9789,
        "f1": 0.8732,
        "auc_roc": 0.8432,
        "avg_precision": 0.9171
    },
    "Lung Lesion": {
        "n_samples": 66,
        "n_positive": 62,
        "n_negative": 4,
        "accuracy": 0.9394,
        "precision": 0.9394,
        "recall": 1.0000,
        "f1": 0.9688,
        "auc_roc": 0.8105,
        "avg_precision": 0.9860
    },
    "Airspace Opacity": {
        "n_samples": 192,
        "n_positive": 169,
        "n_negative": 23,
        "accuracy": 0.8802,
        "precision": 0.8802,
        "recall": 1.0000,
        "f1": 0.9363,
        "auc_roc": 0.6734,
        "avg_precision": 0.9454
    },
    "Edema": {
        "n_samples": 283,
        "n_positive": 176,
        "n_negative": 107,
        "accuracy": 0.7915,
        "precision": 0.8980,
        "recall": 0.7500,
        "f1": 0.8173,
        "auc_roc": 0.9142,
        "avg_precision": 0.9432
    },
    "Consolidation": {
        "n_samples": 106,
        "n_positive": 82,
        "n_negative": 24,
        "accuracy": 0.8585,
        "precision": 0.8941,
        "recall": 0.9268,
        "f1": 0.9102,
        "auc_roc": 0.8664,
        "avg_precision": 0.9567
    },
    "Pneumonia": {
        "n_samples": 233,
        "n_positive": 143,
        "n_negative": 90,
        "accuracy": 0.7597,
        "precision": 0.7605,
        "recall": 0.8881,
        "f1": 0.8194,
        "auc_roc": 0.7908,
        "avg_precision": 0.8141
    },
    "Atelectasis": {
        "n_samples": 242,
        "n_positive": 237,
        "n_negative": 5,
        "accuracy": 0.9793,
        "precision": 0.9793,
        "recall": 1.0000,
        "f1": 0.9896,
        "auc_roc": 0.7992,
        "avg_precision": 0.9942
    },
    "Pneumothorax": {
        "n_samples": 238,
        "n_positive": 48,
        "n_negative": 190,
        "accuracy": 0.7731,
        "precision": 0.4250,
        "recall": 0.3542,
        "f1": 0.3864,
        "auc_roc": 0.7330,
        "avg_precision": 0.4085
    },
    "Pleural Effusion": {
        "n_samples": 379,
        "n_positive": 296,
        "n_negative": 83,
        "accuracy": 0.8707,
        "precision": 0.8896,
        "recall": 0.9527,
        "f1": 0.9201,
        "auc_roc": 0.9018,
        "avg_precision": 0.9695
    },
    "Pleural Other": {
        "n_samples": 27,
        "n_positive": 25,
        "n_negative": 2,
        "accuracy": 0.9259,
        "precision": 0.9259,
        "recall": 1.0000,
        "f1": 0.9615,
        "auc_roc": 0.9600,
        "avg_precision": 0.9970
    },
    "Fracture": {
        "n_samples": 50,
        "n_positive": 42,
        "n_negative": 8,
        "accuracy": 0.8600,
        "precision": 0.8571,
        "recall": 1.0000,
        "f1": 0.9231,
        "auc_roc": 0.7500,
        "avg_precision": 0.9438
    },
    "Support Devices": {
        "n_samples": 232,
        "n_positive": 227,
        "n_negative": 5,
        "accuracy": 0.9784,
        "precision": 0.9784,
        "recall": 1.0000,
        "f1": 0.9891,
        "auc_roc": 0.4687,
        "avg_precision": 0.9790
    }
}

overall = {
    "total_samples": 2471,
    "total_positive": 1811,
    "macro_avg": {
        "accuracy": 0.8705,
        "precision": 0.8448,
        "recall": 0.9098,
        "f1": 0.8733,
        "auc_roc": 0.7952,
        "avg_precision": 0.9059
    }
}

# Label names in order
label_names = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# Short labels for table header (to fit in LaTeX)
short_labels = [
    "No Finding",
    "Enlarged CM",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

def format_value(value, decimals=4):
    """Format value for LaTeX table"""
    if value is None:
        return "---"
    return f"{value:.{decimals}f}"

def generate_latex_table():
    """Generate LaTeX table"""
    
    # Start table
    latex = "\\begin{table*}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{l" + "c" * len(label_names) + "}\n"
    latex += "\\toprule\n"
    
    # Header row
    latex += "Metric & " + " & ".join(short_labels) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Samples row
    latex += "Samples & "
    samples_values = [str(data[label]["n_samples"]) for label in label_names]
    latex += " & ".join(samples_values) + " \\\\\n"
    
    # Accuracy row
    latex += "Accuracy & "
    acc_values = [format_value(data[label]["accuracy"]) for label in label_names]
    latex += " & ".join(acc_values) + " \\\\\n"
    
    # Precision row
    latex += "Precision & "
    prec_values = [format_value(data[label]["precision"]) for label in label_names]
    latex += " & ".join(prec_values) + " \\\\\n"
    
    # Recall row
    latex += "Recall & "
    recall_values = [format_value(data[label]["recall"]) for label in label_names]
    latex += " & ".join(recall_values) + " \\\\\n"
    
    # F1-Score row
    latex += "F1-Score & "
    f1_values = [format_value(data[label]["f1"]) for label in label_names]
    latex += " & ".join(f1_values) + " \\\\\n"
    
    # AUC-ROC row
    latex += "AUC-ROC & "
    auc_values = [format_value(data[label]["auc_roc"]) for label in label_names]
    latex += " & ".join(auc_values) + " \\\\\n"
    
    # Average Precision row
    latex += "Avg Precision & "
    ap_values = [format_value(data[label]["avg_precision"]) for label in label_names]
    latex += " & ".join(ap_values) + " \\\\\n"
    
    # Overall metrics row (macro average across all diseases)
    latex += "\\midrule\n"
    latex += "\\textbf{Overall (Macro Avg)} & "
    # For each metric row, we'll add the overall value
    # Since overall is macro average, we need to add it row by row
    # But for simplicity, we'll add it as a single row with values
    overall_values = []
    # Samples: total samples
    overall_values.append(str(overall["total_samples"]))
    # Accuracy: macro avg
    overall_values.append(format_value(overall["macro_avg"]["accuracy"]))
    # Precision: macro avg
    overall_values.append(format_value(overall["macro_avg"]["precision"]))
    # Recall: macro avg
    overall_values.append(format_value(overall["macro_avg"]["recall"]))
    # F1: macro avg
    overall_values.append(format_value(overall["macro_avg"]["f1"]))
    # AUC-ROC: macro avg
    overall_values.append(format_value(overall["macro_avg"]["auc_roc"]))
    # Avg Precision: macro avg
    overall_values.append(format_value(overall["macro_avg"]["avg_precision"]))
    # Pad with empty cells for remaining columns (we have 7 metrics, need 14 columns)
    overall_values.extend(["---"] * (len(label_names) - len(overall_values)))
    latex += " & ".join(overall_values[:len(label_names)]) + " \\\\\n"
    
    # End table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Evaluation results on MIMIC-CXR test set. Overall metrics are macro-averaged across all diseases.}\n"
    latex += "\\label{tab:evaluation_results}\n"
    latex += "\\end{table*}\n"
    
    return latex

if __name__ == "__main__":
    latex_table = generate_latex_table()
    
    # Save to file
    output_path = Path("temp/evaluation_table.tex")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print("LaTeX table generated successfully!")
    print(f"Saved to: {output_path}")
    print("\n" + "="*80)
    print(latex_table)
    print("="*80)

