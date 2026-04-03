"""
Data Exploration Script - Phase 0
Analyze GT dataset characteristics to inform subsequent development
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Configuration
GT_METADATA_PATH = "d:/workspace/MRes/datasets/mimic-cxr-clean/gt/metadata.csv"
GT_IMAGE_BASE_PATH = "d:/workspace/MRes/datasets/mimic-cxr-clean/gt"
ALLOWED_VIEW_POSITIONS = ['AP', 'PA']

# CheXpert 14 pathology classes
CHEXPERT_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices'
]


def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def load_and_filter_data():
    """Load data and filter by ViewPosition"""
    print_section("1. Data Loading & ViewPosition Filtering")
    
    # Load data
    df = pd.read_csv(GT_METADATA_PATH)
    print(f"Original data: {len(df)} images")
    
    # ViewPosition distribution
    print(f"\nViewPosition distribution:")
    view_counts = df['ViewPosition'].value_counts()
    for view, count in view_counts.items():
        print(f"  {view}: {count} ({count/len(df)*100:.1f}%)")
    
    # Filter
    df_filtered = df[df['ViewPosition'].isin(ALLOWED_VIEW_POSITIONS)].copy()
    print(f"\nFiltered (only {ALLOWED_VIEW_POSITIONS}): {len(df_filtered)} images")
    print(f"Excluded: {len(df) - len(df_filtered)} images ({(len(df)-len(df_filtered))/len(df)*100:.1f}%)")
    
    return df_filtered


def analyze_pathology_distribution(df, pathology_type='pseudo'):
    """Analyze pathology class distribution
    
    Args:
        df: DataFrame
        pathology_type: 'pseudo' or 'gt'
    """
    prefix = 'gt_' if pathology_type == 'gt' else ''
    
    print_section(f"2. {'GT' if pathology_type == 'gt' else 'CheXpert Pseudo-Label'} Pathology Distribution")
    
    results = []
    for pathology in CHEXPERT_PATHOLOGIES:
        col_name = prefix + pathology
        if col_name not in df.columns:
            print(f"⚠️  {pathology}: Column not found")
            continue
        
        series = df[col_name]
        
        # Count different values
        positive = (series == 1.0).sum()
        negative = (series == 0.0).sum()
        uncertain = (series == -1.0).sum()
        missing = series.isna().sum()
        total = len(series)
        
        results.append({
            'pathology': pathology,
            'positive': positive,
            'negative': negative,
            'uncertain': uncertain,
            'missing': missing,
            'total': total
        })
        
        print(f"\n{pathology}:")
        print(f"  Positive (1.0):   {positive:4d} ({positive/total*100:5.1f}%)")
        print(f"  Negative (0.0):   {negative:4d} ({negative/total*100:5.1f}%)")
        print(f"  Uncertain (-1.0): {uncertain:4d} ({uncertain/total*100:5.1f}%)")
        print(f"  Missing (NaN):    {missing:4d} ({missing/total*100:5.1f}%)")
    
    return pd.DataFrame(results)


def plot_label_value_distribution(df):
    """Plot distribution of label values (1, 0, -1, NaN) for GT and Pseudo labels"""
    print_section("Label Value Distribution Analysis")
    
    # Prepare data for GT and Pseudo labels
    gt_data = []
    pseudo_data = []
    
    for pathology in CHEXPERT_PATHOLOGIES:
        # GT labels
        gt_col = f'gt_{pathology}'
        if gt_col in df.columns:
            series = df[gt_col]
            gt_data.append({
                'pathology': pathology,
                'positive': (series == 1.0).sum(),
                'negative': (series == 0.0).sum(),
                'uncertain': (series == -1.0).sum(),
                'missing': series.isna().sum()
            })
        
        # Pseudo labels
        if pathology in df.columns:
            series = df[pathology]
            pseudo_data.append({
                'pathology': pathology,
                'positive': (series == 1.0).sum(),
                'negative': (series == 0.0).sum(),
                'uncertain': (series == -1.0).sum(),
                'missing': series.isna().sum()
            })
    
    # Create stacked bar charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # GT labels
    if gt_data:
        df_gt = pd.DataFrame(gt_data)
        df_gt = df_gt.set_index('pathology')
        
        df_gt[['positive', 'negative', 'uncertain', 'missing']].plot(
            kind='barh', 
            stacked=True, 
            ax=ax1,
            color=['#2ca02c', '#d62728', '#ff7f0e', '#7f7f7f'],
            alpha=0.8
        )
        ax1.set_title('GT Label Value Distribution (14 Pathologies)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_ylabel('')
        ax1.legend(['Positive (1)', 'Negative (0)', 'Uncertain (-1)', 'Missing (NaN)'], 
                   loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid(axis='x', alpha=0.3)
    
    # Pseudo labels
    if pseudo_data:
        df_pseudo = pd.DataFrame(pseudo_data)
        df_pseudo = df_pseudo.set_index('pathology')
        
        df_pseudo[['positive', 'negative', 'uncertain', 'missing']].plot(
            kind='barh', 
            stacked=True, 
            ax=ax2,
            color=['#2ca02c', '#d62728', '#ff7f0e', '#7f7f7f'],
            alpha=0.8
        )
        ax2.set_title('Pseudo Label Value Distribution (14 Pathologies)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_ylabel('')
        ax2.legend(['Positive (1)', 'Negative (0)', 'Uncertain (-1)', 'Missing (NaN)'], 
                   loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("d:/workspace/MRes/NoiseRate/label_value_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Label value distribution chart saved to: {output_path}\n")
    plt.close()


def compare_gt_pseudo(df):
    """Compare GT labels vs pseudo-labels"""
    print_section("3. GT Labels vs Pseudo-Labels Comparison")
    
    # Check which pathologies have GT labels
    gt_cols = [col for col in df.columns if col.startswith('gt_')]
    available_pathologies = [col.replace('gt_', '') for col in gt_cols]
    
    print(f"Pathologies with GT labels: {len(available_pathologies)}")
    print(f"{', '.join(available_pathologies)}\n")
    
    # Store results for visualization
    comparison_results = []
    
    for pathology in available_pathologies:
        gt_col = f'gt_{pathology}'
        pseudo_col = pathology
        
        if pseudo_col not in df.columns:
            continue
        
        # Compare only samples where both GT and pseudo are non-NaN
        valid_mask = df[gt_col].notna() & df[pseudo_col].notna()
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            continue
        
        # Agreement analysis (treating -1 as 0)
        gt_binary = valid_df[gt_col].replace(-1, 0)
        pseudo_binary = valid_df[pseudo_col].replace(-1, 0)
        
        agreement = (gt_binary == pseudo_binary).sum()
        total_valid = len(valid_df)
        agreement_rate = agreement / total_valid * 100
        
        print(f"{pathology}:")
        print(f"  Valid samples: {total_valid}")
        print(f"  Agreement: {agreement}/{total_valid} ({agreement_rate:.1f}%)")
        
        # Distribution comparison
        gt_pos = (gt_binary == 1).sum()
        pseudo_pos = (pseudo_binary == 1).sum()
        gt_pos_rate = gt_pos / total_valid * 100
        pseudo_pos_rate = pseudo_pos / total_valid * 100
        
        print(f"  GT positive: {gt_pos} ({gt_pos_rate:.1f}%)")
        print(f"  Pseudo positive: {pseudo_pos} ({pseudo_pos_rate:.1f}%)")
        print()
        
        # Store for visualization
        comparison_results.append({
            'pathology': pathology,
            'valid_samples': total_valid,
            'agreement_rate': agreement_rate,
            'gt_pos_rate': gt_pos_rate,
            'pseudo_pos_rate': pseudo_pos_rate
        })
    
    # Generate visualizations
    if comparison_results:
        _plot_comparison_charts(comparison_results)


def _plot_comparison_charts(results):
    """Generate comparison charts"""
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('agreement_rate', ascending=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Agreement Rate
    colors = ['#d62728' if x < 80 else '#2ca02c' if x > 90 else '#ff7f0e' 
              for x in df_results['agreement_rate']]
    ax1.barh(df_results['pathology'], df_results['agreement_rate'], color=colors)
    ax1.set_xlabel('Agreement Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('GT vs Pseudo-Label Agreement Rate', fontsize=14, fontweight='bold')
    ax1.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(df_results['agreement_rate']):
        ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # Plot 2: Positive Rate Comparison
    df_sorted = df_results.sort_values('gt_pos_rate', ascending=True)
    x = np.arange(len(df_sorted))
    width = 0.35
    
    bars1 = ax2.barh(x - width/2, df_sorted['gt_pos_rate'], width, 
                     label='GT', color='#1f77b4', alpha=0.8)
    bars2 = ax2.barh(x + width/2, df_sorted['pseudo_pos_rate'], width, 
                     label='Pseudo', color='#ff7f0e', alpha=0.8)
    
    ax2.set_xlabel('Positive Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('GT vs Pseudo Positive Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(df_sorted['pathology'])
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("d:/workspace/MRes/NoiseRate/gt_pseudo_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison charts saved to: {output_path}")
    plt.close()


def identify_data_quality_issues(df):
    """Identify data quality issues"""
    print_section("4. Data Quality Issue Detection")
    
    issues = []
    
    # Check for all-NaN samples
    pathology_cols = [col for col in CHEXPERT_PATHOLOGIES if col in df.columns]
    all_nan_mask = df[pathology_cols].isna().all(axis=1)
    all_nan_count = all_nan_mask.sum()
    
    if all_nan_count > 0:
        issues.append(f"Found {all_nan_count} samples with all pathology labels as NaN")
        print(f"⚠️  {issues[-1]}")
    
    # Check GT label coverage
    gt_cols = [col for col in df.columns if col.startswith('gt_')]
    if gt_cols:
        for gt_col in gt_cols:
            non_nan = df[gt_col].notna().sum()
            coverage = non_nan / len(df) * 100
            pathology = gt_col.replace('gt_', '')
            print(f"GT label {pathology}: {non_nan}/{len(df)} samples annotated ({coverage:.1f}%)")
    
    # Check image file existence (sample first 10)
    print("\nSample image file existence check (first 10):")
    for idx in range(min(10, len(df))):
        img_path = Path(GT_IMAGE_BASE_PATH) / df.iloc[idx]['image_path']
        exists = "✓" if img_path.exists() else "✗"
        print(f"  {exists} {df.iloc[idx]['image_path']}")
    
    if not issues:
        print("\n✓ No obvious data quality issues found")
    
    return issues


def generate_summary(df, pseudo_stats, gt_stats):
    """Generate summary report"""
    print_section("5. Data Exploration Summary")
    
    print(f"📊 Dataset Overview:")
    print(f"  Total samples: {len(df)}")
    print(f"  ViewPosition: {', '.join(ALLOWED_VIEW_POSITIONS)}")
    print(f"  Pathology classes: {len(CHEXPERT_PATHOLOGIES)}")
    
    print(f"\n📈 Pseudo-label Statistics:")
    print(f"  Avg positive rate: {pseudo_stats['positive'].sum() / (pseudo_stats['total'].sum()):.1%}")
    print(f"  Avg uncertain rate: {pseudo_stats['uncertain'].sum() / (pseudo_stats['total'].sum()):.1%}")
    print(f"  Avg missing rate: {pseudo_stats['missing'].sum() / (pseudo_stats['total'].sum()):.1%}")
    
    if gt_stats is not None and len(gt_stats) > 0:
        print(f"\n📈 GT Label Statistics:")
        print(f"  Classes with GT: {len(gt_stats)}")
        print(f"  Avg annotation coverage: {gt_stats['positive'].sum() + gt_stats['negative'].sum()} / {gt_stats['total'].sum()}")
    
    print(f"\n💡 Recommendations:")
    print(f"  1. Use {len(df)} samples for noise rate validation")
    print(f"  2. For uncertain values (-1.0), try u_zeros or u_ones strategies")
    print(f"  3. Pay attention to classes with low positive rates (may affect stratified sampling)")


def main():
    """Main function"""
    print("\n" + "🔍 GT Dataset Exploratory Analysis".center(80, "="))
    print(f"Data source: {GT_METADATA_PATH}")
    
    # 1. Load and filter
    df = load_and_filter_data()
    
    # 2. Analyze pseudo-label distribution
    pseudo_stats = analyze_pathology_distribution(df, pathology_type='pseudo')
    
    # 3. Analyze GT label distribution (if exists)
    gt_cols = [col for col in df.columns if col.startswith('gt_')]
    if gt_cols:
        gt_stats = analyze_pathology_distribution(df, pathology_type='gt')
    else:
        gt_stats = None
        print("\n⚠️  No GT label columns found")
    
    # 4. Plot label value distribution
    plot_label_value_distribution(df)
    
    # 5. Compare GT and pseudo labels
    if gt_cols:
        compare_gt_pseudo(df)
    
    # 5. Identify data quality issues
    issues = identify_data_quality_issues(df)
    
    # 6. Generate summary
    generate_summary(df, pseudo_stats, gt_stats)
    
    print("\n" + "="*80)
    print("✓ Data exploration complete!".center(80))
    print("="*80 + "\n")
    
    return df, pseudo_stats, gt_stats


if __name__ == "__main__":
    df, pseudo_stats, gt_stats = main()
