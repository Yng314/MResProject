"""
修复数据泄露问题：将所有测试标签文件中的 study_id 在 split CSV 中设置为 'test'
检查的文件：
- datasets/medpalm_test_labels.csv
- datasets/final_test_labels.csv
- datasets/medpalm_and_mimic_test_labels.csv
"""
import pandas as pd
from pathlib import Path
import shutil

# 文件路径
test_label_files = [
    'datasets/medpalm_test_labels.csv',
    'datasets/final_test_labels.csv',
    'datasets/medpalm_and_mimic_test_labels.csv'
]
split_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz'
backup_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz.backup'

print("="*70)
print("修复 Split CSV 数据泄露问题（所有测试标签文件）")
print("="*70)

# 1. 收集所有测试标签文件中的 study_id
print(f"\n[1] 读取所有测试标签文件...")
all_test_study_ids = set()

for csv_file in test_label_files:
    if Path(csv_file).exists():
        print(f"  读取 {csv_file}...")
        df = pd.read_csv(csv_file)
        study_ids = set(df['study_id'].astype(int).unique())
        all_test_study_ids.update(study_ids)
        print(f"    找到 {len(study_ids)} 个 study_id")
    else:
        print(f"  跳过（文件不存在）: {csv_file}")

print(f"\n总共找到 {len(all_test_study_ids)} 个唯一的测试 study_id")

# 2. 读取 split CSV
print(f"\n[2] 读取 {split_csv}...")
split_df = pd.read_csv(split_csv)
print(f"Split CSV 总记录数: {len(split_df)}")
print(f"原始 Split 分布:")
print(split_df['split'].value_counts())

# 3. 检查哪些 study_id 需要修改
print(f"\n[3] 检查需要修改的记录...")
split_df['study_id'] = split_df['study_id'].astype(int)
needs_fix = split_df['study_id'].isin(all_test_study_ids)
affected_rows = split_df[needs_fix].copy()

print(f"找到 {len(affected_rows)} 条记录需要修改")
if len(affected_rows) > 0:
    print(f"\n当前这些记录的 split 分布:")
    print(affected_rows['split'].value_counts())
    
    # 检查有多少条记录不是 'test'
    not_test = affected_rows[affected_rows['split'] != 'test']
    print(f"\n需要修改的记录数（当前不是 'test'）: {len(not_test)}")
    
    if len(not_test) > 0:
        print(f"\n这些记录的 split 分布:")
        print(not_test['split'].value_counts())
        
        # 显示一些示例
        print(f"\n示例需要修改的记录（前5条）:")
        print(not_test[['subject_id', 'study_id', 'dicom_id', 'split']].head())
        
        # 4. 创建备份
        print(f"\n[4] 创建备份文件: {backup_csv}...")
        if Path(backup_csv).exists():
            print("  备份文件已存在，跳过备份")
        else:
            shutil.copy(split_csv, backup_csv)
            print("  备份完成")
        
        # 5. 修改 split 值
        print(f"\n[5] 修改 split 值为 'test'...")
        split_df.loc[needs_fix, 'split'] = 'test'
        
        # 6. 验证修改
        print(f"\n[6] 验证修改结果...")
        modified_rows = split_df[needs_fix]
        print(f"修改后的 split 分布:")
        print(modified_rows['split'].value_counts())
        
        # 7. 保存文件
        print(f"\n[7] 保存修改后的文件...")
        split_df.to_csv(split_csv, index=False, compression='gzip')
        print(f"已保存到 {split_csv}")
        
        print(f"\n[完成] 成功修复 {len(not_test)} 条记录的数据泄露问题")
        print(f"备份文件: {backup_csv}")
    else:
        print("\n[信息] 所有相关记录已经是 'test'，无需修改")
        print(f"\n最终 Split 分布:")
        print(split_df['split'].value_counts())
else:
    print("\n[警告] 在 split CSV 中没有找到任何测试标签文件中的 study_id")

print("\n" + "="*70)
print("操作完成")
print("="*70)

