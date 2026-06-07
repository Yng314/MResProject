"""
修复数据泄露问题：将 medpalm_test_labels.csv 中的 study_id 在 split CSV 中设置为 'test'
"""
import pandas as pd
from pathlib import Path

# 文件路径
medpalm_csv = 'datasets/medpalm_test_labels.csv'
split_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz'
backup_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz.backup'

print("="*70)
print("修复 Split CSV 数据泄露问题")
print("="*70)

# 1. 读取 medpalm_test_labels.csv 获取所有 study_id
print(f"\n[1] 读取 {medpalm_csv}...")
medpalm_df = pd.read_csv(medpalm_csv)
medpalm_study_ids = set(medpalm_df['study_id'].astype(int).unique())
print(f"找到 {len(medpalm_study_ids)} 个唯一的 study_id")

# 2. 读取 split CSV
print(f"\n[2] 读取 {split_csv}...")
split_df = pd.read_csv(split_csv)
print(f"Split CSV 总记录数: {len(split_df)}")
print(f"Split 分布:")
print(split_df['split'].value_counts())

# 3. 检查哪些 study_id 需要修改
print(f"\n[3] 检查需要修改的记录...")
split_df['study_id'] = split_df['study_id'].astype(int)
needs_fix = split_df['study_id'].isin(medpalm_study_ids)
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
        
        # 4. 创建备份
        print(f"\n[4] 创建备份文件: {backup_csv}...")
        import shutil
        shutil.copy(split_csv, backup_csv)
        print("备份完成")
        
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
else:
    print("\n[警告] 在 split CSV 中没有找到 medpalm_test_labels.csv 中的任何 study_id")
    print("这可能意味着这些 study_id 不在 split CSV 中，或者文件名/格式有问题")

print("\n" + "="*70)
print("操作完成")
print("="*70)

