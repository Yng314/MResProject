"""检查 train 集合中是否有测试标签文件中的 study_id"""
import pandas as pd

# 测试标签文件
test_files = [
    'datasets/medpalm_test_labels.csv',
    'datasets/final_test_labels.csv',
    'datasets/medpalm_and_mimic_test_labels.csv'
]

split_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz'

print("="*70)
print("检查 Train 集合中的数据泄露问题")
print("="*70)

# 1. 收集所有测试标签文件中的 study_id
print(f"\n[1] 读取所有测试标签文件...")
all_test_study_ids = set()

for csv_file in test_files:
    if pd.io.common.file_exists(csv_file):
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
split_df['study_id'] = split_df['study_id'].astype(int)

# 3. 检查 train 集合中是否有测试标签文件中的 study_id
print(f"\n[3] 检查 train 集合...")
train_df = split_df[split_df['split'] == 'train']
train_test_ids = train_df[train_df['study_id'].isin(all_test_study_ids)]

if len(train_test_ids) > 0:
    print(f"\n[警告] 发现 {len(train_test_ids)} 条记录在 train 集合中，但属于测试标签文件！")
    print(f"这些记录的 study_id:")
    unique_study_ids = train_test_ids['study_id'].unique()
    print(f"  唯一 study_id 数量: {len(unique_study_ids)}")
    print(f"\n前10个 study_id:")
    for sid in unique_study_ids[:10]:
        print(f"  {sid}")
    
    if len(unique_study_ids) > 10:
        print(f"  ... 还有 {len(unique_study_ids) - 10} 个")
    
    print(f"\n这些记录的详细信息（前10条）:")
    print(train_test_ids[['subject_id', 'study_id', 'dicom_id', 'split']].head(10))
    
    print(f"\n[需要修复] 这些记录应该被标记为 'test' 而不是 'train'")
else:
    print(f"\n[成功] Train 集合中没有发现测试标签文件中的 study_id")

# 4. 也检查 validate 集合
print(f"\n[4] 检查 validate 集合...")
validate_df = split_df[split_df['split'] == 'validate']
validate_test_ids = validate_df[validate_df['study_id'].isin(all_test_study_ids)]

if len(validate_test_ids) > 0:
    print(f"\n[警告] 发现 {len(validate_test_ids)} 条记录在 validate 集合中，但属于测试标签文件！")
    print(f"这些记录的 study_id:")
    unique_study_ids = validate_test_ids['study_id'].unique()
    print(f"  唯一 study_id 数量: {len(unique_study_ids)}")
    print(f"\n前10个 study_id:")
    for sid in unique_study_ids[:10]:
        print(f"  {sid}")
    
    if len(unique_study_ids) > 10:
        print(f"  ... 还有 {len(unique_study_ids) - 10} 个")
else:
    print(f"\n[成功] Validate 集合中没有发现测试标签文件中的 study_id")

# 5. 总结
print(f"\n" + "="*70)
print("总结")
print("="*70)
print(f"测试标签文件中的 study_id 总数: {len(all_test_study_ids)}")
print(f"Train 集合中的泄露记录数: {len(train_test_ids)}")
print(f"Validate 集合中的泄露记录数: {len(validate_test_ids)}")

if len(train_test_ids) == 0 and len(validate_test_ids) == 0:
    print("\n[完美] 没有发现数据泄露问题！")
else:
    print(f"\n[需要修复] 总共需要修复 {len(train_test_ids) + len(validate_test_ids)} 条记录")

