"""验证 split CSV 修复结果"""
import pandas as pd

split_csv = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz'

print("验证 Split CSV 修复结果")
print("="*70)

df = pd.read_csv(split_csv)
print(f"\n总记录数: {len(df)}")
print(f"\nSplit 分布:")
print(df['split'].value_counts())

# 检查测试标签文件中的 study_id
test_files = [
    'datasets/medpalm_test_labels.csv',
    'datasets/final_test_labels.csv',
    'datasets/medpalm_and_mimic_test_labels.csv'
]

all_test_ids = set()
for f in test_files:
    try:
        test_df = pd.read_csv(f)
        all_test_ids.update(test_df['study_id'].astype(int).unique())
    except:
        pass

print(f"\n测试标签文件中的 study_id 总数: {len(all_test_ids)}")

# 检查这些 study_id 在 split CSV 中的分布
df['study_id'] = df['study_id'].astype(int)
test_rows = df[df['study_id'].isin(all_test_ids)]
print(f"\n这些 study_id 在 split CSV 中的分布:")
print(test_rows['split'].value_counts())

if len(test_rows[test_rows['split'] != 'test']) == 0:
    print("\n[成功] 所有测试标签文件中的 study_id 都已被正确标记为 'test'")
else:
    print(f"\n[警告] 仍有 {len(test_rows[test_rows['split'] != 'test'])} 条记录不是 'test'")

