import pandas as pd
import os

# 读取两个CSV文件
file1_path = 'datasets/medpalm_test_labels.csv'
file2_path = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv'
output_path = 'datasets/final_test_labels.csv'

print("正在读取文件...")
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

print(f"文件1记录数: {len(df1)}")
print(f"文件2记录数: {len(df2)}")

# 找出重复的study_id
common_study_ids = set(df1['study_id']) & set(df2['study_id'])
print(f"重复的study_id数量: {len(common_study_ids)}")

# 提取重复的记录
# 优先使用file2的数据（mimic-cxr-2.1.0-test-set-labeled.csv），如果file2没有则使用file1
duplicate_records = []
for study_id in sorted(common_study_ids):
    # 优先从file2取，如果file2没有则从file1取
    record = df2[df2['study_id'] == study_id]
    if len(record) > 0:
        duplicate_records.append(record.iloc[0])
    else:
        record = df1[df1['study_id'] == study_id]
        if len(record) > 0:
            duplicate_records.append(record.iloc[0])

# 创建DataFrame并保存
if duplicate_records:
    final_df = pd.DataFrame(duplicate_records)
    # 确保列顺序正确
    final_df = final_df[df1.columns]
    # 确保study_id为整数格式
    final_df['study_id'] = final_df['study_id'].astype(int)
    final_df.to_csv(output_path, index=False)
    print(f"已保存 {len(final_df)} 条重复记录到 {output_path}")
    print(f"\n前5条重复记录:")
    print(final_df.head())
else:
    print("没有找到重复的记录")

