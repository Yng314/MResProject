import pandas as pd

# 读取三个CSV文件
file1_path = 'datasets/medpalm_test_labels.csv'
file2_path = 'datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv'
exclude_path = 'datasets/final_test_labels.csv'
output_path = 'datasets/medpalm_and_mimic_test_labels.csv'

print("正在读取文件...")
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df_exclude = pd.read_csv(exclude_path)

print(f"文件1记录数: {len(df1)}")
print(f"文件2记录数: {len(df2)}")
print(f"排除文件记录数: {len(df_exclude)}")

# 获取要排除的study_id集合
exclude_study_ids = set(df_exclude['study_id'].astype(int))

# 合并两个文件
df_union = pd.concat([df1, df2], ignore_index=True)
print(f"合并后记录数: {len(df_union)}")

# 移除在exclude文件中的记录
df_result = df_union[~df_union['study_id'].astype(int).isin(exclude_study_ids)]
print(f"排除后记录数: {len(df_result)}")

# 去重（基于study_id），保留第一个出现的记录
df_result = df_result.drop_duplicates(subset=['study_id'], keep='first')
print(f"去重后记录数: {len(df_result)}")

# 确保study_id为整数格式
df_result['study_id'] = df_result['study_id'].astype(int)

# 按study_id排序
df_result = df_result.sort_values('study_id').reset_index(drop=True)

# 保存到新文件
df_result.to_csv(output_path, index=False)
print(f"\n已保存 {len(df_result)} 条记录到 {output_path}")
print(f"\n前5条记录:")
print(df_result.head())
print(f"\n后5条记录:")
print(df_result.tail())

# 验证：确保结果中没有exclude文件中的study_id
result_study_ids = set(df_result['study_id'])
overlap = result_study_ids & exclude_study_ids
if overlap:
    print(f"\n警告：发现 {len(overlap)} 个重复的study_id仍在结果中")
else:
    print(f"\n验证通过：结果中不包含exclude文件中的任何study_id")

