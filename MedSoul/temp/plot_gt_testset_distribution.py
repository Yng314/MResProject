import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 读取CSV文件
df = pd.read_csv('datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv')

# 定义标签名称（14个）
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

# 统计每个标签的类别分布
label_stats = defaultdict(lambda: {'1': 0, '0': 0, '-1': 0, 'null': 0})

for label_name in label_names:
    for value in df[label_name]:
        if pd.isna(value):
            label_stats[label_name]['null'] += 1
        elif value == 1.0 or value == 1:
            label_stats[label_name]['1'] += 1
        elif value == 0.0 or value == 0:
            label_stats[label_name]['0'] += 1
        elif value == -1.0 or value == -1:
            label_stats[label_name]['-1'] += 1

# 准备绘图数据
classes = ['1', '0', '-1', 'null']
x = np.arange(len(label_names))
width = 0.2  # 每个柱子的宽度

# 创建图形
fig, ax = plt.subplots(figsize=(16, 8))

# 为每个类别绘制柱状图
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
for i, cls in enumerate(classes):
    counts = [label_stats[label][cls] for label in label_names]
    offset = width * (i - 1.5)
    ax.bar(x + offset, counts, width, label=f'Class {cls}', color=colors[i])

# 设置x轴标签
ax.set_xlabel('Labels', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Ground Truth Test Set Labels (14 Labels × 4 Classes)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(label_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('temp/gt_testset_label_distribution.png', dpi=300, bbox_inches='tight')
print("图表已保存到: temp/gt_testset_label_distribution.png")

# 打印统计信息
print("\nGround Truth测试集标签分布统计:")
print("-" * 80)
print(f"{'Label':<30} {'Class 1':<12} {'Class 0':<12} {'Class -1':<12} {'null':<12}")
print("-" * 80)
total_samples = len(df)
for label in label_names:
    stats = label_stats[label]
    print(f"{label:<30} {stats['1']:<12} {stats['0']:<12} {stats['-1']:<12} {stats['null']:<12}")
print("-" * 80)
print(f"总样本数: {total_samples}")

plt.show()




