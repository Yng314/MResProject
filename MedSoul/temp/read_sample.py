import pandas as pd

df = pd.read_parquet('datasets/mimic_dataset/train-00000-of-00002.parquet')
sample = df.iloc[410]

print('=' * 80)
print('Sample at index 410')
print('=' * 80)
print(f'\nImage size: {len(sample["image"])} bytes')
print(f'\nFindings:')
print(sample["findings"])
print(f'\nImpression:')
print(sample["impression"])
print('=' * 80)

