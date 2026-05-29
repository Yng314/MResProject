# NoiseRate - Label Noise Analysis for Chest X-rays

胸部X光多标签分类的标签噪声分析工具。

## 📁 文件结构

```
NoiseRate/
├── validate_torchxray.py          # TorchXRayVision完整流程（推理+噪声分析+AUROC）
├── train_validate_kfold.py        # K-fold训练和验证（GT和Pseudo）
├── explore_gt_data.py             # 数据探索脚本
├── cxr_real_experiment/           # 正式 MIMIC-CXR-JPG 实验（baseline / cleaned / entry-cleaned）
│
├── utils/                          # 工具函数
│   ├── noise_analysis.py           # Cleanlab噪声分析函数
│   ├── auroc_metrics.py            # AUROC计算函数
│   ├── label_utils.py              # 标签处理工具
│   ├── xrv_utils.py                # TorchXRayVision工具
│   └── training_utils.py           # 训练工具
│
├── models/                         # 模型定义
│   └── lightweight_cnn.py          # 轻量级CNN (391K参数)
│
├── config/                         # 配置文件
│   └── unified_config.yaml         # 统一配置文件
│
├── tests/                          # 单元测试
│   ├── test_label_utils.py
│   └── test_xrv_utils.py
│
└── results/                        # 输出结果目录
```

`cxr_real_experiment/` 当前包含：

- `cxr_real_noise_validation_smoke.py`
  - 正式 MIMIC-CXR-JPG 数据上的 K-fold / OOF / cleanlab 烟测与全量噪声分析
- `cxr_real_full_train_eval_cleanlab.py`
  - 全训练集 DenseNet 训练、GT test 评测、再对 train 生成 cleanlab 结果
  - 现已支持通过 metadata 文件按 `ViewPosition` 过滤，例如 `AP/PA`
- `run_baseline_full_train_test_cleanlab.sh`
  - baseline 训练脚本（现配置：`epochs=100`, `patience=10`, `restore best`）
- `run_cleaned_full_train_test_cleanlab.sh`
  - sample-level cleaned 重训脚本
- `run_entry_cleaned_full_train_test_cleanlab.sh`
  - entry-level cleaned 重训脚本
- `run_appa_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA baseline -> AP/PA sample-cleaned -> AP/PA entry-cleaned`
- `cxr_real_full_train_eval_cleanlab_xrv12.py`
  - `TorchXRayVision` 12类正式实验脚本
  - 支持 `xrv_densenet121_direct` 和 `xrv_densenet121_linearhead`
- `run_appa_xrv12_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA + XRV direct 12类` 的三阶段流程
- `run_appa_xrv12_linearhead_pipeline_full_train.sh`
  - 单个 Slurm 作业串行跑完 `AP/PA + XRV encoder + our head 12类` 的三阶段流程
- `select_topk_issue_samples.py`
  - 从 sample-level cleanlab 结果中选取 top-k / top-percentage 可疑样本
- `select_topk_issue_entries.py`
  - 从 entry-level cleanlab 结果中选取 top-k / top-percentage 可疑标签位
- `run_appa_densenet_topk_sweep.sh`
  - `AP/PA + DenseNet 14类` 的 sample-level top-k sweep（5% / 10% / 20%）
- `run_appa_xrv12_linearhead_topk_sweep.sh`
  - `AP/PA + XRV encoder + our head 12类` 的 sample-level top-k sweep（5% / 10% / 20%）
- `run_appa_densenet_entry_topk_sweep.sh`
  - `AP/PA + DenseNet 14类` 的 entry-level top-k sweep（5% / 10% / 20%）
- `run_appa_xrv12_linearhead_entry_topk_sweep.sh`
  - `AP/PA + XRV encoder + our head 12类` 的 entry-level top-k sweep（5% / 10% / 20%）

## 🚀 使用方法

### 1. TorchXRayVision噪声分析

使用预训练的DenseNet121在GT数据上推理并分析噪声：

```bash
conda activate d:\workspace\MRes\.conda
python NoiseRate/validate_torchxray.py --config NoiseRate/config/unified_config.yaml
```

**输出**：
- `results/perclass_noise_rates_*.csv` - Per-class GT和Pseudo噪声率对比
  - 列: Pathology, GT_Positive, GT_Negative, GT_Issues, GT_Noise_Rate, Pseudo_Issues, Pseudo_Noise_Rate, Diff
- `results/noise_analysis_*.npz` - 详细结果（predictions和per-class issues）
- `logs/noise_validation_*.log` - 运行日志

---

### 2. K-Fold自定义模型训练和验证

训练轻量级CNN (391K参数) 在GT和Pseudo标签上：

```bash
conda activate d:\workspace\MRes\.conda
python NoiseRate/train_validate_kfold.py --config NoiseRate/config/unified_config.yaml
```

**流程**：
1. 使用GT标签训练5-fold模型
2. 生成GT的OOF predictions
3. 计算GT模型的AUROC和噪声率
4. 使用Pseudo标签训练5-fold模型
5. 生成Pseudo的OOF predictions
6. 计算Pseudo模型的AUROC和噪声率

**输出**：
- `kfold_results/oof_predictions.npy` - GT OOF predictions
- `kfold_results/cnn_gt_results_*.csv` - GT模型结果（AUROC + GT噪声率）
  - 列: Pathology, AUROC, GT_Noise_Rate
- `kfold_results/models/fold_*_gt_model.pth` - GT模型权重
- `kfold_results_pseudo/oof_predictions_pseudo.npy` - Pseudo OOF predictions
- `kfold_results_pseudo/cnn_pseudo_results_*.csv` - Pseudo模型结果（AUROC + Pseudo噪声率）
  - 列: Pathology, AUROC, Pseudo_Noise_Rate
- `kfold_results_pseudo/models/fold_*_pseudo_model.pth` - Pseudo模型权重
- 各自的log文件包含详细训练和验证信息

---

### 3. 正式 MIMIC-CXR-JPG + GT Test 实验

这套实验使用：

- 训练图像：`MedSoul/datasets/mimic-cxr-jpg-224/`
- 训练标签：`mimic-cxr-2.0.0-chexpert.csv`
- split：`mimic-cxr-2.0.0-split.csv.gz`
- metadata：`mimic-cxr-2.0.0-metadata.csv.gz`
- GT test：`mimic-cxr-2.1.0-test-set-labeled.csv`

#### baseline（推荐起点）

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_baseline_full_train_test_cleanlab.sh
```

当前 baseline 配置：

- `DenseNet121 pretrained`
- `epochs=100`
- `val_fraction=0.1`
- `early_stopping_patience=10`
- `recover_best_weights=true`

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_baseline_es/slurm_<jobid>/`

主要输出：

- `baseline_run_summary.csv`
- `test_image_auroc_summary.csv`
- `test_study_auroc_summary.csv`
- `train_cleanlab_sample_details.csv`
- `train_cleanlab_sample_issues_only.csv`
- `train_cleanlab_entry_issues_only.csv`

#### sample-level cleaned

删除 `train_cleanlab_sample_issues_only.csv` 中的 issue samples 后重训：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_cleaned_full_train_test_cleanlab.sh
```

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_cleaned_es/slurm_<jobid>/`

#### entry-level cleaned

不删样本，只把 `train_cleanlab_entry_issues_only.csv` 中的 issue label entries 从监督里 mask 掉：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_entry_cleaned_full_train_test_cleanlab.sh
```

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_entry_cleaned_es/slurm_<jobid>/`

#### AP/PA 单作业串行版本

如果想只用 `AP/PA` 图像，并且在同一个作业里顺序跑完三阶段：

- `baseline`
- `sample-level cleaned`
- `entry-level cleaned`

可以直接提交：

```bash
cd MRes
sbatch NoiseRate/cxr_real_experiment/run_appa_pipeline_full_train.sh
```

这条脚本会自动：

1. 读取 `mimic-cxr-2.0.0-metadata.csv.gz`
2. 只保留 `ViewPosition in {AP, PA}`
3. 跑第一阶段 baseline
4. 用 baseline 产出的 `train_cleanlab_sample_issues_only.csv` 跑第二阶段 sample-level cleaned
5. 用 baseline 产出的 `train_cleanlab_entry_issues_only.csv` 跑第三阶段 entry-level cleaned
6. 最后在结果根目录输出一份 `appa_pipeline_compare.csv`

输出目录示例：

- `NoiseRate/cxr_real_experiment/results_appa_pipeline/slurm_<jobid>/`
- 其中包含：
  - `01_baseline/`
  - `02_sample_cleaned/`
  - `03_entry_cleaned/`
  - `appa_pipeline_compare.csv`

---

## 📊 主要功能和噪声分析逻辑

### **重要**: 噪声分析逻辑说明

不同脚本的噪声分析目的不同：

#### 1. **validate_torchxray.py** - 预训练模型标签质量对比
- 使用TorchXRayVision预训练模型的predictions
- **同时分析GT noise和Pseudo noise**
- **目的**: 对比GT和Pseudo标签质量（公平比较，因为模型未用它们训练）

#### 2. **train_validate_kfold.py** - 自训练模型性能评估

**CNN-GT模型**:
- 使用GT标签训练
- **只分析GT noise**
- **目的**: 评估模型在自己训练标签上的噪声检测能力

**CNN-Pseudo模型**:
- 使用Pseudo标签训练
- **只分析Pseudo noise**
- **目的**: 评估模型在自己训练标签上的噪声检测能力

---

## 🔧 工具模块


- `run_cleanlab_multilabel_analysis()` - 多标签cleanlab噪声检测
- `analyze_per_class_noise()` - 每个类别的噪声率分析
- `compare_noise_results()` - GT vs Pseudo对比

### AUROC Metrics (`utils/auroc_metrics.py`)

- `calculate_per_class_auroc()` - 计算per-class AUROC
- `log_auroc_results()` - 格式化输出AUROC结果

### Label Utilities (`utils/label_utils.py`)

- `filter_by_view_position()` - 过滤影像视角
- `handle_uncertain_labels()` - 处理不确定标签
- `convert_to_multilabel_format()` - 转换为cleanlab格式

---

## 🧠 正式实验标签口径

当前 `cxr_real_experiment/` 这一套正式实验，训练和 cleanlab 统一使用二值口径：

- `1 -> 1`
- `-1 -> 1`
- `0 -> 0`
- `NaN/null -> mask`

也就是：

- `positive + uncertain -> positive`
- `negative -> negative`
- `missing -> ignore`

GT test 也使用同样的二值化口径。

额外说明：

- GT test 文件里使用 `Airspace Opacity`
- 代码里会映射到 `Lung Opacity`
- 当前正式实验支持两种口径：
  - 全视角：直接使用能和 `split.csv` 对上的全部图像
  - `AP/PA`：额外读取 `mimic-cxr-2.0.0-metadata.csv.gz` 并按 `ViewPosition` 过滤
- `AP/PA` 过滤已在正式流程中接通，训练与 GT test 两侧都会一致应用

---

## ⚙️ 配置

### 统一配置文件 (`config/unified_config.yaml`)

不再使用分散的配置文件，而是使用统一的 `config/unified_config.yaml` 管理所有设定。

主要包含三个部分：
1. **`common`**: 公用设置（数据路径、预处理等）
2. **`torchxray`**: `validate_torchxray.py` 专用设置
3. **`kfold`**: `train_validate_kfold.py` 专用设置（包含 `common`, `gt`, `pseudo` 子部分）

```yaml
common:
  data:
    gt_metadata_path: "..."
    gt_image_base_path: "..."
    allowed_view_positions: ['AP', 'PA']

torchxray:
  model:
    name: "densenet121-res224-all"
    
kfold:
  common:
    kfold_settings:
      n_splits: 5
    model:
      name: "LightweightCNN"
  gt:
    data:
      use_pseudo_labels: false
  pseudo:
    data:
      use_pseudo_labels: true
```

---

## 🧪 运行测试

```bash
cd NoiseRate/tests
python -m pytest test_label_utils.py
python -m pytest test_xrv_utils.py
```

---

## 📈 实验结果说明

### 噪声率对比方法

**标签质量对比** (validate_torchxray.py):
- 使用预训练模型predictions
- 同时计算GT noise和Pseudo noise
- **目的**: 评估两种标签本身的质量差异

**模型性能评估** (train_validate_kfold.py):
- CNN-GT: 用GT训练 → 只计算GT noise
- CNN-Pseudo: 用Pseudo训练 → 只计算Pseudo noise
- **目的**: 评估模型在各自训练标签上的噪声检测性能

### 典型结果

#### TorchXRayVision (预训练模型)
- **平均AUROC**: ~0.65
- **GT噪声率**: ~8% (per-class)
- **Pseudo噪声率**: ~9% (per-class)
- **结论**: Pseudo标签噪声略高于GT

#### LightweightCNN (从头训练)
- **CNN-GT平均AUROC**: ~0.59
- **CNN-GT的GT噪声率**: ~19% (per-class)
- **CNN-Pseudo平均AUROC**: ~0.60
- **CNN-Pseudo的Pseudo噪声率**: ~19% (per-class)
- **结论**: 自训练模型检测到更多噪声，无数据泄露

### 正式 MIMIC-CXR-JPG + GT Test（2026-04 实验）

GT test 使用：

- 原始 labeled CSV：`687 studies`
- 实际可评测：`685 studies / 1029 images`

#### 全视角 baseline early-stopping

目录：

- `cxr_real_experiment/results_baseline_es/slurm_234377/`

结果：

- `best_epoch = 11`
- `best_val_loss = 0.2901`
- `test_image_macro_auroc_binary = 0.7812`
- `test_study_macro_auroc_binary = 0.7962`
- `train_cleanlab_sample_issue_rate = 0.1506`
- `train_cleanlab_entry_issue_rate = 0.0124`

#### 全视角 sample-level cleaned early-stopping

目录：

- `cxr_real_experiment/results_cleaned_es/slurm_234888/`

结果：

- 删除 `55,491` 个 issue samples
- `best_epoch = 9`
- `best_val_loss = 0.1536`
- `test_image_macro_auroc_binary = 0.7826`
- `test_study_macro_auroc_binary = 0.7891`

结论：

- image-level 略高于 baseline
- study-level 低于 baseline

#### 全视角 entry-level cleaned early-stopping

目录：

- `cxr_real_experiment/results_entry_cleaned_es/slurm_235417/`

结果：

- 保留全部 `368,562` 个样本
- mask 掉 `63,811` 个 issue label entries
- `best_epoch = 7`
- `best_val_loss = 0.1441`
- `test_image_macro_auroc_binary = 0.7799`
- `test_study_macro_auroc_binary = 0.7791`

结论：

- test 表现低于 baseline
- 当前这版 entry-level 屏蔽没有带来收益

#### 全视角当前结论

在目前这套设置下：

1. `baseline_es` 是 study-level 指标最好的方案
2. `sample-level cleaned` 略微提升 image-level，但降低 study-level
3. `entry-level cleaned` 进一步降低了 study-level

因此下一步更值得尝试的是：

- 不做全量清洗
- 改成只清洗 top-k / top-percentage 最可疑样本
- 或者重新评估 `-1 (uncertain)` 的处理策略

#### AP/PA 版本状态

`AP/PA` 版本已经通过正式小烟测，过滤规模如下：

- train split：`368,960 -> 237,972` 张图像
- test split 候选图像：`5,176 -> 3,414`
- 与 GT merge 后可评测：`676 images / 605 studies`

目前 `AP/PA` 的单作业串行脚本已经提供，结果会写到：

- `cxr_real_experiment/results_appa_pipeline/slurm_<jobid>/`

建议读取：

- `01_baseline/baseline_run_summary.csv`
- `02_sample_cleaned/baseline_run_summary.csv`
- `03_entry_cleaned/baseline_run_summary.csv`
- `appa_pipeline_compare.csv`

#### AP/PA + 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_pipeline/slurm_236304/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.7701 | 0.7637 |
| full sample-cleaned | 0.7797 | 0.7767 |
| full entry-cleaned | 0.7528 | 0.7489 |

结论：

- `sample-level cleaned` 有提升
- `entry-level cleaned` 明显不如 baseline

#### AP/PA + XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_pipeline/slurm_236337/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.8158 | 0.8095 |
| full sample-cleaned | 0.8127 | 0.8043 |
| full entry-cleaned | 0.8114 | 0.8028 |

结论：

- baseline 是这条 12类线里最好的全量方案
- 全量 sample / entry cleaning 都略微降低了 test 指标

#### AP/PA + XRV direct decoder（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_pipeline/slurm_236482/`

三阶段结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| baseline | 0.6658 | 0.6638 |
| full sample-cleaned | 0.6370 | 0.6365 |
| full entry-cleaned | 0.6340 | 0.6341 |

结论：

- `XRV direct decoder` 明显弱于 `XRV encoder + our head`
- 因此后续实验默认优先保留 `linear-head` 路线

#### AP/PA + sample-level top-k cleaning

说明：

- 这里的 top-k 指从 baseline 的 `train_cleanlab_sample_issues_only.csv`
- 只选最可疑的前 `5% / 10% / 20%` issue samples 删除后重训

##### 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_densenet_topk/slurm_236927/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 sample | 0.7677 | 0.7574 |
| top10 sample | 0.8226 | 0.8123 |
| top20 sample | 0.8174 | 0.8117 |

结论：

- `top10 sample` 是当前这条线的最佳配置
- 它优于 baseline、full sample-cleaned 和 full entry-cleaned

##### XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_topk/slurm_236928/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 sample | 0.8143 | 0.8073 |
| top10 sample | 0.8384 | 0.8272 |
| top20 sample | 0.8386 | 0.8342 |

结论：

- `top20 sample` 略优于 `top10 sample`
- 两者都明显优于 baseline 和全量 cleaned

#### AP/PA + entry-level top-k cleaning

说明：

- 这里的 top-k 指从 baseline 的 `train_cleanlab_entry_issues_only.csv`
- 只选最可疑的前 `5% / 10% / 20%` issue label entries 做 mask

##### 自己 DenseNet（14类）

目录：

- `cxr_real_experiment/results_appa_densenet_entry_topk/slurm_237240/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 entry | 0.8162 | 0.8134 |
| top10 entry | 0.8141 | 0.8059 |
| top20 entry | 0.7952 | 0.7892 |

结论：

- `top05 entry` 最好
- `entry-level top-k` 明显优于全量 entry-clean
- 但仍略逊于 `top10 sample`

##### XRV encoder + our head（12类）

目录：

- `cxr_real_experiment/results_appa_xrv12_linearhead_entry_topk/slurm_237241/`

结果：

| 方案 | Study Macro AUROC | Image Macro AUROC |
|---|---:|---:|
| top05 entry | 0.8243 | 0.8213 |
| top10 entry | 0.7848 | 0.7753 |
| top20 entry | 0.7879 | 0.7820 |

结论：

- `top05 entry` 略高于 baseline
- 但明显不如 `top10/top20 sample`

#### 当前阶段总结合

当前最值得保留的两条配置是：

1. `AP/PA + 自己 DenseNet + 14类 + top10 sample cleaning`
2. `AP/PA + XRV encoder + our head + 12类 + top20 sample cleaning`

整体结论：

- `cleanlab-based cleaning` 是有效的
- 但不是“全量 cleaning 越多越好”
- 当前结果最支持的是：`温和的 sample-level top-k cleaning`
- `entry-level top-k` 也比全量 entry-clean 更稳，但整体还是弱于最优的 sample-level top-k

---

## 📝 依赖

- Python 3.8+
- PyTorch
- torchxrayvision
- cleanlab
- scikit-learn
- pandas
- numpy
- PyYAML

---

## 🔧 开发说明

- 所有脚本使用英文注释和变量名
- 配置和文档使用中文
- 代码风格遵循PEP 8
- 新功能需要添加单元测试
