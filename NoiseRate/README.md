# NoiseRate - Label Noise Analysis for Chest X-rays

胸部X光多标签分类的标签噪声分析工具。

## 📁 文件结构

```
NoiseRate/
├── validate_torchxray.py          # TorchXRayVision完整流程（推理+噪声分析+AUROC）
├── train_validate_kfold.py        # K-fold训练和验证（GT和Pseudo）
├── explore_gt_data.py             # 数据探索脚本
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
