# MedSoul Experiment Guide

## 实验管理系统

每次实验都会创建独立的文件夹，避免覆盖之前的结果。

## 快速开始

### 运行Pipeline

```bash
python main.py
```

系统会询问：

**1. 实验模式选择：**
- **创建新实验** - 从头开始
- **继续已有实验** - 从已有的实验文件夹继续运行

**2. 如果创建新实验：**
- **实验名称**：例如 `baseline`, `with_cl`, `exp_001`
- **是否重用已有伪标签**：如果已经生成过，可以选择重用（节省时间和API费用）

**3. 如果继续实验：**
- 系统会列出所有已有实验
- 选择要继续的实验序号
- 自动加载该实验的配置

## 实验文件夹结构

```
outputs/
├── your_experiment_name/
│   ├── config.yaml                         # 初始配置
│   ├── pseudo_labels.json                  # 原始伪标签
│   │
│   ├── mae_pretrain/                       # 初始MAE训练
│   │   ├── encoder_best.pth
│   │   └── logs/
│   │
│   ├── wsl_train/                          # 初始WSL训练
│   │   ├── linear_probe_best.pth
│   │   ├── fine_tune_best.pth
│   │   └── logs_*/
│   │
│   ├── confident_learning/                 # Confident Learning
│   │   ├── pseudo_labels_cleaned.json      # 清洗后的标签
│   │   └── label_quality.npz
│   │
│   ├── mae_pretrain_iter1/                 # 第1次迭代MAE (可选)
│   ├── wsl_train_iter1/                    # 第1次迭代WSL (可选)
│   ├── config_iter1.yaml                   # 第1次迭代配置 (可选)
│   │
│   └── evaluation_results.json             # 评估结果
```

## 主要功能

### 1. 创建新实验

适合：第一次运行或开始新的实验

```bash
python main.py
# 选择: 1 (Create new experiment)
# 输入实验名称: baseline_001
# 选择标签: 1 (Reuse) 或 2 (Regenerate)
```

### 2. 继续已有实验

适合：实验中断后继续，或在已有实验基础上继续训练

```bash
python main.py
# 选择: 2 (Continue existing experiment)
# 选择实验序号: 1
```

### 3. Confident Learning自动迭代 ✨

**当CL enabled时，系统会自动询问是否用清洗后的标签重新训练：**

1. Phase 1-3 完成（初始训练）
2. Phase 4 完成（Confident Learning清洗标签）
3. 系统提示：**"Retrain with cleaned labels? (y/n)"**
4. 如果选择 `y`：
   - 自动创建 `config_iter1.yaml`
   - 自动用清洗后的标签运行 Phase 2+3
   - **跳过 Phase 4**（不再运行CL）
   - 结果保存在 `mae_pretrain_iter1/` 和 `wsl_train_iter1/`
5. 可以继续多次迭代（最多5次）

**示例流程：**
```
初始训练 (Phase 1-4)
  ↓
Confident Learning 清洗标签
  ↓
询问："Retrain with cleaned labels?"
  ↓ (yes)
迭代1 (Phase 2-3，使用cleaned labels)
  ↓
完成
```

## 实验场景

### 场景1：Baseline实验（不使用CL）

```yaml
# configs/config.yaml
confident_learning:
  enabled: false
```

```bash
python main.py
# 创建新实验: baseline_no_cl
# 重用标签: 1
```

### 场景2：使用Confident Learning + 自动迭代

```yaml
# configs/config.yaml
confident_learning:
  enabled: true
```

```bash
python main.py
# 创建新实验: with_cl_auto_iter
# 重用标签: 1
# 等待Phase 1-4完成...
# 提示: "Retrain with cleaned labels? (y/n)"
# 输入: y
# 自动运行迭代训练...
```

### 场景3：中断后继续

```bash
# 假设实验 "exp_001" 在Phase 2中断
python main.py
# 选择: 2 (Continue existing experiment)
# 选择实验: exp_001
# 系统会从上次中断的地方继续
```

### 场景4：对比多次迭代

训练完成后，会自动询问评估哪个模型：
- Initial Model (原始标签)
- Iteration 1 Model (清洗后标签)
- Iteration 2 Model (如果有)
- Evaluate all (评估所有)

## 数据保护 ✅

### Confident Learning **不会**修改原始数据

- ✅ 原始伪标签：`outputs/exp_name/pseudo_labels.json`
- ✅ 清洗后标签：`outputs/exp_name/confident_learning/pseudo_labels_cleaned.json`
- ✅ 迭代使用清洗后的标签，原始文件**永远不变**

## Phase控制

在 `configs/config.yaml` 中控制：

```yaml
pipeline:
  phases:
    generate_labels: true   # Phase 1: LLM生成伪标签
    train_mae: true         # Phase 2: MAE预训练
    train_wsl: true         # Phase 3: WSL训练
    confident_learning: true # Phase 4: 标签清洗

confident_learning:
  enabled: false  # true=启用CL，false=跳过
```

## 命令行使用

### 手动运行单个阶段

```bash
# 使用特定实验的配置
python generate_labels.py --config outputs/exp_001/config.yaml
python train_mae.py --config outputs/exp_001/config.yaml
python train_wsl.py --config outputs/exp_001/config.yaml
python confident_learning.py --config outputs/exp_001/config.yaml
python evaluate.py --config outputs/exp_001/config.yaml

# 迭代配置
python evaluate.py --config outputs/exp_001/config_iter1.yaml
```

## TensorBoard

查看所有训练曲线：

```bash
# 查看整个实验（包含所有迭代）
tensorboard --logdir=outputs/your_experiment_name

# 只查看特定阶段
tensorboard --logdir=outputs/your_experiment_name/mae_pretrain
tensorboard --logdir=outputs/your_experiment_name/wsl_train_iter1
```

## 比较实验结果

### 方法1：使用Python脚本

```python
import json
from pathlib import Path

# 比较所有实验
for exp_dir in Path('outputs').iterdir():
    if exp_dir.is_dir():
        # 初始模型
        results = exp_dir / 'evaluation_results.json'
        if results.exists():
            with open(results) as f:
                data = json.load(f)
            overall = data['overall_metrics']['macro_avg']
            print(f"{exp_dir.name} (initial):")
            print(f"  AUC-ROC: {overall.get('auc_roc', 0):.3f}")
            print(f"  F1: {overall.get('f1', 0):.3f}\n")
```

### 方法2：列出所有实验

```bash
python main.py
# 选择: 2 (Continue existing experiment)
# 会列出所有已有实验
```

## 常见问题

### Q: 如何重新训练但保留原实验？

A: 创建新实验，重用伪标签：
```bash
python main.py
# 选择: 1 (Create new experiment)
# 实验名: baseline_v2
# 选择: 1 (Reuse existing labels)
```

### Q: CL迭代训练时可以修改参数吗？

A: 迭代时使用原实验的配置。如需修改参数：
1. 手动编辑 `outputs/exp_name/config.yaml`
2. 或创建新实验

### Q: 如何停止自动迭代？

A: 当提示 "Retrain with cleaned labels?" 时选择 `n`

### Q: 迭代会覆盖之前的模型吗？

A: 不会！每次迭代都有独立的文件夹：
- `mae_pretrain` → 初始
- `mae_pretrain_iter1` → 迭代1
- `mae_pretrain_iter2` → 迭代2

## 最佳实践

1. **实验命名规范**：使用描述性名称，如 `baseline_512`, `cl_iter2_lr0.001`
2. **标签复用**：同一数据集的多个实验可以共享伪标签
3. **定期清理**：删除不需要的实验释放空间
4. **记录实验**：在实验文件夹中添加 `notes.txt` 记录实验目的和结果
5. **迭代训练**：启用CL后至少运行1次迭代以评估清洗效果
