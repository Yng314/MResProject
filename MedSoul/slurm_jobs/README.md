# MedSoul Slurm 任务提交指南

## 📁 文件结构

```
slurm_jobs/
├── config_slurm.yaml         # Slurm 专用配置文件
├── run_train_wsl.sh          # WSL 训练任务
├── run_estimate_noise_rate.sh # Noise Rate 估算任务
└── run_full_pipeline.sh      # 完整 Pipeline (训练 + 估算)
```

---

## 🚀 快速开始

### Step 1: SSH 登录 GPU 集群
```bash
ssh gpucluster2.doc.ic.ac.uk
```

### Step 2: 进入项目目录
```bash
cd /vol/gpudata/yz3522-llmtest/MedSoul
```

### Step 3: 提交训练任务
```bash
# 只训练 WSL 模型 (~24小时)
sbatch slurm_jobs/run_train_wsl.sh

# 或者 完整流程 (训练 + Noise Rate 估算)
sbatch slurm_jobs/run_full_pipeline.sh
```

### Step 4: 查看任务状态
```bash
squeue --me           # 查看自己的任务
squeue --me --start   # 查看预计开始时间
```

---

## 📊 常用命令

| 命令 | 描述 |
|------|------|
| `sbatch <script.sh>` | 提交任务 |
| `squeue --me` | 查看自己的任务 |
| `scancel <job_id>` | 取消任务 |
| `cat slurm_logs/wsl_<job_id>.log` | 查看输出日志 |
| `tail -f slurm_logs/wsl_<job_id>.log` | 实时查看日志 |

---

## ⚙️ 修改配置

编辑 `slurm_jobs/config_slurm.yaml`:

```yaml
# 使用 Ground Truth 标签 (默认)
data:
  source_type: 'mimic_jpg'

# 或者使用 LLM Pseudo Labels
data:
  source_type: 'parquet'
```

---

## 📧 任务完成通知

任务完成后会自动发送邮件到 `yz3522@ic.ac.uk`

---

## ⚠️ 注意事项

1. **确认虚拟环境存在**：脚本会自动查找 `.conda/`, `venv/`, 或 `/vol/gpudata/yz3522-llmtest/venv/`
2. **日志位置**：所有日志保存在 `slurm_logs/` 目录
3. **输出位置**：训练结果保存在 `outputs/slurm_experiment/`
