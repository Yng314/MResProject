# MedSoul: Weakly Supervised Medical Image Classification

A complete pipeline for training medical image classifiers using weakly supervised learning with LLM-generated pseudo labels.

## Pipeline Overview

```
Reports → Qwen LLM → Pseudo Labels
                         ↓
Images → MAE Pretrain → Encoder
                         ↓
    [Pseudo Labels + Encoder] → WSL Training
                         ↓
              Confident Learning → Clean Labels
                         ↓
                    (Iterate)
```

## Features

- **Phase 1**: LLM-based pseudo label generation (Qwen-max)
- **Phase 2**: Self-supervised pretraining with Masked Autoencoder (MAE)
- **Phase 3**: Two-stage weakly supervised learning
  - Linear Probe: Train classification head only
  - Fine-tune: Unfreeze top encoder layers
- **Phase 4**: Confident Learning for label noise detection and cleaning

## Setup

### 1. Environment Setup

```bash
# Activate conda environment
conda activate .\.conda

# Install dependencies (already done if you followed installation)
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file with your API key:

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

Edit `configs/config.yaml` to customize:
- Data paths and subset size
- Model hyperparameters
- Training epochs and batch sizes
- Pipeline phases to run

## Usage

### Quick Start (Run Full Pipeline)

```bash
python main.py
```

This will run all phases sequentially.

### Run Individual Phases

```bash
# Phase 1: Generate pseudo labels
python generate_labels.py

# Phase 2: MAE pretraining
python train_mae.py

# Phase 3: WSL training
python train_wsl.py

# Phase 4: Confident Learning
python confident_learning.py
```

### Iterative Training with Cleaned Labels

After Phase 4, you can retrain with cleaned labels:

1. Update `configs/config.yaml`:
   ```yaml
   llm:
     cache_file: "outputs/confident_learning/pseudo_labels_cleaned.json"
   ```

2. Retrain:
   ```bash
   python train_wsl.py
   ```

3. Optionally run CL again for further refinement

## Project Structure

```
MedSoul/
├── configs/
│   └── config.yaml           # Configuration file
├── data/
│   └── dataset.py            # Dataset classes
├── models/
│   ├── mae.py               # MAE implementation
│   └── resnet.py            # ResNet50 + Classifier
├── utils/
│   ├── qwen_api.py          # Qwen API wrapper
│   └── metrics.py           # Evaluation metrics
├── datasets/
│   └── mimic_dataset/       # MIMIC-CXR parquet files
├── outputs/                 # All outputs go here
│   ├── pseudo_labels.json
│   ├── mae_pretrain/
│   ├── wsl_train/
│   └── confident_learning/
├── generate_labels.py       # Phase 1 script
├── train_mae.py            # Phase 2 script
├── train_wsl.py            # Phase 3 script
├── confident_learning.py   # Phase 4 script
├── main.py                 # Pipeline controller
└── requirements.txt
```

## Configuration

Key settings in `configs/config.yaml`:

```yaml
data:
  num_samples: 100           # Use subset for testing (-1 for full)
  labels: [...]              # 12 disease labels
  
mae:
  epochs: 10
  batch_size: 8
  mask_ratio: 0.75
  
wsl:
  linear_probe:
    epochs: 10
    freeze_encoder: true
  fine_tune:
    epochs: 20
    unfreeze_layers: 10
    
confident_learning:
  enabled: true
  max_iterations: 3
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=outputs
```

## Outputs

- **Pseudo Labels**: `outputs/pseudo_labels.json`
- **MAE Encoder**: `outputs/mae_pretrain/encoder_best.pth`
- **WSL Models**: `outputs/wsl_train/linear_probe_best.pth`, `fine_tune_best.pth`
- **Cleaned Labels**: `outputs/confident_learning/pseudo_labels_cleaned.json`
- **Predictions**: `outputs/wsl_train/val_predictions.npz`

## Label Schema

Multi-label classification with 12 pathologies:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Fracture
- Lung Lesion
- Lung Opacity
- Pleural Effusion
- Pneumonia
- Pneumothorax
- Support Devices
- No Finding

Label values:
- `1.0`: Present
- `0.0`: Absent
- `-1.0`: Uncertain
- `null`: Not mentioned

## Notes

- First run uses 100 samples with 10 epochs for testing
- For full training, set `data.num_samples: -1` and increase epochs
- GPU memory: ~10GB for MAE, ~8GB for WSL with current batch sizes
- Confident Learning removes ~20% noisiest samples by default

## Troubleshooting

**API Errors**: Check your `.env` file and API key

**CUDA OOM**: Reduce batch size in config.yaml

**Import Errors**: Make sure conda environment is activated

**No Pretrained Encoder**: MAE training creates encoder weights. If skipped, WSL will use random initialization.

## Citation

Based on the pipeline design from your flowchart:
- **LLM**: Qwen-max (Alibaba Cloud)
- **SSL**: Masked Autoencoder (MAE)
- **WSL**: Two-stage training with pseudo labels
- **Noise Handling**: Confident Learning (cleanlab)
