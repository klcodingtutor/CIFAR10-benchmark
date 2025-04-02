# CIFAR-10 Model Benchmark Repository

This repository provides a benchmark for training and evaluating popular deep learning models (e.g., ResNet, EfficientNet) on the CIFAR-10 dataset using PyTorch.

## Repository Structure
```
cifar10_benchmark/
├── configs/              # Configuration files (YAML)
├── dataloaders/          # Data loading scripts
├── models/               # Model definitions
├── utils/                # Utility functions (training, config parsing)
├── train.py              # Main training script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Directory setup**:
   Ensure you have write permissions for the `./data` directory (for CIFAR-10 download).

## Usage
Run the training script with a configuration file:
```bash
python train.py --config configs/efficientnet_b0_cifar10.yaml
```

### Configuration
Edit or create YAML files in `configs/` to specify model, dataset, and training parameters. Example:
```yaml
model: efficientnet-b0
model_family: efficientnet
dataset: cifar10
task: classification
epochs: 10
batch_size: 64
optimizer: adam
lr: 0.001
scheduler: null
pretrained: True
transfer_learning: true
```