# CIFAR-10 Benchmark Repository

This repository provides a framework for benchmarking various deep learning models on the CIFAR-10 dataset. It includes scripts for training, evaluation, and utilities to facilitate the benchmarking process.

## Directory Structure

- `checkpoints/`: This directory stores trained model weights, such as `resnet18_best.pth`. It will be populated by the `Trainer` class during training.
  
- `results/`: This directory is meant for storing benchmark results, like `benchmark.txt`. It will be populated by the `benchmark.py` script.

- `src/`: This directory contains the source code for the project.
  - `data/`: Contains data loaders and augmentations for the CIFAR-10 dataset.
  - `models/`: Contains definitions for various deep learning models like ResNet and EfficientNet.
  - `training/`: Contains the training script responsible for loading data, initializing models, and saving checkpoints.
  - `evaluation/`: Contains the evaluation script used for assessing trained models and computing performance metrics.
  - `utils/`: Contains utility functions for logging, metrics, and configuration handling.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Training a Model**: To train a model, run the training script located in `src/training/train.py`. Ensure that your data is properly set up and accessible.

2. **Evaluating a Model**: After training, you can evaluate the model using the script in `src/evaluation/evaluate.py`. This will load the model weights from the `checkpoints/` directory and compute performance metrics.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.