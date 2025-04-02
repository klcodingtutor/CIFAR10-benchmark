# CIFAR-10 Benchmark Project

This project provides a benchmark for training and evaluating deep learning models on the CIFAR-10 dataset. It includes implementations for various model architectures, data loading utilities, and configuration settings.

## Project Structure

- **configs/**: Contains YAML configuration files for model and training parameters.
  - `efficientnet_b0_cifar10.yaml`: Configuration for EfficientNet-B0 on CIFAR-10.

- **dataloaders/**: Contains scripts for loading datasets.
  - `cifar10_loader.py`: Implements the CIFAR-10 dataloader with data transformations.

- **models/**: Contains implementations of different model architectures.
  - `__init__.py`: Model registry for easy access to different architectures.
  - `resnet.py`: Implements ResNet variants.
  - `efficientnet.py`: Implements EfficientNet variants.

- **README.md**: Project documentation.

- **requirements.txt**: Lists the dependencies required for the project.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Configure the training parameters in the `configs/efficientnet_b0_cifar10.yaml` file.
2. Load the CIFAR-10 dataset using the provided dataloader in `dataloaders/cifar10_loader.py`.
3. Choose a model architecture from the `models` directory and train it using the specified configurations.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.