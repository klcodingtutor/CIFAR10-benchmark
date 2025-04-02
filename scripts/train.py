import argparse
from data.dataloaders import get_cifar10_dataloaders
from trainers.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logger
from models import *

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup data, model, and logger
    train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=config['batch_size'])

    if "resnet" in config["model"].lower():
        model = globals()[f'get_resnet'](
            model_name=config["model"],
            pretrained=config["pretrained"],
            num_classes=config["num_classes"]
        )
    elif "vgg" in config["model"].lower():
        model = globals()[f'get_vgg'](
            model_name=config["model"],
            pretrained=config["pretrained"],
            num_classes=config["num_classes"]
        )
    else:
        raise ValueError(f"Unsupported model type: {config['model']}")

    logger, writer = setup_logger()

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, logger, writer)
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()

# yaml
# model: resnet18
# epochs: 10
# batch_size: 128
# optimizer: adam
# lr: 0.001
# momentum: 0.9
# scheduler: steplr
# step_size: 30
# gamma: 0.1
# pretrained: False
# num_classes: 10