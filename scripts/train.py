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
    model = globals()[f'get_{config["model"].split("-")[0]}'](model_name=args.model)
    logger, writer = setup_logger()

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, logger, writer)
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()