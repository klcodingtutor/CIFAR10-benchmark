import argparse
from data import get_cifar10_dataloaders, get_face_dataloaders
from models import *  # Import all model functions
from trainers.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)


    # Setup data
    if config['dataset'] == 'cifar10':
        train_loader, val_loader, _ = get_cifar10_dataloaders()
    elif config['dataset'] == 'face':
        if 'task' not in config:
            raise ValueError("Task must be specified for face dataset")
        train_loader, val_loader, _ = get_face_dataloaders(
            data_dir='./data/face',
            batch_size=config['batch_size'],
            task=config['task']
        )

    # Setup model and logger
    model_func = globals()[f'get_{config.replace("-", "_")}']
    model = model_func(
        model_name=config['model'],
        num_classes=10 if config['dataset'] == 'cifar10' else len(train_loader.dataset.label_to_idx),
        pretrained=config.get('pretrained', False),
        transfer_learning=config.get('transfer_learning', False)
    )
    logger, writer = setup_logger()

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, logger, writer)
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()