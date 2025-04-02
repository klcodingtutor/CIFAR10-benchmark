import argparse
from data import get_cifar10_dataloaders, get_face_dataloaders
from models import *  # Import all model functions
from trainers.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet18)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'face'], help='Dataset to use')
    parser.add_argument('--task', type=str, default=None, help='Task for face dataset (e.g., gender)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['model'] = args.model
    config['dataset'] = args.dataset
    if args.task:
        config['task'] = args.task

    # Setup data
    if config['dataset'] == 'cifar10':
        train_loader, val_loader, _ = get_cifar10_dataloaders(
            batch_size=config['batch_size'],
            partial_ratio=config.get('partial_ratio', 0.1)
        )
    elif config['dataset'] == 'face':
        if 'task' not in config:
            raise ValueError("Task must be specified for face dataset")
        train_loader, val_loader, _ = get_face_dataloaders(
            data_dir='./data',
            batch_size=config['batch_size'],
            task=config['task']
        )

    # Setup model and logger
    model_func = globals()[f'get_{args.model.replace("-", "_")}']
    model = model_func(model_name=args.model, num_classes=10 if config['dataset'] == 'cifar10' else len(train_loader.dataset.label_to_idx))
    logger, writer = setup_logger()

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, logger, writer)
    trainer.train()
    writer.close()

if __name__ == '__main__':
    main()