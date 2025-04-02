import argparse
import torch
from data import get_cifar10_dataloaders, get_face_dataloaders
from models import *  # Import all model functions
from trainers.evaluator import Evaluator
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on a dataset')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet18)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'face'], help='Dataset to use')
    parser.add_argument('--task', type=str, default=None, help='Task for face dataset (e.g., gender)')
    parser.add_argument('--transfer_learning', default=False, help='Use transfer learning model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    args = parser.parse_args()

    # Load config if provided, otherwise use defaults
    config = load_config(args.config) if args.config else {}
    config['model'] = args.model
    config['dataset'] = args.dataset
    if args.task:
        config['task'] = args.task

    # Setup data
    if config['dataset'] == 'cifar10':
        _, _, test_loader = get_cifar10_dataloaders()
    elif config['dataset'] == 'face':
        if 'task' not in config:
            raise ValueError("Task must be specified for face dataset")
        _, _, test_loader = get_face_dataloaders(
            data_dir='./data/face',
            batch_size=config.get('batch_size', 64),  # Default to 64 if not in config
            task=config['task']
        )

    # Setup model
    if args.transfer_learning:
        model_func = globals()[f'get_{args.model.replace("-", "_")}_transfer_learning']
        print("Using transfer learning model!")
    else:
        model_func = globals()[f'get_{args.model.replace("-", "_")}']
        print("Using standard model!")
    
    # Determine number of classes
    num_classes = 10 if config['dataset'] == 'cifar10' else len(test_loader.dataset.label_to_idx)
    model = model_func(model_name=args.model, num_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint))

    # Evaluate
    evaluator = Evaluator(model, test_loader)
    acc = evaluator.evaluate()
    print(f'Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main()