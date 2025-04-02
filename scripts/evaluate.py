from utils.config import load_config
import argparse
import torch
from models import *
from data.dataloaders import get_cifar10_dataloaders
from trainers.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on CIFAR-10')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup data and model
    _, _, test_loader = get_cifar10_dataloaders()
    model = globals()[f'get_{config.split("-")[0]}'](model_name=config.model)
    model.load_state_dict(torch.load(args.checkpoint))

    # Evaluate
    evaluator = Evaluator(model, test_loader)
    acc = evaluator.evaluate()
    print(f'Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main()