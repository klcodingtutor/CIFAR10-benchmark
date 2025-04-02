import argparse
import torch
from models import *
from data.dataloaders import get_cifar10_dataloaders
from trainers.evaluator import Evaluator
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Benchmark models on CIFAR-10')
    parser.add_argument('--models', nargs='+', required=True, help='List of models to benchmark')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='List of checkpoint paths')
    args = parser.parse_args()

    assert len(args.models) == len(args.checkpoints), "Number of models and checkpoints must match"

    # Setup data
    _, _, test_loader = get_cifar10_dataloaders()

    # Benchmark
    results = {}
    for model_name, checkpoint in zip(args.models, args.checkpoints):
        model = globals()[f'get_{model_name.split("-")[0]}'](model_name=model_name)
        model.load_state_dict(torch.load(checkpoint))
        evaluator = Evaluator(model, test_loader)
        acc = evaluator.evaluate()
        results[model_name] = acc
        print(f'{model_name}: {acc:.2f}%')

    # Save results
    with open('results/benchmark.txt', 'w') as f:
        for model, acc in results.items():
            f.write(f'{model}: {acc:.2f}%\n')

if __name__ == '__main__':
    main()