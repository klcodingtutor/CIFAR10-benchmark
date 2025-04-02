import argparse
import torch
from data.dataloaders import get_cifar10_dataloaders
from models import get_resnet, get_efficientnet  # Add other imports as needed
from trainers.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet18)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    # Setup data and model
    _, _, test_loader = get_cifar10_dataloaders()
    model = globals()[f'get_{args.model.split("-")[0]}'](model_name=args.model)
    model.load_state_dict(torch.load(args.checkpoint))

    # Evaluate
    evaluator = Evaluator(model, test_loader)
    acc = evaluator.evaluate()
    print(f'Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main()