import torch
from utils.metrics import compute_accuracy

class Evaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                acc = compute_accuracy(outputs, targets)[0].item()
                total_acc += acc
        avg_acc = total_acc / len(self.test_loader)
        return avg_acc