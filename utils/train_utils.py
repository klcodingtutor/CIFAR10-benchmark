import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(trainloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def save_checkpoint(model, config, filepath):
    """Save the model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Simple test (requires other modules)
    from ..dataloaders.cifar10_loader import get_cifar10_dataloaders
    from ..models import get_model
    
    config = {'batch_size': 64, 'model': 'resnet18', 'model_family': 'resnet', 'pretrained': True}
    trainloader, testloader = get_cifar10_dataloaders(config['batch_size'])
    model = get_model(config['model'], config['model_family'], config['pretrained'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")