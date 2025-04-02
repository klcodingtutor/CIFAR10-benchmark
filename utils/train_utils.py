import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.decorator import print_args

@print_args
def train_epoch(model, trainloader, valloader, criterion, optimizer, device):
    """Train the model for one epoch and evaluate on validation set."""
    # Training phase
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
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(valloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(valloader)
    val_acc = 100. * val_correct / val_total
    
    return train_loss, train_acc, val_loss, val_acc

@print_args
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

@print_args
def save_checkpoint(model, config, filepath):
    """Save the model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, filepath)
    print(f"Model saved to {filepath}")

@print_args
def load_checkpoint(model, filepath, device):
    """Load a model checkpoint and return the model."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model, checkpoint['config']

if __name__ == "__main__":
    # Simple test (requires other modules)
    from dataloaders.cifar10_loader import get_cifar10_dataloaders
    from models import get_model
    
    config = {'batch_size': 64, 'model': 'resnet18', 'model_family': 'resnet', 'pretrained': True}
    trainloader, valloader, testloader = get_cifar10_dataloaders(config['batch_size'])
    model = get_model(config['model'], config['model_family'], config['pretrained'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test training
    train_loss, train_acc, val_loss, val_acc = train_epoch(model, trainloader, valloader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Test saving and loading
    save_checkpoint(model, config, "test_checkpoint.pth")
    model, loaded_config = load_checkpoint(model, "test_checkpoint.pth", device)
    print(f"Loaded config: {loaded_config}")