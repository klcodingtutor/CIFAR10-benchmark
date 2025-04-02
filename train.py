import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from utils.config import load_config
from utils.train_utils import train_epoch, evaluate, save_checkpoint
from dataloaders.cifar10_loader import get_cifar10_dataloaders
from models import get_model

def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="CIFAR-10 Model Benchmarking")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataloaders
    trainloader, testloader = get_cifar10_dataloaders(config['batch_size'])
    
    # Load model
    model = get_model(
        model_name=config['model'],
        model_family=config['model_family'],
        pretrained=config['pretrained'],
        transfer_learning=config['transfer_learning'],
        num_classes=10
    ).to(device)
    
    # Define loss, optimizer
    criterion = nn.CrossEntropyLoss()
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    # Optional: Add a learning rate scheduler if specified in the config
    scheduler = None
    if config.get('scheduler') == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            filepath = f"{config['model']}_{config['dataset']}_{config['task']}_best.pth"
            save_checkpoint(model, config, filepath)
    
    print(f"Training completed. Best Test Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()